import { test } from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULTS,
  PI_COMPACTION_DEFAULTS,
  boundaryStart,
  budgetFor,
  compactionTrigger,
  decideCompaction,
  deterministicSummary,
  isCutPoint,
  mergeConfig,
  modelRefOf,
  newState,
  piCompactionFrom,
  recut,
  resolveConfigForModel,
  spanFor,
  spanMessages,
  tokensToFree,
} from "./plan.ts";

const SMALL = 32_768;   // qwen3.5-4b-mlx
const BIG = 262_144;    // the local qwen3.8 lane
const pi = (over = {}) => ({ enabled: true, reserveTokens: 16384, keepRecentTokens: 32000, ...over });

const chars = (n) => "x".repeat(n);
let seq = 0;
const msg = (role, n, extra = {}) => ({ type: "message", id: `e${seq++}`, message: { role, content: [{ type: "text", text: chars(n) }], ...extra } });

// A branch of `steps` assistant/tool-result pairs after one user message: ~880 tokens a step at the
// default 4.49 chars per token, the shape the 4b session was looping on.
function branch(steps) {
  seq = 0;
  const entries = [msg("user", 200)];
  for (let i = 0; i < steps; i++) {
    entries.push(msg("assistant", 660));
    entries.push({ ...msg("toolResult", 3300), message: { role: "toolResult", toolCallId: `call${i}`, content: [{ type: "text", text: chars(3300) }] } });
  }
  return entries;
}

test("a window with room for the configured target is left exactly as configured", () => {
  const b = budgetFor(DEFAULTS, BIG, pi());
  assert.equal(b.clamped, false);
  assert.equal(b.cfg, DEFAULTS); // same object: nothing about a big window is rewritten
  assert.equal(b.trigger, BIG - 16384);
  assert.equal(b.cap, 0.6 * BIG);
});

test("a small window is clamped under Pi's threshold, and squeeze is forced on to enforce it", () => {
  const b = budgetFor(DEFAULTS, SMALL, pi());
  assert.equal(b.trigger, 16384); // Pi compacts at half of a 32k window with the default reserve
  assert.equal(b.clamped, true);
  assert.ok(b.cap < b.trigger, `cap ${b.cap} must sit under the threshold ${b.trigger}`);
  assert.equal(b.cfg.squeeze, true);
  assert.ok(b.cfg.targetFraction < DEFAULTS.targetFraction);
  assert.ok(b.cfg.highWaterFraction <= b.cfg.targetFraction);
  assert.ok(b.cfg.startAtFraction <= b.cfg.targetFraction);
  assert.equal(DEFAULTS.squeeze, false); // the caller's config is never mutated
});

test("more reserve headroom is all the 32k lane needs; the clamp then does nothing", () => {
  const b = budgetFor(DEFAULTS, SMALL, pi({ reserveTokens: 8192, keepRecentTokens: 16000 }));
  assert.equal(b.trigger, 24_576);
  assert.equal(b.clamped, false);
  assert.equal(b.cfg, DEFAULTS);
  assert.ok(0.6 * SMALL < b.trigger);
});

test("with Pi's compaction off there is no threshold to stay under", () => {
  const b = budgetFor(DEFAULTS, SMALL, pi({ enabled: false }));
  assert.equal(b.trigger, Number.POSITIVE_INFINITY);
  assert.equal(b.clamped, false);
  assert.equal(tokensToFree(99_999, b.trigger), 0);
});

test("Pi's compaction settings are read with Pi's own fallbacks", () => {
  assert.deepEqual(piCompactionFrom({}), PI_COMPACTION_DEFAULTS);
  assert.deepEqual(piCompactionFrom({ compaction: { reserveTokens: "big", keepRecentTokens: 0 } }), PI_COMPACTION_DEFAULTS);
  assert.deepEqual(piCompactionFrom({ compaction: { enabled: false, reserveTokens: 8192, keepRecentTokens: 16000 } }), {
    enabled: false,
    reserveTokens: 8192,
    keepRecentTokens: 16000,
  });
  assert.equal(compactionTrigger(pi(), SMALL), 16384);
});

test("a recut never lands on a tool result and always keeps less than Pi's cut", () => {
  const entries = branch(20);
  const cut = recut(entries, 3, 6000, DEFAULTS);
  assert.ok(cut, "a 41-entry branch has a later cut point");
  assert.ok(cut.index > 3);
  assert.equal(isCutPoint(entries[cut.index]), true);
  assert.notEqual(entries[cut.index].message.role, "toolResult");
  assert.equal(recut(entries, entries.length - 1, 6000, DEFAULTS), undefined);
});

test("the compactable span starts after what the last compaction kept", () => {
  const entries = branch(4);
  assert.equal(boundaryStart(entries), 0);
  const compacted = [...entries.slice(0, 3), { type: "compaction", id: "c1", firstKeptEntryId: entries[3].id }, ...entries.slice(3)];
  assert.equal(boundaryStart(compacted), 4);
  assert.equal(spanMessages(compacted, 4, 6).length, 2);
});

// The loop observed on 2026-09-08: a 32768-token model with reserveTokens 16384 and
// keepRecentTokens 32000. Pi compacts above 16384, its cut keeps more tokens than the window holds,
// so each compaction dropped the first few entries and the next request compacted again.
test("Pi's cut frees far less than the threshold needs; the recut frees enough", () => {
  const entries = branch(20);
  const budget = budgetFor(DEFAULTS, SMALL, pi());
  const prep = {
    firstKeptEntryId: entries[3].id,
    messagesToSummarize: spanMessages(entries, 0, 3),
    turnPrefixMessages: [],
    tokensBefore: 24_288,
  };
  const own = spanFor(prep, [], budget, tokensToFree(24_288, budget.trigger), DEFAULTS);
  assert.equal(own.recut, false);
  assert.ok(own.tokens < 7904, "Pi's own cut frees under 2k of the ~7.9k needed");

  const d = decideCompaction({ prep, entries, budget, reason: "threshold", state: newState(), cfg: DEFAULTS });
  assert.equal(d.need, 24_288 - 16_384);
  assert.equal(d.recut, true);
  assert.equal(d.cancel, false);
  assert.ok(d.freed >= d.need, `recut frees ${d.freed}, needs ${d.need}`);
  const kept = entries.findIndex((e) => e.id === d.firstKeptEntryId);
  assert.ok(kept > 3);
  assert.notEqual(entries[kept].message.role, "toolResult");
});

test("a threshold compaction that cannot get under the threshold is cancelled, not repeated", () => {
  const budget = budgetFor(DEFAULTS, SMALL, pi());
  const prep = { firstKeptEntryId: "e3", messagesToSummarize: spanMessages(branch(20), 0, 3), turnPrefixMessages: [], tokensBefore: 24_288 };
  const args = { prep, entries: [], budget, state: newState(), cfg: DEFAULTS };
  assert.equal(decideCompaction({ ...args, reason: "threshold" }).cancel, true);
  // Overflow is the turn that already failed for size; cancelling it would only fail it again.
  assert.equal(decideCompaction({ ...args, reason: "overflow" }).cancel, false);
  assert.equal(decideCompaction({ ...args, reason: "manual" }).cancel, false);
});

test("a compaction that does get under the threshold keeps Pi's own cut", () => {
  const entries = branch(20);
  const budget = budgetFor(DEFAULTS, BIG, pi());
  const prep = {
    firstKeptEntryId: entries[30].id,
    messagesToSummarize: spanMessages(entries, 0, 30),
    turnPrefixMessages: [],
    tokensBefore: BIG - 10_000,
  };
  const d = decideCompaction({ prep, entries, budget, reason: "threshold", state: newState(), cfg: DEFAULTS });
  assert.equal(d.recut, false);
  assert.equal(d.cancel, false);
  assert.equal(d.firstKeptEntryId, entries[30].id);
});

test("the session goal survives later compactions through the previous summary", () => {
  const state = newState();
  const first = deterministicSummary({ messagesToSummarize: [{ role: "user", content: "check my calendar for Wednesday" }] }, state, DEFAULTS);
  assert.match(first, /^## Goal\ncheck my calendar for Wednesday$/m);
  const second = deterministicSummary({ messagesToSummarize: [], previousSummary: first }, state, DEFAULTS);
  assert.match(second, /^## Goal\ncheck my calendar for Wednesday$/m);
  assert.doesNotMatch(second.split("## Constraints")[0], /not captured/);
});

// --- absolute thresholds, per-model overrides ---------------------------------------------------

test("the estimator default is the measured 4.49, and a real config can still set its own", () => {
  // Pinned deliberately: 3.3 was a guess that overstated token counts by 36%, so every threshold
  // fired that much earlier than configured. No other test would notice the regression, because they
  // derive their windows from the estimator instead of hardcoding token counts.
  assert.equal(DEFAULTS.charsPerToken, 4.49);
  assert.equal(mergeConfig({}).charsPerToken, 4.49);
  assert.equal(mergeConfig({ charsPerToken: 3.3 }).charsPerToken, 3.3, "a measured value still wins");
  assert.equal(mergeConfig({ charsPerToken: 0 }).charsPerToken, 4.49, "0 falls back rather than dividing by zero");
});

test("an absolute threshold is the same fraction the extension would estimate, and nothing more", () => {
  // The conversion is a division by the window and nothing else. A correction factor here would
  // double-count charsPerToken: the threshold is already stated in this extension's own estimate.
  const abs = budgetFor({ ...DEFAULTS, maxPromptTokens: 55_000, maxHardTokens: 109_000 }, 1_000_000, pi());
  const frac = budgetFor({ ...DEFAULTS, startAtFraction: 0.055, highWaterFraction: 0.109 }, 1_000_000, pi());
  assert.equal(abs.cfg.startAtFraction, 0.055);
  assert.equal(abs.cfg.highWaterFraction, 0.109);
  // Same resolved plan on every key that decides when and what to elide. The absolute keys stay on
  // the resolved config as the record of where the fractions came from; nothing reads them again.
  for (const k of Object.keys(DEFAULTS)) assert.deepEqual(abs.cfg[k], frac.cfg[k], k);
  // charsPerToken is not consulted at this layer: two configs that differ only in it convert alike.
  const slow = budgetFor({ ...DEFAULTS, charsPerToken: 3.3, maxPromptTokens: 55_000, maxHardTokens: 109_000 }, 1_000_000, pi());
  assert.equal(slow.cfg.startAtFraction, 0.055);
});

test("a config with no absolute threshold is passed through by identity, as before", () => {
  assert.equal(budgetFor(DEFAULTS, BIG, pi()).cfg, DEFAULTS);
  const b = budgetFor({ ...DEFAULTS, maxPromptTokens: 100_000 }, BIG, pi());
  assert.notEqual(b.cfg, DEFAULTS);
  assert.equal(b.cfg.startAtFraction, 100_000 / BIG);
  assert.equal(b.cfg.highWaterFraction, DEFAULTS.highWaterFraction, "the untouched key still comes from the global config");
});

test("absolute thresholds still get pulled under Pi's compaction threshold", () => {
  // Same 32k lane as the fraction test: Pi compacts at half the window, so no target near the top of
  // the window survives regardless of how it was written.
  const b = budgetFor({ ...DEFAULTS, maxPromptTokens: 20_000, maxHardTokens: 30_000 }, SMALL, pi());
  assert.equal(b.clamped, true);
  assert.equal(b.cfg.squeeze, true);
  assert.ok(b.cap < b.trigger, `cap ${b.cap} must sit under the threshold ${b.trigger}`);
  assert.ok(b.cfg.highWaterFraction <= b.cfg.targetFraction);
  assert.ok(b.cfg.startAtFraction <= b.cfg.targetFraction);
});

test("a soft threshold at or above the hard one is pulled below it rather than left contradictory", () => {
  // Nothing is both above the high water and below the start line. Left as written, the plan's
  // advance check would be unreachable; the hard threshold wins and the start line halves.
  const b = budgetFor({ ...DEFAULTS, maxPromptTokens: 200_000, maxHardTokens: 100_000 }, 1_000_000, pi());
  assert.equal(b.cfg.highWaterFraction, 0.1);
  assert.equal(b.cfg.startAtFraction, 0.05);
  assert.ok(b.cfg.startAtFraction > 0);
  assert.ok(b.cfg.startAtFraction < b.cfg.highWaterFraction);

  // Equal is the same contradiction, and it holds even on a window small enough that a fixed floor
  // would have landed above the high water.
  const eq = budgetFor({ ...DEFAULTS, maxPromptTokens: 8_000, maxHardTokens: 8_000 }, 32_768, pi());
  assert.ok(eq.cfg.startAtFraction > 0);
  assert.ok(eq.cfg.startAtFraction < eq.cfg.highWaterFraction);
});

test("mergeConfig accepts the new keys at the right types and drops the wrong ones", () => {
  const cfg = mergeConfig({
    maxPromptTokens: 55_000,
    maxHardTokens: 109_000,
    modelOverrides: {
      "anthropic/claude-sonnet-4": { keepThinkingSteps: 10, startAtFraction: 0.4, bogus: 1, maxPromptTokens: 12_000 },
      "bad/provider": "not an object",
      "bad/types": { keepThinkingSteps: "10", cacheMode: "sideways", cacheLagSteps: 0 },
    },
  });
  assert.equal(cfg.maxPromptTokens, 55_000);
  assert.equal(cfg.maxHardTokens, 109_000);
  assert.equal(cfg.modelOverrides["anthropic/claude-sonnet-4"].keepThinkingSteps, 10);
  assert.equal(cfg.modelOverrides["anthropic/claude-sonnet-4"].startAtFraction, 0.4);
  assert.equal(cfg.modelOverrides["anthropic/claude-sonnet-4"].maxPromptTokens, 12_000, "the optional keys are validated per override too");
  assert.equal("bogus" in cfg.modelOverrides["anthropic/claude-sonnet-4"], false);
  assert.equal("bad/provider" in cfg.modelOverrides, false, "a non-object override is dropped, not trusted");
  // A mistyped key is absent from the Partial rather than coerced, so the global value still applies
  // once the override is merged in. An override is a delta, not a complete config.
  assert.equal("keepThinkingSteps" in cfg.modelOverrides["bad/types"], false);
  assert.equal(resolveConfigForModel(cfg, "bad/types").keepThinkingSteps, DEFAULTS.keepThinkingSteps);
  assert.equal(cfg.modelOverrides["bad/types"].cacheMode, undefined);
  assert.equal(resolveConfigForModel(cfg, "bad/types").cacheMode, cfg.cacheMode);
  assert.equal(cfg.modelOverrides["bad/types"].cacheLagSteps, 0, "acceptFields does not clamp; it only types");
  assert.equal(resolveConfigForModel(cfg, "bad/types").cacheLagSteps, 1, "resolving an override clamps it like a global value");
});

test("a typo'd absolute threshold is refused, so it cannot silently become a 20-token window", () => {
  assert.equal(mergeConfig({ maxPromptTokens: "55000" }).maxPromptTokens, undefined);
  assert.equal(mergeConfig({ maxPromptTokens: 0 }).maxPromptTokens, undefined);
  assert.equal(mergeConfig({ maxPromptTokens: -1 }).maxPromptTokens, undefined);
  // JSON has no Infinity literal, but a large enough numeric literal parses to one, and 1e999 is a
  // number to typeof. Kept, it would be a threshold no prompt ever reaches.
  assert.equal(mergeConfig({ maxPromptTokens: 1e999 }).maxPromptTokens, undefined);
  assert.equal(mergeConfig({ maxPromptTokens: Number.POSITIVE_INFINITY }).maxPromptTokens, undefined);
  assert.equal(mergeConfig({ maxHardTokens: 1e999 }).maxHardTokens, undefined);
  assert.equal(mergeConfig({ maxPromptTokens: Number.NaN }).maxPromptTokens, undefined);
  assert.equal(mergeConfig({ maxHardTokens: null }).maxHardTokens, undefined);
  assert.equal(mergeConfig({ modelOverrides: [] }).modelOverrides, undefined);
  assert.equal(mergeConfig({}).maxPromptTokens, undefined, "unset by default, so 0.6 config files keep working");
  // A finite value still survives, so the check above does not reject the whole key.
  assert.equal(mergeConfig({ maxPromptTokens: 55_000 }).maxPromptTokens, 55_000);
});

test("an absolute threshold at or above the whole window is dropped, not turned into a no-op", () => {
  // 2e9 against a 1M window is a fraction of 2000: every prompt is below it, so the extension would
  // silently do nothing — the failure mode that looks like a working config file.
  const huge = budgetFor({ ...DEFAULTS, maxPromptTokens: 2e9, maxHardTokens: 2e9 }, 1_000_000, pi());
  assert.equal(huge.cfg.startAtFraction, DEFAULTS.startAtFraction, "the fraction the bad key overrode still applies");
  assert.equal(huge.cfg.highWaterFraction, DEFAULTS.highWaterFraction);
  assert.equal(huge.cfg.maxPromptTokens, 2e9, "the value itself is kept as the record of what was asked");
  // Exactly the window is the same no-op — nothing is above a full window and still actionable.
  const exact = budgetFor({ ...DEFAULTS, maxPromptTokens: 1_000_000 }, 1_000_000, pi());
  assert.equal(exact.cfg.startAtFraction, DEFAULTS.startAtFraction);
  // One token under it is still a threshold, so the cutoff does not swallow workable values. It
  // then runs into the soft-above-hard rule — a start line at 0.999999 with the high water at 0.6
  // would never open the batch check — and is pulled to half the high water like any other.
  const under = budgetFor({ ...DEFAULTS, maxPromptTokens: 999_999, maxHardTokens: 999_999 }, 1_000_000, pi());
  assert.equal(under.cfg.highWaterFraction, 0.999999);
  assert.equal(under.cfg.startAtFraction, 0.999999 / 2);
});

test("resolveConfigForModel merges one model's fields over the global config, key by key", () => {
  const cfg = mergeConfig({
    startAtFraction: 0.3,
    cacheMode: "lagged",
    modelOverrides: { "local/qwen3.8": { startAtFraction: 0.055, maxHardTokens: 109_000 } },
  });
  const local = resolveConfigForModel(cfg, "local/qwen3.8");
  assert.equal(local.startAtFraction, 0.055);
  assert.equal(local.maxHardTokens, 109_000);
  // Field-level, not a whole-block replacement: one key tuned for one model must not reset the rest.
  assert.equal(local.cacheMode, "lagged");
  assert.equal(local.highWaterFraction, cfg.highWaterFraction);
  assert.equal(local.keepRecentSteps, cfg.keepRecentSteps);
  // An unmatched or missing model resolves to the global config's values, so the common path is
  // unchanged — but as its own object, never as the shared cfg itself (see the next test).
  assert.deepEqual(resolveConfigForModel(cfg, "other/model"), cfg);
  assert.deepEqual(resolveConfigForModel(cfg, undefined), cfg);
  const bare = mergeConfig({ modelOverrides: { "local/qwen3.8": { startAtFraction: 0.055 } } });
  assert.equal(resolveConfigForModel(bare, "local/qwen3.8").keepRecentSteps, bare.keepRecentSteps);
});

test("resolving a model never hands back the shared global config, and never writes to it", () => {
  // index.ts loads one config and reads it for every request, while resolveConfigForModel runs per
  // request. Returning the shared object would let any later write — a clamp, a caller that decides
  // to adjust a threshold for this model — follow every other request and model, which is how one
  // request's tuning shows up in the next one's plan.
  const cfg = mergeConfig({ cacheMode: "lagged", modelOverrides: { "local/qwen3.8": { startAtFraction: 0.055 } } });
  const snapshot = JSON.parse(JSON.stringify(cfg));
  assert.notEqual(resolveConfigForModel(cfg, "local/qwen3.8"), cfg);
  assert.notEqual(resolveConfigForModel(cfg, "other/model"), cfg);
  assert.notEqual(resolveConfigForModel(cfg, undefined), cfg);
  // Two models in a row, the second with no override at all: the global must be untouched, and the
  // two resolutions must not see each other.
  const first = resolveConfigForModel(cfg, "local/qwen3.8");
  const second = resolveConfigForModel(cfg, "unlisted/model");
  assert.equal(first.startAtFraction, 0.055);
  assert.equal(second.startAtFraction, cfg.startAtFraction, "the unlisted model keeps the global value");
  first.startAtFraction = 0.9;
  assert.equal(second.startAtFraction, cfg.startAtFraction, "the second resolution did not inherit the first's write");
  assert.deepEqual(cfg, snapshot, "and the global config is byte-for-byte what it was");
});

test("the model reference is read defensively: a stub without provider or id falls back to global", () => {
  assert.equal(modelRefOf({ provider: "local", id: "qwen3.8" }), "local/qwen3.8");
  assert.equal(modelRefOf({ provider: "local" }), undefined);
  assert.equal(modelRefOf({ id: "qwen3.8" }), undefined);
  assert.equal(modelRefOf({}), undefined);
  assert.equal(modelRefOf(undefined), undefined);
  // and the fallback is the global config's values, not a throw on the request path
  const cfg = mergeConfig({ modelOverrides: { "local/qwen3.8": { startAtFraction: 0.055 } } });
  assert.deepEqual(resolveConfigForModel(cfg, modelRefOf({ id: "qwen3.8" })), cfg);
});
