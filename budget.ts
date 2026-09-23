// Where this plan's budget has to sit relative to Pi's own compaction threshold. Pure; no Pi imports.
//
// Pi compacts when the prompt passes contextWindow - reserveTokens, and its cut point keeps
// keepRecentTokens of the tail (core/compaction: shouldCompact, findCutPoint). Both fields accept
// a per-model override (Pi 0.86+ compaction.modelOverrides), resolved field by field: the model's
// override wins over the ordinary setting, which wins over Pi's default. On a 262144-token
// window the default 16384 reserve puts the threshold at 94%, far above this extension's 60%
// target, and the two never meet. On a 32768-token model the same reserve puts it at 50%, below
// the target: Pi then compacts on every request however well the plan is doing. If
// keepRecentTokens is also larger than the window, the cut point keeps everything, each compaction
// frees a few hundred tokens and the next request compacts again.
import type { Config } from "./config.ts";

export interface PiCompaction {
  enabled: boolean;
  reserveTokens: number;
  keepRecentTokens: number;
}

// Pi's own fallbacks (core/settings-manager.js) for keys settings.json leaves out.
export const PI_COMPACTION_DEFAULTS: PiCompaction = { enabled: true, reserveTokens: 16384, keepRecentTokens: 20000 };

const MAX_HEADROOM = 4096;        // tokens left between the plan's cap and Pi's threshold
const MIN_TARGET_FRACTION = 0.2;  // never aim below this, however little reserve leaves us
const MIN_KEEP_TOKENS = 1000;

// The compaction block of a settings.json, before any per-model resolution. modelOverrides is
// keyed by "provider/id" and may override reserveTokens and keepRecentTokens per model; enabled
// stays global (Pi has no per-model switch for compaction itself).
type CompactionBlock = Record<string, unknown>;

const asBlock = (v: unknown): CompactionBlock | undefined =>
  v && typeof v === "object" && !Array.isArray(v) ? (v as CompactionBlock) : undefined;

const token = (v: unknown, dflt: number): number => (typeof v === "number" && v >= 0 ? v : dflt);

// Merge Pi's global and project compaction blocks the way SettingsManager's deepMergeSettings
// does: plain objects merge recursively, everything else is replaced — so modelOverrides merge
// key by key and each model entry merges field by field. A global entry for a model keeps
// working when the project file only sets one of its tokens, and a global model entry survives
// a project file that only sets the ordinary (model-less) fields.
export function mergePiCompaction(globals: unknown, project: unknown): CompactionBlock | undefined {
  const g = asBlock(globals);
  const p = asBlock(project);
  if (!g) return p;
  if (!p) return g;
  const merged: CompactionBlock = { ...g, ...p };
  const gOver = asBlock(g.modelOverrides);
  const pOver = asBlock(p.modelOverrides);
  if (gOver || pOver) merged.modelOverrides = mergeOverrides(gOver, pOver);
  return merged;
}

// modelOverrides merge key by key, and each model entry field by field, the same recursion
// deepMergeSettings applies: the project's entry keeps the global entry's other token.
function mergeOverrides(g: CompactionBlock | undefined, p: CompactionBlock | undefined): CompactionBlock {
  if (!g) return p ?? {};
  if (!p) return g;
  const merged: CompactionBlock = { ...g };
  for (const [key, value] of Object.entries(p)) {
    const base = asBlock(merged[key]);
    const next = asBlock(value);
    merged[key] = base && next ? { ...base, ...next } : value;
  }
  return merged;
}

// The compaction settings that apply to one model, resolved field by field the way Pi's
// getCompactionTokenSetting does: modelOverrides["provider/id"][field] ?? compaction[field] ??
// Pi's default. A value that is not a non-negative number falls back at its own level, so a
// typo'd override cannot hide a valid ordinary setting. An absent model resolves to the ordinary
// fields everywhere, which is exactly the pre-modelOverrides behaviour.
export function piCompactionFor(raw: unknown, modelRef: string | undefined): PiCompaction {
  const block = asBlock(raw) ?? {};
  const override = (modelRef ? asBlock(asBlock(block.modelOverrides)?.[modelRef]) : undefined) ?? {};
  return {
    enabled: typeof block.enabled === "boolean" ? block.enabled : PI_COMPACTION_DEFAULTS.enabled,
    reserveTokens: token(override.reserveTokens, token(block.reserveTokens, PI_COMPACTION_DEFAULTS.reserveTokens)),
    keepRecentTokens: token(override.keepRecentTokens, token(block.keepRecentTokens, PI_COMPACTION_DEFAULTS.keepRecentTokens)),
  };
}

// The settings-wrapper form of piCompactionFor with no model: global settings only.
export function piCompactionFrom(raw: unknown): PiCompaction {
  return piCompactionFor((raw as { compaction?: unknown } | undefined)?.compaction, undefined);
}

// The prompt size above which Pi runs a compaction. Infinity when Pi's compaction is off.
export function compactionTrigger(pi: PiCompaction, contextWindow: number): number {
  return pi.enabled && contextWindow > 0 ? Math.max(0, contextWindow - pi.reserveTokens) : Number.POSITIVE_INFINITY;
}

// An absolute threshold is stated in real tokens, the same unit the estimator now returns: the head
// of every measurement is the provider's own count for the newest answered request (anchor.ts), so
// `maxPromptTokens: 55000` means 55000 tokens of prompt whatever charsPerToken is set to. Dividing
// by contextWindow is therefore the whole conversion, and both sides of the comparison are in the
// provider's unit. (On a session with no usage yet there is nothing to anchor to and the threshold
// is compared against a chars-based estimate; that is the first request of a session, where the
// prompt is small and the choice of threshold is not yet close.)
//
// A fraction of 1 or more is not a threshold either: nothing is above a whole window and still a
// request the planner can act on before the provider rejects it. Left as written, `maxPromptTokens:
// 2e9` on a 1M window would resolve to startAtFraction 2000 and the extension would do nothing at
// all — the hardest failure to diagnose of any config value, because the config file looks
// deliberate. Dropped like every other unusable value (see acceptFields in config.ts), which hands
// the guard back to the fraction it overrides.
function fractionFrom(absolute: number | undefined, contextWindow: number): number | undefined {
  if (absolute == null || !(absolute > 0) || !(contextWindow > 0)) return undefined;
  const fraction = absolute / contextWindow;
  return fraction < 1 ? fraction : undefined;
}

// The configured thresholds with the absolute ones applied over them. Returns undefined when neither
// absolute key is set, so the common path passes the caller's own config object through untouched
// rather than copying a config nothing changed.
//
// A soft threshold at or above the hard one is a contradiction — nothing is both above the high water
// and below the start line — and left alone it would put the plan in a state where the batch check
// in plan.ts can never open. The hard threshold wins and the start line is pulled below it, so the
// plan keeps pruning to the number the user actually asked for.
function absoluteThresholds(cfg: Config, contextWindow: number): { startAtFraction: number; highWaterFraction: number } | undefined {
  const startAt = fractionFrom(cfg.maxPromptTokens, contextWindow);
  const highWater = fractionFrom(cfg.maxHardTokens, contextWindow);
  if (startAt == null && highWater == null) return undefined;
  const highWaterFraction = highWater ?? cfg.highWaterFraction;
  const configured = startAt ?? cfg.startAtFraction;
  // Half the high water, the same relation budgetFor's own clamp uses (target / 2 for startAt), so
  // the fallback is always > 0 and always < highWaterFraction however small the user's hard
  // threshold is. A floor like MIN_TARGET_FRACTION would instead sit above the high water on a small
  // enough window, which is the contradiction this branch exists to avoid.
  return { startAtFraction: configured < highWaterFraction ? configured : highWaterFraction / 2, highWaterFraction };
}

export interface Budget {
  cfg: Config;        // the plan config to actually use for this window
  clamped: boolean;   // true when Pi's threshold, not the configured target, set the cap
  trigger: number;    // Pi compacts above this many tokens
  cap: number;        // the plan aims to stay at or below this many tokens
  keepTokens: number; // tail to keep when we have to pick a cut point ourselves
}

// Unchanged on any window where the configured target already sits below Pi's threshold, which is
// every window large enough that reserveTokens is a small share of it. Below that, the target is
// pulled under the threshold and squeeze is turned on, because a target the plan will not enforce
// leaves Pi compacting on every request.
export function budgetFor(cfg: Config, contextWindow: number, pi: PiCompaction): Budget {
  const trigger = compactionTrigger(pi, contextWindow);
  const keepTokens = Math.max(MIN_KEEP_TOKENS, Math.min(pi.keepRecentTokens, Math.round((trigger - headroom(contextWindow)) / 2)));
  const absolute = absoluteThresholds(cfg, contextWindow);
  const configured = absolute ? { ...cfg, ...absolute } : cfg;
  const plain = { cfg: configured, clamped: false, trigger, cap: configured.targetFraction * contextWindow, keepTokens };
  if (!cfg.enabled || !(contextWindow > 0) || !Number.isFinite(trigger)) return plain;
  const capFraction = (trigger - headroom(contextWindow)) / contextWindow;
  if (capFraction >= configured.targetFraction) return plain;
  const target = Math.max(capFraction, MIN_TARGET_FRACTION);
  return {
    cfg: {
      ...configured,
      targetFraction: target,
      highWaterFraction: Math.min(configured.highWaterFraction, target),
      startAtFraction: Math.min(configured.startAtFraction, target / 2),
      squeeze: true,
    },
    clamped: true,
    trigger,
    cap: target * contextWindow,
    keepTokens,
  };
}

function headroom(contextWindow: number): number {
  return Math.min(Math.round(0.1 * contextWindow), MAX_HEADROOM);
}

// Tokens a compaction has to free for the prompt to come back under the threshold that fired it.
// Below this the same decision is made again next request: Pi's cut point is anchored to
// keepRecentTokens from the end of the branch, so it does not move on its own.
export function tokensToFree(tokensBefore: number, trigger: number): number {
  return Number.isFinite(trigger) ? Math.max(0, tokensBefore - trigger) : 0;
}
