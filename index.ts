// context-budget — keeps a long Pi session inside the window the model works best in,
// without paraphrasing anything: old thinking is archived and dropped from the prompt,
// stale tool outputs and large tool-call arguments become addressable citations
// pointing at a spill snapshot. Pure logic lives in ./plan.ts and its modules.
//
// Config: ~/.pi/agent/context-budget.json (any subset of DEFAULTS), or the file named
// by CONTEXT_BUDGET_CONFIG. Per-request stats go to CONTEXT_BUDGET_LOG when set.
import { appendFileSync } from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { budgetFor, piCompactionFor, type PiCompaction } from "./budget.ts";
import { decideCompaction, type Preparation } from "./compact.ts";
import { applySubagentThresholds, estimate, modelRefOf, resolveConfigForModel } from "./config.ts";
import type { Entry } from "./cut.ts";
import { formatPin, seedScratch, stripPin } from "./pin.ts";
import { plan, type PlanState, type Stats } from "./plan.ts";
import { gcArchives, loadConfig, loadPiCompaction, loadState, saveState, spillFor } from "./store.ts";
import { deterministicSummary } from "./summary.ts";
import { registerCtxCommand, registerPin, registerRecall } from "./tools.ts";

const NOTE =
  "\n\n## Context budget\nOlder tool outputs may appear as \"[context-budget] id=cb-…\" citations " +
  "with a head/tail preview, or as a single \"[context-budget] cb-… <tool> step N … tok archived\" line " +
  "with no preview at all, or — for a tool search — as the top hits of each query with the remaining " +
  "matches listed by name only after \"also:\". Large arguments of older tool calls appear as " +
  "\"[context-budget] id=ca-…\" citations with a head. Your own text and the user's messages are never altered; older thinking is " +
  "archived under th-<hash> and dropped from the prompt (it is re-billed on every resend). Citations " +
  "are snapshots of what was there then. To recover the exact original, call context_budget_recall " +
  "with that id — do not re-run the tool, the world may have changed. Pass list=true for the archive " +
  "catalog. Large snapshots return in chunks; use the next_offset the tool reports. For an elided " +
  "read, re-read the path only if you want the current disk contents.";

// Spill snapshots of sessions nobody can return to any more: swept once per session start, off the
// startup path. A month is longer than any session that is still worth recalling an archive from.
const ARCHIVE_MAX_AGE_DAYS = 30;

export default function (pi: ExtensionAPI) {
  // Subagent workers (Pi spawns them as `… -p --no-session --model …`) take the `subagent` block's
  // absolute thresholds instead of the main session's, at the one place the config is loaded.
  const cfg = applySubagentThresholds(loadConfig(), process.argv);
  const states = new Map<string, PlanState>();
  let last: Stats | undefined;
  let lastSessionId: string | undefined;
  let lastWindow = 131072;
  let piCompaction = loadPiCompaction(process.cwd());
  let budget = budgetFor(cfg, lastWindow, piCompactionFor(piCompaction, undefined));
  let cancelled = false;         // this extension cancelled the compaction Pi is reporting as failed
  const warned = new Set<string>();

  const stateFor = (sessionId: string): PlanState => {
    let s = states.get(sessionId);
    if (!s) states.set(sessionId, (s = loadState(sessionId)));
    return s;
  };
  const sessionIdOf = (ctx: { sessionManager?: { getSessionId?: () => string } } | undefined) =>
    ctx?.sessionManager?.getSessionId?.() ?? lastSessionId;

  pi.on("session_start", (_event, ctx) => {
    const sessionId = ctx.sessionManager.getSessionId();
    lastSessionId = sessionId;
    states.set(sessionId, loadState(sessionId));
    piCompaction = loadPiCompaction(ctx.cwd);
    // Never awaited: the sweep walks and stats every session directory, and a slow disk (or a
    // permission error) must not delay the session that just started. gcArchives swallows its own
    // failures; this catch only covers the scheduling around it.
    setTimeout(() => {
      try {
        gcArchives(ARCHIVE_MAX_AGE_DAYS, sessionId);
      } catch (err) {
        console.error(`context-budget: archive cleanup failed: ${err instanceof Error ? err.message : err}`);
      }
    }, 0).unref();
  });

  pi.on("context", (event, ctx) => {
    if (!cfg.enabled) return;
    try {
      const sessionId = ctx.sessionManager.getSessionId();
      lastSessionId = sessionId;
      const window = ctx.model?.contextWindow ?? 131072;
      lastWindow = window;
      // This request's settings: a per-model override wins over the global config for the model that
      // is actually answering, and the resolved budget follows it, so ctx.budget() and the /ctx
      // command describe the request that is about to be sent. `cfg` stays the global config.
      const modelCfg = resolveConfigForModel(cfg, modelRefOf(ctx.model));
      budget = budgetFor(modelCfg, window, piCompactionFor(piCompaction, modelRefOf(ctx.model)));
      const state = stateFor(sessionId);
      const incoming = stripPin(event.messages as never[]);
      if (modelCfg.pin && seedScratch(state.scratch, incoming, modelCfg)) saveState(sessionId, state);
      let base = 0;
      try { base = estimate(ctx.getSystemPrompt(), modelCfg); } catch { /* not every context exposes it */ }
      const { messages, stats } = plan(incoming, state, budget.cfg, window, spillFor(sessionId), base);
      last = stats;
      if (stats.advanced) saveState(sessionId, state);
      if (process.env.CONTEXT_BUDGET_LOG) {
        const elided = Object.values(state.elided).map((e) => `${e.id}:${e.step}:${e.tool}:${e.tokens}${e.path ? ":spilled" : ""}`);
        const shape = { cap: Math.round(budget.cap), trigger: budget.trigger, clamped: budget.clamped };
        appendFileSync(process.env.CONTEXT_BUDGET_LOG, JSON.stringify({ t: new Date().toISOString(), window, gen: state.gen, ...stats, ...shape, elided }) + "\n");
      }
      if (ctx.hasUI) {
        const pct = Math.round((100 * stats.ctxAfter) / window);
        ctx.ui.setStatus("ctx-budget", stats.elidedTotal > 0 || stats.squeezed ? `ctx ${pct}% −${Math.round(stats.elidedTotal / 1000)}k g${state.gen}` : pct >= 40 ? `ctx ${pct}%` : undefined);
      }
      const pin = modelCfg.pin ? formatPin(state.scratch) : undefined;
      const out = pin ? [...messages, { role: "user", content: pin }] : messages;
      return { messages: out as never[] };
    } catch (err) {
      console.error(`context-budget: falling back to full context: ${err instanceof Error ? err.message : err}`);
      return;
    }
  });

  pi.on("before_agent_start", (event) => {
    if (!cfg.enabled) return;
    let extra = NOTE;
    if (cfg.pin) extra += ' A trailing "[context-budget pin]" holds the session goal; update it with context_budget_pin.';
    if (cfg.interceptCompact) extra += " If Pi auto-compacts, this extension supplies a deterministic index instead of an LLM summary.";
    // A section instead of returning `systemPrompt`: a returned prompt is forced out as one whole
    // system message on every request, so any wording change is a full cache miss. Sections are
    // diffed and patched instead, and event.systemPrompt is read-only in 0.87 anyway.
    event.systemPromptOptions.sections["context-budget"] = extra;
  });

  pi.on("session_before_compact", (event, ctx) => {
    if (!cfg.enabled || !cfg.interceptCompact) return;
    cancelled = false;
    if (event.signal?.aborted) {
      cancelled = true;
      return { cancel: true };
    }
    const sessionId = ctx.sessionManager.getSessionId();
    lastSessionId = sessionId;
    const state = stateFor(sessionId);
    const prep = event.preparation;
    try {
      const window = ctx.model?.contextWindow ?? lastWindow;
      // Same per-model resolution as the context hook: a compaction is measured against this model's
      // window, so it has to be judged against this model's thresholds too.
      const b = budgetFor(
        resolveConfigForModel(cfg, modelRefOf(ctx.model)),
        window,
        (prep.settings as PiCompaction | undefined) ?? piCompactionFor(piCompaction, modelRefOf(ctx.model)),
      );
      const decision = decideCompaction({
        prep: prep as unknown as Preparation,
        entries: (event.branchEntries ?? []) as Entry[],
        budget: b,
        reason: event.reason,
        state,
        cfg,
        customInstructions: event.customInstructions,
      });
      if (decision.cancel) {
        cancelled = true;
        if (ctx.hasUI && !warned.has(sessionId)) {
          warned.add(sessionId);
          ctx.ui.notify(
            `context-budget: skipping compaction — it would free ~${decision.freed} tokens of the ${decision.need} needed to get ` +
            `under Pi's threshold (${b.trigger} of a ${window}-token window). Lower compaction.keepRecentTokens or ` +
            "reserveTokens in settings.json for this model; the plan keeps pruning the prompt meanwhile.",
            "warning",
          );
        }
        return { cancel: true };
      }
      if (ctx.hasUI) ctx.ui.notify(`context-budget: deterministic compaction (no LLM summary${decision.recut ? ", cut resized to the window" : ""})`, "info");
      return {
        compaction: {
          summary: decision.summary,
          firstKeptEntryId: decision.firstKeptEntryId,
          tokensBefore: prep.tokensBefore,
          details: { from: "context-budget", archived: Object.keys(state.elided).length, gen: state.gen, freed: decision.freed, recut: decision.recut },
        },
      };
    } catch (err) {
      console.error(`context-budget: compact intercept failed: ${err instanceof Error ? err.message : err}`);
      return {
        compaction: {
          summary: deterministicSummary({ tokensBefore: prep.tokensBefore }, state, cfg),
          firstKeptEntryId: prep.firstKeptEntryId,
          tokensBefore: prep.tokensBefore,
          details: { from: "context-budget", fallback: true },
        },
      };
    }
  });

  pi.on("session_before_tree", (event, ctx) => {
    if (!cfg.enabled || !cfg.interceptCompact || !event.preparation?.userWantsSummary) return;
    if (event.signal?.aborted) return { cancel: true };
    try {
      const state = stateFor(ctx.sessionManager.getSessionId());
      const msgs = (event.preparation.entriesToSummarize ?? []).flatMap((e) => (e.type === "message" ? [e.message] : []));
      const summary = deterministicSummary({ messagesToSummarize: msgs as never[] }, state, cfg);
      return { summary: { summary, details: { from: "context-budget" } } };
    } catch (err) {
      console.error(`context-budget: tree summary failed: ${err instanceof Error ? err.message : err}`);
      return;
    }
  });

  pi.on("session_compact_failed", (event, ctx) => {
    if (!cfg.enabled) return;
    if (cancelled && event.aborted) {
      cancelled = false;
      return; // our own cancel, already explained
    }
    const msg = event.errorMessage ?? "unknown";
    // Pi reports these for /compact on a session below keepRecentTokens; not a failure.
    if (/too small|Already compacted/i.test(msg)) return;
    console.error(`context-budget: Pi compaction failed (${event.reason ?? "?"}): ${msg}`);
    if (ctx.hasUI) ctx.ui.notify(`context-budget: Pi compact failed: ${msg}`, "warning");
  });

  const host = { cfg, stateFor, sessionIdOf, last: () => last, lastWindow: () => lastWindow, budget: () => budget };
  registerRecall(pi, host);
  if (cfg.pin) registerPin(pi, host);
  registerCtxCommand(pi, host);
}
