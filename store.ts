// Filesystem side: config file, Pi's own compaction settings, per-session state.json, spill snapshots.
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { mergePiCompaction } from "./budget.ts";
import { DEFAULTS, mergeConfig, type Config } from "./config.ts";
import type { Elided, Kind, Spill, Tier } from "./archive.ts";
import { emptyScratch } from "./pin.ts";
import { newState, type PlanState } from "./plan.ts";

// Same resolution as Pi's getAgentDir().
export function agentDir(): string {
  const env = process.env.PI_CODING_AGENT_DIR;
  if (!env) return join(homedir(), ".pi", "agent");
  return env === "~" || env.startsWith("~/") ? join(homedir(), env.slice(1)) : env;
}

export const SPILL_ROOT = join(agentDir(), "context-budget");

function readJson(path: string): Record<string, unknown> | undefined {
  try {
    if (existsSync(path)) return JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
  } catch (err) {
    console.error(`context-budget: ignoring ${path}: ${err instanceof Error ? err.message : err}`);
  }
  return undefined;
}

export function loadConfig(): Config {
  const path = process.env.CONTEXT_BUDGET_CONFIG ?? join(agentDir(), "context-budget.json");
  const raw = readJson(path);
  return raw ? mergeConfig(raw) : { ...DEFAULTS };
}

// Pi's compaction block: global settings.json with the project's .pi/settings.json merged over it,
// the same precedence Pi's SettingsManager applies — modelOverrides merged key by key like the
// rest of the settings. Raw, unresolved: resolve one model's effective numbers with
// piCompactionFor(block, modelRef).
export function loadPiCompaction(cwd?: string): Record<string, unknown> | undefined {
  const globals = readJson(join(agentDir(), "settings.json"))?.compaction;
  const project = cwd ? readJson(join(cwd, ".pi", "settings.json"))?.compaction : undefined;
  return mergePiCompaction(globals, project);
}

export function sessionDir(sessionId: string): string {
  return join(SPILL_ROOT, sessionId);
}

function statePath(sessionId: string): string {
  return join(sessionDir(sessionId), "state.json");
}

function kindFor(key: string, e: Partial<Elided>): Kind {
  if (e.kind === "result" || e.kind === "thinking" || e.kind === "arg") return e.kind;
  return key.startsWith("think:") ? "thinking" : key.startsWith("arg:") ? "arg" : "result";
}

// A tier has to survive a Pi restart. Losing it would read a lean entry back as a citation,
// which is a promotion — the one direction the tiers must never move, since it would rewrite
// the prompt back to a larger form and move the prefix-cache miss point for nothing.
function tierFor(e: Partial<Elided>): Tier | undefined {
  return e.tier === "cite" || e.tier === "reduced" || e.tier === "lean" ? e.tier : undefined;
}

// Reads 0.2 state too: entries gain a kind, the positional thinkCut is dropped (thinking is now keyed by
// content), and a pre-0.6 entry with no tier reads as "cite". cacheMode is absent before 0.7.
export function loadState(sessionId: string): PlanState {
  try {
    const p = statePath(sessionId);
    if (existsSync(p)) {
      const raw = JSON.parse(readFileSync(p, "utf8")) as { elided?: Record<string, Partial<Elided>>; gen?: number; scratch?: PlanState["scratch"]; frozen?: unknown; advancedAtStep?: unknown };
      const elided: Record<string, Elided> = {};
      for (const [k, e] of Object.entries(raw.elided ?? {})) {
        if (!e || typeof e.id !== "string") continue;
        elided[k] = { id: e.id, kind: kindFor(k, e), path: e.path, step: e.step ?? 0, tool: e.tool ?? "tool", tokens: e.tokens ?? 0, duplicateOf: e.duplicateOf, tier: tierFor(e) };
      }
      const state: PlanState = { elided, gen: raw.gen ?? 0, scratch: raw.scratch ?? emptyScratch() };
      // Left off when it is not set, like tier and duplicateOf above: a state file written by a
      // cacheMode "off" session stays byte-for-byte what 0.6 wrote.
      if (raw.frozen === true) state.frozen = true;
      if (typeof raw.advancedAtStep === "number") state.advancedAtStep = raw.advancedAtStep;
      return state;
    }
  } catch (err) {
    console.error(`context-budget: ignoring ${statePath(sessionId)}: ${err instanceof Error ? err.message : err}`);
  }
  return newState();
}

export function saveState(sessionId: string, state: PlanState): void {
  try {
    mkdirSync(sessionDir(sessionId), { recursive: true, mode: 0o700 });
    writeFileSync(statePath(sessionId), JSON.stringify(state), { mode: 0o600 });
  } catch (err) {
    console.error(`context-budget: could not persist state: ${err instanceof Error ? err.message : err}`);
  }
}

export function readSpill(path: string | undefined): string | undefined {
  if (!path || !existsSync(path)) return undefined;
  return readFileSync(path, "utf8");
}

// The newest write anywhere in a session directory, not the directory's own mtime. Rewriting an
// existing file leaves its parent's mtime alone, and state.json is rewritten on every advance while
// snapshots are written once and never overwritten — so a long-running session whose last new
// snapshot is a month old has a directory mtime a month old even though it just wrote state.json.
// Measuring every entry instead lands on exactly the file the rewrite touched.
//
// One level deep is enough: spillFor writes its snapshots directly into the session directory, so
// there is no deeper tree whose own contents could be newer.
function lastWritten(dir: string): number {
  let newest = statSync(dir).mtimeMs;
  for (const name of readdirSync(dir)) {
    try {
      const m = statSync(join(dir, name)).mtimeMs;
      if (m > newest) newest = m;
    } catch {
      // an entry that vanished between readdir and stat cannot be the one keeping a session alive
    }
  }
  return newest;
}

// Deletes whole per-session spill directories older than maxAgeDays. Snapshots are the one thing here
// that grows without bound: every elided result, argument and thinking block of every session ever
// run, none of it reachable once its session is gone.
//
// Deliberately the most defensive function in the file: it runs on Pi's startup path, so a throw or a
// slow failure would cost a session rather than a directory. Every error is swallowed, and the ids are
// checked before age — a session that is still writing must survive even if a stale mtime says
// otherwise, and mtimes do go stale (a restored backup, a clock that stepped).
//
// Returns the number of directories removed.
export function gcArchives(maxAgeDays: number, keepSessionId?: string): number {
  let removed = 0;
  try {
    const cutoff = Date.now() - maxAgeDays * 86_400_000;
    for (const name of readdirSync(SPILL_ROOT)) {
      if (name === keepSessionId) continue;
      const dir = join(SPILL_ROOT, name);
      try {
        if (!statSync(dir).isDirectory()) continue;
        // The directory's own mtime is a cheap one-way pre-filter, not the judgement. A fresh
        // directory mtime means at least one entry was created or renamed there inside the window,
        // so keeping the directory can only ever be over-cautious; it never deletes a live session.
        // It is worth the check because the walk below is one stat per snapshot, and a session that
        // just wrote a new snapshot is the common case. An old directory mtime proves nothing — a
        // rewrite of an existing name does not move it — so that is where the walk has to run.
        if (statSync(dir).mtimeMs >= cutoff) continue;
        if (lastWritten(dir) >= cutoff) continue;
        rmSync(dir, { recursive: true, force: true });
        removed++;
      } catch {
        // one unreadable or undeletable directory does not stop the sweep
      }
    }
  } catch {
    // no spill root at all is the normal first run
  }
  return removed;
}

const safe = (s: string) => s.replace(/[^A-Za-z0-9._-]/g, "");

// Snapshot file: <step>-<tool>-<key tail>.txt, 0600, never overwritten.
export function spillFor(sessionId: string): Spill {
  return (key, tool, step, text) => {
    try {
      const dir = sessionDir(sessionId);
      mkdirSync(dir, { recursive: true, mode: 0o700 });
      const slug = safe(key).slice(-12) || "x";
      const path = join(dir, `${String(step).padStart(3, "0")}-${safe(tool).slice(0, 40) || "tool"}-${slug}.txt`);
      if (!existsSync(path)) writeFileSync(path, text, { mode: 0o600 });
      return path;
    } catch {
      return undefined;
    }
  };
}
