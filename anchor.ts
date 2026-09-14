// The provider's own token count as the estimate's anchor. Pure; no Pi imports.
//
// Everything the planner measures arrives through config.estimate() as len / charsPerToken, and a
// fixed ratio is wrong by a fifth or more between models (the spread is recorded at charsPerToken
// in config.ts). A threshold stated as a fraction of the window therefore fires at a different real
// size on every model. The provider, however, has already reported the exact size of the prompt it
// answered, so the head of the estimate is taken from that number and charsPerToken only ever scales
// the tail — the messages produced since, where the same error is a small fraction of the whole.
//
// Matches Pi's own accounting (core/compaction: calculateContextTokens, getAssistantUsage) field for
// field, so the size this extension plans against and the size Pi compacts against agree.
import type { Msg } from "./messages.ts";

export interface Usage {
  input?: number;
  output?: number;
  cacheRead?: number;
  cacheWrite?: number;
  totalTokens?: number;
}

export interface Anchor {
  idx: number;    // index of the assistant message the count came from
  tokens: number; // prompt + completion tokens the provider reported for it
}

function num(v: unknown): number {
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}

// totalTokens wins when the provider sends one, otherwise these components add up. This is Pi's
// list exactly: reasoning is not one of its terms.
export function calculateContextTokens(usage: Usage): number {
  const total = num(usage.totalTokens);
  return total > 0 ? total : num(usage.input) + num(usage.output) + num(usage.cacheRead) + num(usage.cacheWrite);
}

// The newest assistant turn whose usage can be trusted, or undefined when the session has none — a
// fresh session, or a log replayed without usage. An aborted or errored turn is skipped for the
// reason Pi skips it: its usage describes a request that never completed, so the prompt it counted
// is not the prompt the next request carries.
export function findAnchor(messages: Msg[]): Anchor | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role !== "assistant") continue;
    if (m.stopReason === "aborted" || m.stopReason === "error") continue;
    const usage = m.usage as Usage | undefined;
    if (!usage || typeof usage !== "object") continue;
    const tokens = calculateContextTokens(usage);
    if (tokens > 0) return { idx: i, tokens };
  }
  return undefined;
}
