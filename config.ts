// context-budget configuration. Pure; no Pi imports.
export type CacheMode = "off" | "lagged" | "frozen";
const CACHE_MODES: readonly CacheMode[] = ["off", "lagged", "frozen"];

// typeof cannot police a string union, so this is the one key that needs to be named rather than
// compared against DEFAULTS.
const asCacheMode = (v: unknown): CacheMode | undefined => (CACHE_MODES.includes(v as CacheMode) ? (v as CacheMode) : undefined);

const isPositiveFinite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v) && v > 0;

export interface Config {
  enabled: boolean;
  startAtFraction: number;      // do nothing below this fraction of the context window
  highWaterFraction: number;    // above this, advance the plan as soon as anything is eligible
  targetFraction: number;       // squeeze target, used only when squeeze is on
  keepRecentSteps: number;      // tool results/arguments younger than this (assistant steps) are untouched; min 1
  keepThinkingSteps: number;    // thinking kept for this many most recent assistant steps
  minResultTokens: number;      // smaller tool results are never elided
  leanAfterSteps: number;       // an elided result this many steps old drops to a one-line index entry; 0 disables
  leanMinTokens: number;        // ...and results this small are left alone even then
  reduceSearch: boolean;        // reduce tool-search results to top hits + names instead of head/tail
  searchKeepTop: number;        // hits kept with their description per query block
  argMinTokens: number;         // tool-call string arguments larger than this are archived; 0 disables
  batchTokens: number;          // advance the plan only when >= this many tokens can be elided
  thinkBatchSteps: number;      // ...or when this many thinking blocks became eligible
  protectLatestReadTokens: number; // budget for keeping the latest un-superseded read per path
  stubHeadChars: number;        // citation keeps up to this many leading chars of the original
  stubTailChars: number;        // ...and up to this many trailing chars (errors always keep the tail)
  argHeadChars: number;         // archived argument keeps this many leading chars in the call
  recallLimitChars: number;     // default chunk size for addressable recall
  emergencyKeepSteps: number;   // squeeze tries to keep this many recent result steps
  emergencyKeepThinking: number;// squeeze drops thinking to this many recent steps
  scratchLimitChars: number;    // hard cap for the session pin
  squeeze: boolean;             // off by default: emergency elide down to targetFraction
  pin: boolean;                 // off by default: trailing session goal pin
  interceptCompact: boolean;    // on by default: replace Pi LLM compaction with a deterministic index
  charsPerToken: number;
  cacheMode: CacheMode;         // how often the elision boundary may move, so the prefix cache survives (plan.ts)
  cacheLagSteps: number;        // "lagged": allow the next advance this many assistant steps later
  maxPromptTokens?: number;     // absolute start threshold in estimated tokens; wins over startAtFraction
  maxHardTokens?: number;       // absolute high-water threshold in estimated tokens; wins over highWaterFraction
  modelOverrides?: Record<string, Partial<Config>>; // per "<provider>/<modelId>", merged field by field
}

export const DEFAULTS: Config = {
  enabled: true,
  startAtFraction: 0.3,
  highWaterFraction: 0.6,
  targetFraction: 0.6,
  keepRecentSteps: 8,
  keepThinkingSteps: 6,
  minResultTokens: 300,
  leanAfterSteps: 24,
  leanMinTokens: 60,
  reduceSearch: true,
  searchKeepTop: 3,
  argMinTokens: 150,
  batchTokens: 6000,
  thinkBatchSteps: 4,
  protectLatestReadTokens: 12000,
  stubHeadChars: 400,
  stubTailChars: 400,
  argHeadChars: 160,
  recallLimitChars: 24000,
  emergencyKeepSteps: 2,
  emergencyKeepThinking: 1,
  scratchLimitChars: 1500,
  squeeze: false,
  pin: false,
  interceptCompact: true,
  // Only ever scales the messages after the newest provider-reported usage (anchor.ts), so it no
  // longer has to be right — a few trailing messages at the wrong ratio move a threshold by far less
  // than a whole session did. The measured median is still worth shipping as the default.
  //
  // Was measured by fixed-effect regression over recorded sessions; the spread is real — 3.0 for
  // deepseek-v4-flash and gpt-5.6-luna, 3.6 for glm-5.3-flash, 6.5 for deepseek-v4.1-flash. An
  // earlier revision of this fork shipped 4.49, from a measurement that counted JSON.stringify of
  // the whole content block instead of the text the estimator actually reads (messages.ts textOf)
  // and used a session that had already compacted; that overstated the ratio by about a third.
  charsPerToken: 3.35,
  cacheMode: "off",
  cacheLagSteps: 8,
};

export function estimate(text: string, cfg: Config): number {
  return Math.ceil(text.length / cfg.charsPerToken);
}

// Accepts a raw JSON object: unknown keys are ignored, the pre-0.2 errorHeadChars key still works.
export function mergeConfig(raw: Record<string, unknown>): Config {
  return normalize({ ...DEFAULTS, ...acceptFields(raw) });
}

// Field-level validation in the shape of DEFAULTS: a key is kept only at the type its default has.
// modelOverrides holds Partial<Config> values, so every entry goes through this same check.
function acceptFields(raw: Record<string, unknown>): Partial<Config> {
  const out: Partial<Config> = {};
  const bag = out as unknown as Record<string, unknown>;
  for (const k of Object.keys(DEFAULTS) as (keyof Config)[]) {
    if (k === "cacheMode") continue; // a string union: typeof cannot police it, so it is read below
    const v = raw[k];
    if (typeof v === typeof DEFAULTS[k]) bag[k] = v;
  }
  // An unknown mode is dropped rather than carried in as a string no comparison in plan.ts matches,
  // which would silently turn the one key that must be "off", "lagged" or "frozen" into "off".
  const cacheMode = asCacheMode(raw.cacheMode);
  if (cacheMode) out.cacheMode = cacheMode;
  // The optional thresholds have no DEFAULTS entry to take a type from: a zero, a negative or a
  // non-finite value is a typo rather than a threshold, so it is dropped instead of coerced. The
  // finite check is the part typeof cannot do: 1e999 is a JSON number literal, parses to Infinity,
  // and would otherwise survive as a threshold no prompt can ever reach.
  if (isPositiveFinite(raw.maxPromptTokens)) out.maxPromptTokens = raw.maxPromptTokens;
  if (isPositiveFinite(raw.maxHardTokens)) out.maxHardTokens = raw.maxHardTokens;
  // An override is "any provider/modelId" -> the keys that model overrides. A malformed entry is
  // dropped whole: a half-read override is worse than none, since the user cannot tell it was lost.
  if (raw.modelOverrides && typeof raw.modelOverrides === "object" && !Array.isArray(raw.modelOverrides)) {
    const kept: Record<string, Partial<Config>> = {};
    for (const [ref, v] of Object.entries(raw.modelOverrides as Record<string, unknown>)) {
      if (v && typeof v === "object" && !Array.isArray(v)) kept[ref] = acceptFields(v as Record<string, unknown>);
    }
    if (Object.keys(kept).length) out.modelOverrides = kept;
  }
  if (raw.stubHeadChars == null && typeof raw.errorHeadChars === "number") out.stubHeadChars = raw.errorHeadChars;
  return out;
}

// Keys that are a number with a minimum: 0 and negatives mean the user meant something else, so they
// are clamped rather than accepted. Runs on whatever was merged, so it also catches a per-model
// override, whose value would otherwise be the one thing that skips the clamp.
//
// Returns a new object instead of writing into its argument. The merged config is shared for as long
// as it lives: index.ts loads one and every request and every model switch reads it, so a clamp
// applied in place would be a value the next request inherits from a config it never built.
function normalize(cfg: Config): Config {
  return {
    ...cfg,
    keepRecentSteps: Math.max(1, cfg.keepRecentSteps),
    searchKeepTop: Math.max(0, cfg.searchKeepTop),
    leanAfterSteps: Math.max(0, cfg.leanAfterSteps),
    leanMinTokens: Math.max(0, cfg.leanMinTokens),
    cacheLagSteps: Math.max(1, cfg.cacheLagSteps),
    charsPerToken: cfg.charsPerToken > 0 ? cfg.charsPerToken : DEFAULTS.charsPerToken,
    cacheMode: asCacheMode(cfg.cacheMode) ?? DEFAULTS.cacheMode,
  };
}

// Pi's canonical reference for a model. Anything unexpected — a stub without provider or id — reads as
// undefined, which resolves to the global config rather than throwing on the request path.
export function modelRefOf(model: { provider?: string; id?: string } | undefined): string | undefined {
  return model?.provider && model.id ? `${model.provider}/${model.id}` : undefined;
}

// A model's own settings, merged field by field over the global ones: tuning one key for one model
// must not force the user to restate the other twenty, and a key left out keeps tracking the global
// value. An undefined or unmatched modelRef resolves to the global config's values, so a provider
// that renames a model cannot silently turn the plan off.
//
// Always a fresh object, never the global cfg itself: the caller holds what it gets for the whole
// request, and handing back the shared object would make every later reader share any write to it.
// Shallow is enough — clamps are all top-level, and the one nested value, modelOverrides, is only
// ever read on the request path.
export function resolveConfigForModel(cfg: Config, modelRef: string | undefined): Config {
  const override = modelRef ? cfg.modelOverrides?.[modelRef] : undefined;
  return normalize(override ? { ...cfg, ...override } : cfg);
}
