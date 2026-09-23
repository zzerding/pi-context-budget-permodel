# pi-context-budget-permodel

> **This is a fork.** It is based on
> [Don-Works/pi-context-budget](https://github.com/Don-Works/pi-context-budget)
> by MCPlexer Engineering, at upstream commit `8078f60` (v0.6.0). The upstream
> project is the original work; this fork adds the changes listed under
> [Fork changes](#fork-changes) below. Licensed AGPL-3.0-or-later, same as
> upstream.

A [Pi](https://pi.dev) coding-agent extension that keeps a long session inside
the context window a model actually works well in. No LLM calls, nothing
paraphrased: the model's own text and the user's messages are never altered.
Anything removed from the prompt is snapshotted to an addressable archive the
model can recall exactly.

## Why

Pi's built-in compaction waits until the context passes
`contextWindow - reserveTokens`, then replaces everything except the most
recent `keepRecentTokens` with one LLM-written summary. Three things fill the
window long before that on a local or self-hosted model:

- **Thinking from earlier steps.** With an OpenAI-compatible endpoint Pi
  re-sends every earlier assistant step's `reasoning_content`, and a chat
  template that keeps it bills it on every request. In agentic sessions
  measured against a vLLM-served Qwen3 model this was about a third of the
  prompt.
- **Tool outputs the model has already reasoned about.** Shell output, file
  reads and search results stay verbatim forever.
- **Tool-call arguments.** The content of every `write`, the old and new text
  of every `edit`, every script passed to `bash` or `mcpx_exec` is re-sent as
  part of the assistant message that made the call.

Pi's own auto-compaction then fails on reasoning models: it asks the same
model to write a summary, thinking tokens count against `maxTokens`,
generation hits the cap, and you get `Summarization failed: generation hit
the token cap and the summary is incomplete`. The session keeps growing.

Replaying two recorded sessions through the planner at a 100k window:

| session | window | peak prompt | cumulative prefill | plan advances |
|---|---|---|---|---|
| 288 requests | 131072 | 86k → 57k | 13.6M → 10.0M | 47 → 37 |
| 324 requests | 131072 | 121k → 81k | 21.2M → 14.9M | 108 → 39 |
| 288 requests | 32768 | 77k → 48k | 9.8M → 6.1M | 246 → 220 |

(against no pruning at all, the same 288-request session peaks at 321k.)

The default target is 60% of the window, so Pi's summarizer should rarely
need to run. If it does, this extension supplies a deterministic index
instead.

Lossless here means *recoverable*, not *still in the prompt*. 2026 work on
addressable recall (ARC), VISTA, and structurally lossless trimming all make
the same split: keep a bounded active view, keep every original in an
append-only store, recover by id rather than by re-running the tool or by
embedding search. Re-running is not lossless (`git status`, logs, and files
change). Summarising is not lossless (the paraphrase cannot be inverted).

## What it does

On every request (Pi's `context` hook, which sees a copy of the messages and
never touches the session file):

1. **Thinking** blocks of assistant steps older than `keepThinkingSteps` are
   dropped from the prompt (they would be re-billed every turn) and written
   to the archive under `th-<hash>`. The id is a hash of the text, so a
   compaction or branch switch that renumbers the steps changes nothing.
2. **Tool results** older than `keepRecentSteps` and larger than
   `minResultTokens` become a citation under `cb-<id>`: tool, step, size,
   head and tail of the original. A citation never keeps more than about 40%
   of a short result; errors always keep the configured tail, which is where
   compiler and stack-trace needles usually sit. A repeated call with the
   same tool and arguments marks the older citation as the same call as the
   later one.
   Results have three tiers and only ever move downwards, so the prompt never
   thrashes between forms:
   - **`reduced`** — a projection of the result's own structure rather than a
     window onto its text. A tool-search result keeps the top `searchKeepTop`
     hits of each query with their descriptions and lists the rest by name
     after `also:`; the `## Code API` signature dump is dropped. Because this
     is the same list and not a summary of it, it does not wait for
     `keepRecentSteps` — one step of age is enough.
   - **`cite`** — the head/tail citation above. The default for unstructured
     output.
   - **`lean`** — one index line, no preview, once a result is
     `leanAfterSteps` old and nothing has recalled it:
     `[context-budget] cb-a1b2c3d4  bash  step 41  1243 tok archived; recall
     by id.` The call it answers is still beside it in the prompt, which for a
     shell result is the command itself. A result the model passed to
     `context_budget_recall` keeps its citation instead. The lean line also
     pays for itself below `minResultTokens`, which a citation never did, so
     old results above `leanMinTokens` are leaned whether or not they were
     ever citation-eligible.
3. **Tool-call arguments** older than `keepRecentSteps` and larger than
   `argMinTokens` are archived under `ca-<id>` and replaced inside the call
   with a one-line citation and a short head. The rest of the assistant
   message is untouched.
4. The **latest un-superseded `read` of each path** is kept verbatim within
   `protectLatestReadTokens`, so an edit does not force a re-read. A later
   `edit` or `write` of that path releases it; the snapshot still holds the
   file as of that step.
5. The **latest assistant step is never elided.** Its results are what the
   model is about to read.
6. The plan **only grows, and advances in batches**: when at least
   `batchTokens` can be elided, when `thinkBatchSteps` thinking blocks became
   eligible, or when anything is eligible above `highWaterFraction`. Between
   advances the serialized prefix is byte-identical, so a server prefix cache
   (vLLM `--enable-prefix-caching`, llama.cpp cache) keeps hitting. Nothing
   happens below `startAtFraction` of the window.

On by default:

7. **`interceptCompact`**: `session_before_compact` and `/tree` return a
   deterministic archive index instead of asking the session model to
   summarise. This is the fix for `generation hit the token cap and the
   summary is incomplete` on reasoning models.

Opt-in (default **false**):

8. **`squeeze`**: if the sent prompt is still above `targetFraction`, elide
   more, including recent and protected results, until it is at or below
   the cap. Measured against the exact citations that will be sent.
9. **`pin`**: a trailing `[context-budget pin]` for the session goal. Update
   with `context_budget_pin`.

The model recovers a snapshot with `context_budget_recall` (id from the
citation, or `list=true` for the catalog). Large snapshots page via
`offset` / `next_offset`. A short paragraph is appended to the system prompt
so the model knows what a citation means.

Snapshots live under `~/.pi/agent/context-budget/<sessionId>/` as 0600
files; plan state is persisted next to them so a Pi restart does not forget
the archive.

Not elided: user messages, assistant text, results that carry an image,
`!` shell output, and anything in the latest step.

## Pi's own compaction thresholds

Pi compacts when the prompt passes `contextWindow - reserveTokens`, and its cut
point keeps `keepRecentTokens` of the tail. Both live in the `compaction` block
of `~/.pi/agent/settings.json`, and since Pi 0.86 either field can also be set
per model under `compaction.modelOverrides["provider/id"]`; the override wins
over the ordinary setting, which wins over Pi's default. A project
`.pi/settings.json` sets the ordinary fields for every model in it, while a
global per-model value beats the project's ordinary fallback for that model.

- On a 262144-token window the default 16384 reserve puts the threshold at 94%
  of the window, above the 60% target, and the two never meet.
- On a 32768-token model the same reserve puts it at 50%, below the target. Pi
  then compacts on every request however well the plan is doing.
- When `keepRecentTokens` is larger than what the window holds, the cut keeps
  the whole branch: the compaction frees a few hundred tokens, and the next
  request compacts again. Observed on a 4B model at 32768 with
  `keepRecentTokens: 32000` — three compactions in 117 seconds, each dropping
  about ten of 136 entries.

The extension reads that block (a project `.pi/settings.json` merged over the
global file) and, for the window in use:

1. Pulls its cap under Pi's threshold when the configured `targetFraction` sits
   above it, and turns `squeeze` on so the cap is enforced. Where the target
   already fits, the configured values are used unchanged and nothing is
   rewritten.
2. Resizes the compaction cut when Pi's own would free less than the compaction
   has to free, choosing a cut point by Pi's rule: any context-visible message
   except a tool result, which stays with the call it answers.
3. Cancels a threshold compaction that still cannot get under the threshold,
   instead of writing an index entry every turn. Manual and overflow
   compactions always run — overflow is the turn that already failed for size.

`/ctx` prints the cap, Pi's threshold, and whether the cap was clamped. Two
settings keep the layers out of each other's way: `keepRecentTokens` below
`contextWindow - reserveTokens` for the *smallest* model you run, and
`reserveTokens` no more than about a quarter of that window. Without per-model
overrides those ordinary values have to hold for every model at once; when the
models disagree by more than a little, set `compaction.modelOverrides` for the
small model instead of degrading the large one.

## Install

```bash
pi install git:github.com/zzerding/pi-context-budget-permodel
```

or for one project: `pi install -l git:github.com/zzerding/pi-context-budget-permodel`.
It needs nothing beyond Pi itself; the tests and replay harness use Node 23+
for built-in TypeScript type stripping.

## Configure

Copy `context-budget.example.json` to `~/.pi/agent/context-budget.json` and
edit; every key is optional and the example holds the defaults.
`CONTEXT_BUDGET_CONFIG=<file>` points at a different file, `"enabled": false`
switches the extension off. Keys are checked by type; a mistyped value falls
back to the default.

| key | default | meaning |
|---|---|---|
| `startAtFraction` | 0.3 | the start line, not a cap: below this fraction of the window the plan does nothing at all |
| `highWaterFraction` | 0.6 | the high-water line: above it the batch gate is skipped and anything eligible is elided at once |
| `interceptCompact` | true | replace Pi's LLM compaction with a deterministic archive index (false to use Pi's summarizer) |
| `squeeze` | false | if true, elide further — past `keepRecentSteps`, into protected results — until the sent prompt is at or below `targetFraction`; the only key that gives `targetFraction` any effect |
| `pin` | false | if true, inject a trailing session-goal pin |
| `targetFraction` | 0.6 | the squeeze cap as a fraction of the window — a ceiling, not a compression ratio; inert unless `squeeze` is true |
| `keepRecentSteps` | 8 | results and arguments younger than this many assistant steps are untouched (minimum 1) |
| `keepThinkingSteps` | 6 | thinking kept for this many most recent steps |
| `minResultTokens` | 300 | smaller results are never cited (they can still be leaned) |
| `leanAfterSteps` | 24 | an un-recalled result this many steps old drops to a one-line index entry; 0 disables |
| `leanMinTokens` | 60 | ...and results this small are left alone even then |
| `reduceSearch` | true | reduce tool-search results to their top hits plus names |
| `searchKeepTop` | 3 | hits kept with their description, per query block |
| `argMinTokens` | 150 | smaller tool-call arguments are never elided; 0 disables argument archiving |
| `batchTokens` | 6000 | advance only when at least this much can be elided in one move |
| `thinkBatchSteps` | 4 | or when this many thinking blocks became eligible at once |
| `protectLatestReadTokens` | 12000 | budget for keeping the latest read per path |
| `stubHeadChars` | 400 | a result citation keeps up to this many leading chars |
| `stubTailChars` | 400 | …and up to this many trailing chars |
| `argHeadChars` | 160 | an argument citation keeps this many leading chars |
| `recallLimitChars` | 24000 | default chunk size for `context_budget_recall` |
| `emergencyKeepSteps` | 2 | squeeze tries to keep this many recent result steps |
| `emergencyKeepThinking` | 1 | squeeze drops thinking to this many recent steps |
| `scratchLimitChars` | 1500 | hard cap for the session pin |
| `charsPerToken` | 3.35 | scales the messages after the newest provider-reported usage; timing only, so it no longer needs calibrating |
| `cacheMode` | off | how often the elision boundary may move: `off`, `lagged`, `frozen` |
| `cacheLagSteps` | 8 | `lagged`: minimum assistant steps between two advances — a brake that can only make pruning rarer, never sooner (minimum 1) |
| `maxPromptTokens` | unset | absolute start line in real tokens; wins over `startAtFraction`. Converted to a fraction of the window, so the same number is a different share of every model, and it is never a cap |
| `maxHardTokens` | unset | absolute high-water line in real tokens; wins over `highWaterFraction` |
| `modelOverrides` | unset | per-model settings, keyed `"<provider>/<modelId>"` |

### What actually happens on a request

The keys above are easier to set once the order they are consulted in is
explicit. Every request is tested against the same gates, and **the first three
all have to let it through before a fourth does any eliding** (`plan.ts`):

```text
1. effective size >= startAtFraction x
   window (already-elided savings
   discounted)?                              no  -> send the prompt unchanged
                                             yes -> 2
2. cacheHolds(): inside a `lagged` cooldown
   or a `frozen` lock?                       yes -> send the prompt unchanged
                                             no  -> 3
3. batch gate: >= batchTokens eligible,
   or >= thinkBatchSteps thinking blocks,
   or (above highWaterFraction and
       anything at all is eligible)?         no  -> send the prompt unchanged
                                             yes -> 4
4. elide in one batch: thinking older than
   keepThinkingSteps, results older than
   keepRecentSteps, arguments over argMinTokens
   -- then record advancedAtStep for gate 2

then, only if `squeeze` is true:
5. still above targetFraction x window ? -> elide further, past the
   age lines and into protected results, until under it
```

Two consequences are worth internalising, because both are counter-intuitive
and both produce "my threshold is not being respected" reports.

**A threshold is a start line, never a cap.** `startAtFraction` and
`maxPromptTokens` say *when the plan may begin*, not *how large the prompt may
grow*. The only ceiling in the extension is `targetFraction`, and it is inert
unless `squeeze` is true. So a config of `maxPromptTokens: 32000` with
`squeeze: false` on a 1,048,576-token window means "start tidying once the
prompt passes 32000 tokens — and then allow it to grow to the whole window",
which is very likely not what was meant. On that window 32000 tokens is **3%**
of the room available; the plan will cross it early and then have nothing left
to do, because the default `targetFraction` of 0.6 puts the only real ceiling
at 629,145 tokens. The prompt then grows until Pi's own compaction fires, or
until the session ends — with the plan sitting idle in the `batch gate` the
whole way, which looks exactly like a broken extension and is in fact the
documented behaviour of those two keys.

To make an absolute figure an actual ceiling, all three keys have to agree:

```json
{
  "maxPromptTokens": 32000,
  "squeeze": true,
  "modelOverrides": {
    "newapi/deepseek-v4.1-flash": { "targetFraction": 0.0305 }
  }
}
```

`targetFraction` is the ceiling as a fraction of that model's window
(`32000 / 1048576 = 0.0305`), and `squeeze: true` is what gives it force.
Note the cost before reaching for this: keeping a 1M-token model under 32k
means the boundary moves on nearly every request, which is the opposite of what
the caching keys below exist to buy. On a very large window, "hold a tight
absolute budget" and "keep the prefix cache warm" are close to mutually
exclusive; pick which one the deployment actually needs.

**Eligible is not the same as done.** Crossing the start line only starts the
counter. Between two advances the prompt normally grows, because gate 3 waits
for `batchTokens` to accumulate and gate 2 forces the wait in `lagged` mode.
The context curve is therefore a staircase — grow, elide in a batch, grow
again — not a line held at the threshold. Item 6 under
[What it does](#what-it-does) is the design reason: between advances the
serialized prefix is byte-identical, which is the whole point of the batch.
A sawtooth that peaks well above the start line is the plan working, not
failing.

### `cacheLagSteps` and prefix caching

`cacheLagSteps` (default 8, only read when `cacheMode` is `lagged`) is the
minimum number of assistant steps between two advances. It is a **brake**: it
can only make pruning rarer, never sooner, and no value of it can relax a tier
back upwards.

It exists because a prompt cache is a cache of the *common prefix*, and this
extension elides by rewriting **early** messages. Each advance therefore
invalidates the whole cached prefix — the miss is exactly the tokens that were
already there, so the smarter the server's cache the more an over-eager plan
throws away (vLLM `--enable-prefix-caching`, llama.cpp's KV cache). The lag
trades a little peak context for fewer full re-prefills of the session.

Two things it is *not*: it does not bound the prompt size (only
`targetFraction` + `squeeze` do that), and it is not what holds the prompt back
in most sessions. Whenever gate 3 is the binding constraint, `cacheLagSteps: 1`
and `cacheLagSteps: 8` produce byte-identical plans — the lag never gets a
chance to fire because the plan cannot accumulate `batchTokens` quickly enough
to want to advance during the cooldown. Raise the lag only after `/ctx` or
`CONTEXT_BUDGET_LOG` shows advances actually being blocked by gate 2; leaving
it at the default is harmless when gate 3 dominates, which it usually does.

`keepThinkingSteps` is the one knob with a measured quality trade-off:
published multi-turn tool-calling benchmarks give Qwen3-class models a few
points for retained thinking history, so keep it at 6 or above unless the
window is very small. Dropped thinking is still in the archive.

`maxPromptTokens` and `maxHardTokens` are stated in **real tokens** — the same
unit the provider reports — so `55000` means a 55k-token prompt whatever model
is answering. That is because the estimate is anchored: the newest assistant
message in the `context` hook carries the usage the provider returned for it,
and the size is measured as that number plus a short estimated tail. The tail is
the only part `charsPerToken` scales, so a wrong ratio moves a threshold by a few
percent of a tail rather than by the whole session, and switching models no
longer re-places every threshold. The anchor itself is exact for models that
have Pi resend thinking (signature-bearing ones do), and overshoots by at most
the newest turn's thinking for those that do not — bounded either way.

The two absolute thresholds are converted to fractions of the context window, so
they still mean different things on different models; use `modelOverrides` when
that matters:

```json
{
  "charsPerToken": 3.35,
  "maxPromptTokens": 55000,
  "maxHardTokens": 109000,
  "modelOverrides": { "local/qwen3.8-27b": { "maxPromptTokens": 200000, "cacheMode": "frozen" } }
}
```

`charsPerToken` is no longer a calibration step. Before, every threshold was
compared against `len / charsPerToken` over the whole session, so a ratio that
was off by half moved the threshold by half and the extension fired at the wrong
size; that is what made `real = BASE + chars / charsPerToken` worth fitting per
session (median 3.35 here, spread roughly 3.0–6.5, with the system prompt and
tool schemas in the intercept). With the anchor in place only the tail is scaled,
so the default is fine unless you want a tighter tail.

A model's entry is merged key by key over the global settings, not swapped in
whole: an override that sets one key keeps tracking the global value for every
other. If `maxPromptTokens` is at or above `maxHardTokens` the hard one wins and
the start line is pulled below it, since nothing can be above the high water and
below the start line at once. A threshold that is not a finite positive number,
or one at or above the whole context window, is dropped rather than kept: it
could not fire, and a threshold that silently never fires is the hardest kind of
misconfiguration to notice.

A subagent process — Pi's subagent extension spawns workers as `--mode json -p
--no-session --model …` — takes the `subagent` block's absolute thresholds
instead of the global ones, replaced wholesale: a field the block leaves out is
unset, not inherited. `modelOverrides` still applies on top of the replacement,
and with no `subagent` block subagents use exactly what the main session uses:

```json
{
  "maxPromptTokens": 40000,
  "maxHardTokens": 80000,
  "subagent": { "maxPromptTokens": 120000, "maxHardTokens": 200000 }
}
```

`cacheMode` trades pruning for a smaller prefix-cache miss: `frozen` locks the
boundary at the first move, `lagged` allows one every `cacheLagSteps` assistant
steps. Both only ever move the boundary less often — no mode can relax a tier
back upwards. `frozen` holds back the squeeze as well, because a squeeze moves
that same boundary and the mode is a promise that it will not move again;
`lagged` does not, because delay is all it asks for and the squeeze exists to get
under a cap the provider rejects above.

An older `errorHeadChars` key is still read as `stubHeadChars`.

## Measure

`replay.ts` runs a recorded session through the planner request by request and
reports peak context, cumulative prefill and plan advances.
`CONTEXT_BUDGET_REPLAY_CFG` overrides any config key, so one session can be
replayed under two settings and the difference read off directly:

```bash
node replay.ts ~/.pi/agent/sessions/<project>/<session>.jsonl 32768
CONTEXT_BUDGET_REPLAY_CFG='{"leanAfterSteps":0,"reduceSearch":false}' \
  node replay.ts ~/.pi/agent/sessions/<project>/<session>.jsonl 32768
```

`search-hitrate.ts` answers the question `reduceSearch` exists for: of the tools
a tool-search result listed, how many were ever invoked afterwards? On the
author's sessions it is 5.7%, and hits at rank 7 or worse are used 3.2% of the
time. Run it before and after changing a gateway's default result count — a
drop in "listed" with a flat "invoked" is the change working.

```bash
node search-hitrate.ts               # defaults to ~/.pi/agent/sessions
```

## Observe

- `/ctx` prints the provider-reported usage, plugin-sent estimate vs the cap,
  the cap against Pi's compaction threshold, the pin, the plan generation,
  archive counts by kind and the spill directory. The last-request line ends
  with `N tokens waiting for next batch` — that is gate 3 reporting how far
  the plan is from its `batchTokens` trigger. A number that keeps growing while
  nothing is elided means the prompt is over the start line and under the
  batch trigger, which is the normal staircase, not a fault.
- **A threshold that looks ignored is usually a start line doing its job.** If
  the prompt passes `maxPromptTokens` and then keeps growing, check whether
  `squeeze` is true and what `targetFraction` resolves to *for that model*
  (`/ctx` prints the resolved cap). Both are needed for an absolute ceiling;
  without them there is no ceiling to exceed. Remember that the absolute
  thresholds are converted to a fraction of the window, so the same
  `maxPromptTokens` is a far smaller share of a 1M window than of a 32k one.
- **`modelOverrides` matches `"<provider>/<modelId>"` as an exact string** and
  falls back to the global config silently when nothing matches — no warning,
  no log line. A key for a provider Pi does not have (a misspelling, or a model
  renamed upstream) is inert. Verify a key against `~/.pi/agent/models.json`
  rather than against memory; a typo here is indistinguishable from a setting
  that has no effect.
- The footer shows `ctx 48% −Nk gG` once something is elided (`G` is the plan
  generation; each increment moved the prefix-cache miss point once).
- `CONTEXT_BUDGET_LOG=<file>` appends one JSON line per request with
  `ctxBefore`, `ctxAfter`, `advanced`, `squeezed`, `resultsElided`,
  `argsElided`, `thinkingDropped`, the resolved `cap` / `trigger` / `clamped`,
  and the list of archived items.

## Replay a recorded session

```bash
node replay.ts ~/.pi/agent/sessions/<dir>/<session>.jsonl 100000
node replay.ts <session>.jsonl 32768 8192 3400   # window, Pi reserve, system-prompt tokens
```

Runs the shipped planner request by request over the session's active branch
and prints the resolved budget, peak and cumulative prompt tokens before and
after, the number of plan advances and two sample citations. Pass the reserve
and the size of the system prompt to see what the prompt would actually reach
against Pi's threshold.

## Test

```bash
npm test
```

## Alternatives

- [pi-dcp](https://github.com/PSU3D0/pi-dcp): heuristic like this one, does
  not handle thinking blocks or archive snapshots.
- [pi-context-prune](https://github.com/championswimmer/pi-context-prune) and
  [pi-condense](https://github.com/jjuraszek/pi-condense): summarise finished
  tool batches with an LLM call, which is a second lossy step on the same
  model.

## Fork changes

Forked from
[Don-Works/pi-context-budget](https://github.com/Don-Works/pi-context-budget)
@ `8078f60` (v0.6.0, by MCPlexer Engineering). Differences from upstream:

- **`maxPromptTokens` / `maxHardTokens`** — absolute token thresholds, so a
  target like "keep the prompt under 40k" can be written directly instead of
  being derived from a fraction of whatever window the model has. A fraction
  cannot express an absolute figure: the same fraction on a 200k and a 1M
  window is a 5x difference in real tokens.
- **`modelOverrides`** — per-model configuration keyed by `"provider/modelId"`,
  merged field-by-field over the global config. Upstream applies one global
  config to every model.
- **`cacheMode`** (`off` / `lagged` / `frozen`) — controls how often the plan
  is allowed to move the elision boundary, for deployments that pay for prompt
  caching.
- **`charsPerToken` default is now `3.35`**, measured by fixed-effect regression
  over recorded sessions (fitted per session, so the system prompt lands in the
  intercept). The upstream default of `3.3` turned out to be close; this fork
  briefly shipped `4.49`, which came from a flawed measurement and has been
  reverted — see the note under [Configure](#configure).
- **Per-model `charsPerToken` is a real effect**, not noise: `deepseek-v4-flash`
  and `gpt-5.6-luna` measure around `3.0`, `glm-5.3-flash` around `3.6`, and
  `deepseek-v4.1-flash` around `6.5`. Set it per model when the timing matters.
- **Archive garbage collection** — `gcArchives()`, run once per session start,
  deletes per-session snapshot directories older than 30 days. Upstream never
  removes them; they grow without bound. Liveness is judged by the newest mtime
  inside a directory, not the directory's own mtime, because rewriting an
  existing file (`state.json` on every request) does not move the directory
  mtime — using it deleted live sessions.
- **Fixes** — `cacheMode: "frozen"` now also holds back the squeeze, which
  moved the same boundary while bypassing the lock; malformed absolute
  thresholds (non-finite, or at/above the whole window) are refused instead of
  silently disabling the extension; `normalize()` no longer mutates its input,
  so resolving a per-model config cannot leak into the shared global config.

## License

AGPL-3.0-or-later, inherited from the upstream project. See [LICENSE](LICENSE).
Original work copyright MCPlexer Engineering; modifications in this fork are
released under the same license.
