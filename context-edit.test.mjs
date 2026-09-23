import { test } from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULTS,
  PI_COMPACTION_DEFAULTS,
  applyContextEdits,
  budgetFor,
  isCutPoint,
  spanFor,
  spanMessages,
  tokensToFree,
  textOf,
} from "./plan.ts";

const SMALL = 32768;
const pi = (over = {}) => ({ ...PI_COMPACTION_DEFAULTS, keepRecentTokens: 32000, ...over });

const chars = (n) => "x".repeat(n);
let seq = 0;
const msg = (role, n) => ({ type: "message", id: `e${seq++}`, message: { role, content: [{ type: "text", text: chars(n) }] } });
const edit = (targetId, replacement) => ({ type: "context_edit", id: `x${seq++}`, targetId, replacement });

// Same shape budget.test.mjs builds: one user message, then assistant/tool-result pairs.
function branch(steps) {
  seq = 0;
  const entries = [msg("user", 200)];
  for (let i = 0; i < steps; i++) {
    entries.push(msg("assistant", 660));
    entries.push({ ...msg("toolResult", 3300), message: { role: "toolResult", toolCallId: `call${i}`, content: [{ type: "text", text: chars(3300) }] } });
  }
  return entries;
}

const messagesOf = (entries) => spanMessages(applyContextEdits(entries), 0, entries.length);

test("a branch without context_edit entries comes back untouched", () => {
  const entries = branch(3);
  assert.equal(applyContextEdits(entries), entries);
});

test("a null replacement omits the target message from the context", () => {
  const entries = branch(3);
  const target = entries[3].message; // first assistant message
  const out = [...entries, edit(entries[3].id, null)];
  const msgs = messagesOf(out);
  assert.ok(msgs.length < spanMessages(entries, 0, entries.length).length, "one message fewer");
  assert.equal(msgs.includes(target), false);
});

test("a non-null replacement rewrites only the content of the target", () => {
  const entries = branch(3);
  const out = [...entries, edit(entries[3].id, { content: [{ type: "text", text: "short" }] })];
  const msgs = messagesOf(out);
  assert.equal(msgs.length, spanMessages(entries, 0, entries.length).length, "same message count");
  const replaced = msgs.filter((m) => m.role === "assistant")[1]; // entries[3], the edited one
  assert.equal(textOf(replaced), "short");
  // Everything else keeps its identity, so the serialized prefix is unchanged outside the edit.
  assert.equal(msgs.find((m) => m.role === "toolResult"), entries[2].message);
});

test("the last edit per target on the path wins", () => {
  const entries = branch(3);
  const out = [
    ...entries,
    edit(entries[1].id, { content: [{ type: "text", text: "first" }] }),
    edit(entries[1].id, null),
    edit(entries[1].id, { content: [{ type: "text", text: "last" }] }),
  ];
  const msgs = messagesOf(out);
  assert.equal(textOf(msgs[1]), "last");
  assert.ok(msgs.length > 0, "the message is not omitted");
});

test("an edit whose target is not on this branch is ignored", () => {
  const entries = branch(3);
  const out = [
    ...entries,
    edit("e-other-branch", { content: [{ type: "text", text: "nowhere" }] }),
    edit("e-other-branch", null),
  ];
  assert.deepEqual(messagesOf(out).map((m) => textOf(m)), spanMessages(entries, 0, entries.length).map((m) => textOf(m)));
});

// The edit entries themselves carry no message and no tokens, and an omitted target costs
// nothing — the span has to be priced at what the context actually holds.
test("spanFor prices the recut span without omitted messages and at replaced content", () => {
  const entries = branch(20);
  const budget = budgetFor(DEFAULTS, SMALL, pi());
  const prep = {
    firstKeptEntryId: entries[3].id,
    messagesToSummarize: spanMessages(entries, 0, 3),
    turnPrefixMessages: [],
    tokensBefore: 24_288,
  };
  const need = tokensToFree(prep.tokensBefore, budget.trigger);
  const base = spanFor(prep, entries, budget, need, DEFAULTS);
  assert.equal(base.recut, true, "the fixture recuts, like budget.test.mjs's loop case");

  // Omit one big message inside the span and shrink another: the span must get cheaper.
  const lighter = [...entries, edit(entries[6].id, null), edit(entries[7].id, { content: [{ type: "text", text: "short" }] })];
  const cheap = spanFor(prep, lighter, budget, need, DEFAULTS);
  assert.ok(cheap.tokens < base.tokens, `omitted/shrunk messages cut the span: ${cheap.tokens} < ${base.tokens}`);
  assert.ok(!cheap.messagesToSummarize.includes(entries[6].message), "the omitted message is in no span");

  // Grow a message instead and the span gets dearer.
  const heavier = [...entries, edit(entries[7].id, { content: [{ type: "text", text: chars(33000) }] })];
  const dear = spanFor(prep, heavier, budget, need, DEFAULTS);
  assert.ok(dear.tokens > base.tokens, `a larger replacement costs more: ${dear.tokens} > ${base.tokens}`);

  // The cut still lands on a real, non-toolResult message id.
  assert.ok(cheap.firstKeptEntryId && isCutPoint(applyContextEdits(lighter).find((e) => e.id === cheap.firstKeptEntryId)));
});
