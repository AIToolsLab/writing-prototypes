# Interaction sketches

Concrete examples of the eight candidate features in [HANDOFF.md](../HANDOFF.md), Task 2. One document
runs through all of them so they're easy to compare. **Probabilities in the mockups are made up** to illustrate
the idea. Sections marked *Measured* report real runs against the engine (Qwen3.8-27B, 2026-11-11), reproducible
with [`experiments/interactions/probe.py`](../experiments/interactions/probe.py).

The fence, restated: the AI may sketch ahead *or* spell the next word, never both. When revising, the model
reads and the human writes.

## The running example

Assignment (the prompt): *Write a 500-word op-ed for the campus paper about a change you'd like to see on campus.*

> **P1.** The library should stay open all night during finals week. Right now it closes at midnight, which
> pushes students into dorm lounges that are loud and crowded. A quiet place to study is not a luxury during
> the most stressful week of the semester.
>
> **P2.** Of course, keeping the building open costs money. But the library already pays two staff members
> to stay until midnight; extending their shift for five nights a year is a small expense next to the tuition
> students pay for those same exams.

For features 1, 2, and 5, the cursor is at the end of P1 and P2 doesn't exist yet.

Prompt layout used throughout (what the Trie client does today): the assignment is the user turn; the draft
is the assistant turn.

```
<|im_start|>user
Write a 500-word op-ed ...<|im_end|>
<|im_start|>assistant
The library should stay open all night ...
```

---

## 1. Pencil stubs

**What the writer sees.** They type a note ahead of the cursor, in a different style (pencil-gray, italic).

```
...during the most stressful week of the semester. ▌       ✎ but cost? staff already paid
  Of course 31%   Some 12%   Critics 9%   Keeping 7%   The 6%
```

Without the stub, the same spot predicts something like `The 18%  Students 11%  Other 8%  Finals 6%`
(more reasons to stay open). The stub moves probability toward the turn the writer said was coming.

**How.** The stub goes into the user turn, after the assignment:

```
user:      Write a 500-word op-ed ...
           [My notes on where the draft goes next: but cost? staff already paid]
assistant: The library should stay open ... semester.▌
```

Then `/v1/continuations` as usual. Works today.

**Cost.** Anything after the stub is the whole draft, so editing the stub re-decodes the draft (a few hundred
ms at 700 tokens). Apply the stub when the writer presses Enter or leaves it, not on every keystroke.

An alternative that costs nothing: put the note *inline* at the cursor, as text the model sees:
`...semester. [note to self: but cost? staff already paid] ▌`. Only the note and what follows it are decoded.
The risk is the model treating the bracket as part of the prose (closing it, writing more notes). Worth a
side-by-side test.

**Questions.**
- When the writer reaches the stub, does it disappear, or do they delete it? Several stubs at once is an
  outline; do they all go in the prompt, in order?
- A stub tends to make the model turn to it *immediately* ("Of course"), even if the writer meant "after one
  more sentence." Is that a problem, or exactly the nudge you want?

## 2. Moves, not words

**What the writer sees.** A strip above the suggestions, on demand (a key, or when they pause at a sentence end):

```
What could come next?  give an example 34% · concede a cost 22% · restate the claim 15% ·
                       cite evidence 12% · describe a scene 8%
```

Clicking "concede a cost" turns it into a stub (feature 1): `✎ concede a cost`. No prose is ever shown.

**How: not a multiple-choice letter.** The first sketch asked a lettered multiple-choice question and read
the probability of each answer letter, in one forward pass. *Measured:* it doesn't work.

- Only 34–41% of the probability goes to any letter at all; the rest starts a prose answer ("To…", "The…").
- The letter **B** wins in all three option orders tried (15–23%), whichever move it labels: "concede a cost",
  then "cite evidence", then "give an example". The numbers mostly measure position.

Scoring each fixed label as text after "The next sentence should" didn't work either: the model wants to bold
its answer (`**`), and that swamps the differences between labels.

**How: free phrases.** Let the model finish "The next sentence should" itself, with `/v1/continuations`
(4-word stop, about 400 ms). *Measured,* on P1:

```
0.131  expand on the negative consequences
0.112  expand on the specific negative
0.025  expand on the specific consequences
0.023  broaden the scope of the
0.022  provide specific evidence or a
0.012  transition from the problem to
0.010  acknowledge the likely counter
```

Coarse, specific to this draft, with real probabilities. The cost is that one move is split across several
wordings, so something has to group them (and strip the `**`).

**Two numbers that mean different things.**
- *The model's advice as an editor*: the free phrases above. "What should come next."
- *What writers do next*: take the engine's likely next sentences of the draft itself and label the move each
  makes (with the campus model). Then 22% means "22% of likely next sentences concede something." Costlier, and
  the prose is never shown.

Predictive text shows the second kind of number. Showing the first kind in the same style could mislead.

**Question.** A fixed list of moves is teachable and comparable across students; free phrases are specific to
the draft but closer to supplying the idea. Free phrases with light grouping may be a middle ground.

## 3. The mirror

**What the writer sees.** Their own text, shaded by how predictable each word was given the assignment and
everything before it. ░ = the model's likely choice (over 50%), ▒ = plausible (5–50%), █ = the writer's
surprise (under 5%).

```
The library should stay open all night during finals week.
▒▒▒ ███████ ░░░░░░ ▒▒▒▒ ░░░░ ▒▒▒ ░░░░░ ▒▒▒▒▒▒ ▒▒▒▒▒▒ ░░░░░

Right now it closes at midnight, which pushes students into dorm
▒▒▒▒▒ ░░░ ▒▒ ░░░░░░ ░░ ▒▒▒▒▒▒▒▒▒ ▒▒▒▒▒ ██████ ░░░░░░░░ ▒▒▒▒ ████
lounges that are loud and crowded.
▒▒▒▒▒▒▒ ▒▒▒▒ ░░░ ░░░░ ░░░ ░░░░░░░░

A quiet place to study is not a luxury during the most stressful week
▒ ▒▒▒▒▒ ░░░░░ ░░ ░░░░░ ░░ ▒▒▒ ░ ░░░░░░ ▒▒▒▒▒▒ ░░░ ░░░░ ░░░░░░░░░ ░░░░
of the semester.
░░ ░░░ ░░░░░░░░░
```

The useful message is at the sentence level: *the third sentence is the one the model would have written.*
That's a prompt to ask "is this sentence doing work?", not a grade.

Optional: **the generic draft**, collapsed by default. "Here's what the model writes from the assignment
alone":

> Every finals week, students across campus face the same problem: where to study. Extending library hours
> would ...

Comparing the two shows which of the writer's ideas the model didn't have (the dorm lounges, the staff
already being paid).

**How.** Shading: one `/v1/score` call (what highlights already do), then group tokens into words and use
the word's probability (product of its tokens'). Using raw tokens would shade the second half of every
long word, which says nothing about the writer.

Generic draft: the handoff says the engine can't generate, but I think `/v1/continuations` already does
greedy decoding if you ask it to: `{prompt, text: "", branch: 1, min_prob: 0, max_depth: 400,
max_rounds: 400, time_ms: 30000}`. Each round extends the single branch by one token. Untested; `min_prob`
must be 0 or the search stops once the path's probability falls under 1%. Greedy output is the model's single
most likely draft, which may suit the mirror better than a sample, though it can loop.

**Framing: two-sided.** Both too surprising ("weird") and too predictable ("boring") read badly. So the mirror
isn't "more surprise is better", which also makes it harder to game: odd word choices get flagged too. The
closest prior work I know of is GLTR (2019), which colors each token by its rank under the model, but to detect
generated text, not to help a writer.

**Doubts.**
- **Model surprise isn't reader surprise.** A typo is a big spike to the model and nothing to a reader, who
  reads through it. Highlights already return the model's top choice at each token, so a spike whose top choice
  is within an edit or two of the written word can be marked as a likely typo instead. Rare-but-correct words
  (names, jargon) are harder: the model is surprised and the reader isn't.
- **The right amount of surprise depends on position.** A thesis sentence should surprise; a transition
  shouldn't. One fixed "good" band would mark both wrong. Safer to flag only clear extremes (a sentence with
  almost no surprise, or a single spike) and let the writer judge the rest.
- "Predictable" means given the writer's *own* earlier text too. If a writer sets up a surprising idea, the
  sentences that follow from it are shaded as predictable. I think that's right (originality is credited where
  it enters), but it'll look unfair to students.
- Function words are always ░. That's noise; maybe shade only content words, or only show sentence totals.

## 4. Count choices, not bits

**What the writer sees.** Permanent underlines showing how each span got there:

```
A quiet place to study is not a luxury during the most stressful week of the semester.
──────────── typed ──────────────────── │2/5│ ······ default ······
```

- no underline: typed
- `│k/n│`: picked suggestion k of n shown (a deliberate choice among alternatives)
- dotted: accepted the top suggestion (Tab, or Alt+1)

A footer: *312 words: 270 typed, 30 accepted as suggested, 12 picked from alternatives.*

**How.** Client only. Record spans when suggestions are accepted; shift them on edits (the page already
shifts highlight ranges to follow edits); any edit inside a span turns those characters into "typed". Save
with the document.

**Doubts.**
- Typed doesn't mean thought of. A student can retype the visible top suggestion. A fourth category catches
  part of that: "typed what was on screen." It doesn't catch copying from another window.
- Declining every suggestion is also a choice, and it isn't counted.
- This and the mirror might be one view: shading is how predictable the words are, underlines are how they
  got there. "Typed, but ░ predictable" and "picked 4/5, █ surprising" are both interesting.

## 5. Your old draft as the pencil layer

**What the writer sees.** They select the third sentence and press a key. It turns pencil-gray, and the
cursor is on a fresh line under it:

```
✎ A quiet place to study is not a luxury during the most stressful week of the semester.
During finals, a quiet place▌
  to ⟲ 72%   is 9%   matters 4%
```

`⟲` marks the old version's next word, so keeping it is one tap. When the writer diverges (`During finals,
quiet is ▌`), the old words stop leading and the suggestions are ordinary predictions.

**How.** It's feature 1 with the old sentence as the stub, plus the sentence that follows it (since the model
can't see past the cursor):

```
user:      [assignment]
           [I'm rewriting this sentence: "A quiet place to study is not a luxury ..."
            It's followed by: "Of course, keeping the building open costs money."]
assistant: The library should stay open ... loud and crowded. During finals, a quiet place▌
```

The existing "rewrite this document" mode is the whole-document version of this.

**Doubt.** The old version is the strongest suggestion, so it pulls the writer back toward the sentence they
chose to change. Maybe show the old word separately from the alternatives (`keep: to · or: is, matters`) so
keeping is a decision, not a default.

## 6. Ripples

**What the writer sees.** They change P1's "closes at midnight" to "closes at 10 p.m." A mark appears
downstream:

```
P2  ...the library already pays two staff members to stay until midnight; extending...
                                                           ~~~~~~~~~~~~~~ 40× more surprising since your edit
```

The ripple caught an inconsistency the writer created: the staff now stay two hours past closing.

**How.** Two `/v1/score` results: the one from before the edit (the page already has it, from highlights) and
one after. Line up the tokens after the edit by character offset, and mark spans whose logprob fell by more
than a threshold.

**Doubts.**
- It only flows forward. Editing P2 can't flag an inconsistency in P1, because the model reads left to right.
  A writer won't expect that asymmetry.
- Numerics move logits by up to about 0.25 between runs, so the threshold needs to clear that; word-level sums
  help.
- Surprise can also *drop* (the edit set up later text better). Show that, or only rises?

## 7. Either/or spans

**What the writer sees.** They keep two versions of P1's last sentence. A marker shows which one P2 reads
better after:

```
⟨A⟩ A quiet place to study is not a luxury during the most stressful week of the semester.
⟨B⟩ Students shouldn't have to choose between sleep and a quiet desk.        ◀ sets up P2 better
```

Hovering shows where in P2 the difference is (say, "two staff members to stay until midnight" is more
expected after B, which mentions sleep and the hour).

**How.** Score P2 after `...P1 + A` and after `...P1 + B`; sum the logprobs over P2's characters. Two
`/v1/score` calls today (the second rewinds to the checkpoint before the span). A `/v1/compare` that forks
one sequence per option would be faster but isn't needed to try it.

**This is the same computation as ripples:** ripples compares the following text after the old and new
version of an edit; either/or compares it after two versions the writer keeps side by side.

**Doubts.**
- It rewards the version that *previews* what follows. A sentence that mentions cost makes "costs money"
  predictable. So "reads better after" partly means "telegraphs the next point." Label it "sets up what follows"
  rather than "better."
- Should P(A) vs P(B) themselves show? That's a different question ("which is more expected"), and it favors
  the generic version. I'd keep it out.
- How much following text: the next sentence, the paragraph, the rest? Longer dilutes the signal.

## 8. Gist-back

**What the writer sees.** A short margin note per paragraph: what a reader takes it to claim.

```
P1 │ The library should stay open all night...   ┊ midnight closing drives students to noisy dorms
P2 │ Of course, keeping the building open...     ┊ cost is trivial next to tuition
```

Suppose the writer meant P2 to say "the staff are already there." The gist says the tuition comparison is
what lands, so the writer knows which clause is carrying the paragraph.

**How.** The campus model, one chat request per paragraph, capped at about 10 words: "In at most 10 words,
what does a reader take this paragraph to claim?"

**A variant that keeps the "model reads, human writes" rule strictly: gist-*check*.** The writer types their
own intended gist in the margin first. The local engine scores it:
`log P(gist | paragraph) − log P(gist | no paragraph)`, by placing the gist after the paragraph with a
prompt like "This paragraph's point:". It shows whether the paragraph supports the note, strongly or weakly,
without generating anything. Runs on the local engine today, and makes the writer say what they meant before
seeing a verdict.

**Several readings, so it doesn't read as authoritative.** Show a few divergent interpretations instead of one
gist. The divergence should come from the paragraph, not be built in. Reader personas (skeptical administrator,
tired student) give several readings even for a perfectly clear paragraph; then the spread says nothing, and the
writer learns to defend against misreadings no one would have. If the readings come from the model's own
distribution, the spread is the signal: they agree on a clear paragraph and split on an ambiguous one.

*Measured,* a cheap version on the local engine: free continuations of "This paragraph claims that" (6-word stop).

```
P1  0.165  the library should extend its hours to
    0.162  the library should remain open 24
    0.097  the library should remain open all night
    0.043  the library should remain open overnight during
P2  0.120  extending the library's
    0.076  extending library hours for
    0.046  the cost of keeping
    0.039  extending library hours is
```

P1's readings agree, as they should. P2's are still restating the topic after six words, so they can't yet
show whether readers take the point as "staff are already there" or "small next to tuition". The readings
mostly differ in wording, not meaning. This needs longer readings plus grouping by meaning, probably with the
campus model.

**Doubt.** Gist-back generates text, but it's behind the writer (summarizing what they wrote), not ahead. A
student could still collect the gists into an abstract or outline they didn't write.

---

## Across all eight

**Fewer mechanisms than features.**

| Mechanism | Features | Engine today |
|---|---|---|
| Condition predictions on a note | 1, 2 (its output), 5 | works |
| Distribution over short coarse phrases | 2, 8 (several readings) | works; needs grouping by meaning |
| Probability of a given text | gist-check | works |
| Following text scored after two versions | 6, 7 | works (two scores); `/v1/compare` would speed it up |
| Per-token score of the writer's text | 3 | works |
| Free generation | 3 (generic draft), 8 | greedy works, untested; 8 wants the campus model |
| Provenance of each span | 4 | client only |

**Sessions.** Each feature with its own prompt layout (moves puts the draft in the user turn, stubs change the
user turn) is a different cached text. Give each its own session name per tab (`${SESSION}:moves`) or they'll
evict each other's cache on every switch. That multiplies sessions per writer. About four fit on the GPU before
they move to host memory, but restoring one is cheap, so this isn't a real limit at prototype scale.

**Suggested order.** Build stubs (1) first, since 2's output and 5 both reuse it. Then the either/or scorer
(7), which gives you ripples (6) almost free. The mirror (3) is mostly a display change on top of highlights.
