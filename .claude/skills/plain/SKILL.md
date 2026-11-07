---
name: plain
description: Say the last answer again in plain language and compactly. The user is a control and optimization engineer, not yet a software engineer, and reads mathematics and physics fluently; an explanation that only makes sense to someone who already knows the codebase has not explained anything.
when_to_use: The user asks to re-explain, says an answer was too long or too dense, asks "what do you mean?" or "what am I supposed to understand from this?", or types /plain.
allowed-tools: Read Grep Glob
---

# Say it again, plainly

Re-explain what was just said. Same content, different words — this is not a summary and not a
new answer: nothing is dropped, nothing new is added, no hedging is introduced.

## What to change

- **Plain words.** Name the thing by what it does, not by what it is called. "A closure is a
  function that carries its own state, like `pid()` in control.py handing back a controller that
  keeps its integrator between calls." Say the jargon term once if the user will meet it again
  on screen, then use the plain words.
- **Mathematics keeps its terms.** Plain is for software and codebase jargon, not for maths or
  physics. A mathematical or physical object is named by its standard term, with its
  parameters: "a uniform distribution over ±9 g", not "a range of what the target can do".
- **Compact.** Short sentences. No preamble, no restating the question, no closing summary of
  the summary. If a table says it in four rows, use four rows.
- **Lead with the answer.** The finding first, then why, then what it means for the work. Never
  build up to it.
- **Concrete over abstract.** A command they can type, a number that was measured, a file they
  can open — not a description of a category of thing.

## What must survive

- **Every number keeps its conditions.** A measurement without what it was measured under is
  not plainer, it is wrong.
- **Uncertainty stays uncertain.** If something was not verified, the plain version still says
  it was not verified. Simplifying is not the same as becoming confident.
- **Any correction stays.** If the original said "I was wrong about X", so does this one.
- **Their decision stays theirs.** If the original ended with a choice for the user, the
  plain version ends with the same choice, not with a recommendation dressed as a conclusion.

## Check before sending

Could someone who has never opened this repository follow it? If a sentence only parses for
someone who already knows the code, it has not been re-explained — it has been re-typed.
