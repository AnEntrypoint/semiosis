---
key: mem-c9acdabf080ccaea-1393
ns: default
created: 1782756416118
updated: 1782756416118
---

{"key":"gm-implicit-predicament-over-exposition","value":"gm-method lesson (2026-06-29): for an instruction-emitting loop (see gm-instruction-emitting-loop-pattern), there are two ways to drive the agent and the implicit one is stronger. EXPLICIT: tell the agent the apparatus ('you are the LLM', here are the four stages, here is the stage->action table) -- it then performs the labeled steps self-consciously. IMPLICIT: emit each step as a genuine predicament/question the agent is cornered into resolving ('X keeps coming up and you do not actually know if it holds -- what would make you wrong?'; 'You cannot settle this from where you sit -- go look'), and the skill framing drops the agent INTO the situation rather than describing the goal TO it. The agent then does the wanted behavior because it is cornered into it, not because it was instructed. Key separation: keep the apparatus vocabulary in maintainer-facing surfaces (internal _STAGES names, method names, architecture docs annotated 'the driven agent never sees these') but strip it from every agent-facing surface (emitted prose, the skill body). The struct field the callback reads can keep its name; what matters is the agent acts on the predicament text, never a narrated stage label. When reshaping prose that tests assert on, update the witness substrings to the new implicit markers without loosening what they prove."}
