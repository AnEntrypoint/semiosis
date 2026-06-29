"""Self-improving research loop that instructs the calling agent; no in-process LLM."""
from __future__ import annotations

import json
import os

from .kb_types import (
    Directive,
    Hypothesis,
    Observation,
    ResearchResult,
    ResearchStep,
)
from .settings import Settings

_DEFAULT_INSTRUCTIONS = {
    "propose": "'{region}' keeps coming up and you do not actually know if it holds (it is loose, aperture {aperture:.3f}). What would you bet on here, and what single thing would make you wrong?",
    "experiment": "You cannot settle '{query}' from where you sit. Go look at octave {octave}, pull what actually bears on it, and bring back the texts.",
    "observe": "Now that you have looked: did it hold? Say what you saw and how sure that leaves you, from not-at-all to certain.",
    "refine": "Something here still does not sit right. What is the one question you most need answered next?",
}

_STAGES = ("propose", "experiment", "observe", "refine")


class ResearchLoop:
    """Drive a KnowledgeBase through propose-experiment-observe-refine cycles via emitted Directives."""

    def __init__(self, kb, settings: "Settings | None" = None) -> None:
        self._kb = kb
        self._settings = settings or getattr(kb, "_settings", None) or Settings()
        self._cfg = self._settings.research
        self.instructions = dict(_DEFAULT_INSTRUCTIONS)
        if self._cfg.instruction_persist_path:
            self._load_instructions(self._cfg.instruction_persist_path)
        self.hypotheses: list[Hypothesis] = []
        self._tried: set[str] = set()
        self._steps: list[ResearchStep] = []
        self._prev_energy: float | None = None

    # --- frontier ------------------------------------------------------------
    def _frontier(self) -> list[tuple[str, float, int]]:
        """Under-explored regions: highest-aperture nodes, deduped against tried hypotheses."""
        diag = self._kb.diagnose()
        if diag.nodes == 0:
            return []
        pipeline = getattr(self._kb, "_pipeline", None)
        if pipeline is None:
            return []
        nodes = [n for n in pipeline.store.all_nodes() if n.members]
        ranked = sorted(nodes, key=lambda n: float(n.aperture), reverse=True)
        out: list[tuple[str, float, int]] = []
        for n in ranked:
            region = self._kb._primary_text(n) if hasattr(self._kb, "_primary_text") else (n.digest or n.label or str(n.id))
            if not region or region in self._tried:
                continue
            out.append((region, float(n.aperture), int(n.prefix)))
            if len(out) >= self._cfg.frontier_top_k:
                break
        return out

    # --- stages --------------------------------------------------------------
    def propose(self, cycle: int) -> "Directive | None":
        """Emit a hypothesis-forming Directive for the most under-explored frontier region."""
        frontier = self._frontier()
        if not frontier:
            return None
        region, aperture, octave = frontier[0]
        self._tried.add(region)
        self.hypotheses.append(Hypothesis(text=region, octave=octave))
        text = self.instructions["propose"].format(region=region, aperture=aperture)
        return Directive(stage="propose", instruction_text=text, cycle=cycle,
                         context=(region,), target_query=region, target_octave=octave)

    def experiment(self, hyp: Hypothesis, cycle: int) -> Directive:
        """Emit a retrieval-experiment Directive grounded in the hypothesis."""
        text = self.instructions["experiment"].format(query=hyp.text, octave=hyp.octave)
        return Directive(stage="experiment", instruction_text=text, cycle=cycle,
                         context=(hyp.text,), target_query=hyp.text, target_octave=hyp.octave,
                         expected="texts bearing on the hypothesis")

    def observe(self, hyp: Hypothesis, obs: Observation, query: str) -> float:
        """Record the agent's observation, update usage, score support; returns support_score."""
        evidence = list(obs.evidence)
        if evidence:
            self._kb.record_outcome(query, evidence)
        support = float(obs.success_signal)
        hyp.support_score = support
        hyp.status = "supported" if support >= self._cfg.min_support_score else "refuted"
        return support

    def refine(self) -> tuple[Directive, dict]:
        """Consolidate the KB and regenerate the emitted instruction set (the trained artifact)."""
        report = self._kb.consolidate()
        supported = [h.text for h in self.hypotheses if h.status == "supported"]
        refuted = [h.text for h in self.hypotheses if h.status == "refuted"]
        # the refined instruction folds in what the agent learned this run
        refined = dict(self.instructions)
        if supported:
            refined["propose"] = (
                _DEFAULT_INSTRUCTIONS["propose"]
                + " You already trust this much: " + "; ".join(supported[:3]) + "; look near it."
            )
        if refuted:
            refined["experiment"] = (
                _DEFAULT_INSTRUCTIONS["experiment"]
                + " You already went down these and they led nowhere: " + "; ".join(refuted[:3]) + "."
            )
        self.instructions = refined
        directive = Directive(
            stage="refine",
            instruction_text=self.instructions["refine"],
            context=tuple(supported),
        )
        return directive, {"consolidated": report.changed, "merges": report.merges}

    # --- driver --------------------------------------------------------------
    def step(self, cycle: int, observe_fn) -> "ResearchStep | None":
        """One full cycle; observe_fn(directive)->Observation stands for the calling agent."""
        proposal = self.propose(cycle)
        if proposal is None:
            return None
        observe_fn(proposal)  # agent forms hypothesis; its text is already the frontier region
        hyp = self.hypotheses[-1]
        exp_directive = self.experiment(hyp, cycle)
        obs = observe_fn(exp_directive)
        if obs is None or not (obs.result_text or obs.evidence):
            return ResearchStep(cycle=cycle, directive=exp_directive, observation=None)
        support = self.observe(hyp, obs, hyp.text)
        energy = float(self._kb.diagnose().total_energy)
        delta = abs(energy - self._prev_energy) if self._prev_energy is not None else float("inf")
        self._prev_energy = energy
        return ResearchStep(cycle=cycle, directive=exp_directive, observation=obs, energy_delta=delta)

    def run(self, observe_fn) -> ResearchResult:
        """Run cycles until energy converges or max_cycles; refined_instructions is the trained output."""
        if self._kb.diagnose().nodes == 0:
            return ResearchResult(converged=True, refined_instructions=dict(self.instructions))
        converged = False
        no_obs = 0
        for cycle in range(self._cfg.max_cycles):
            step = self.step(cycle, observe_fn)
            if step is None:
                converged = True
                break
            self._steps.append(step)
            if step.observation is None:
                no_obs += 1
                if no_obs >= self._cfg.max_no_observation:
                    break
                continue
            no_obs = 0
            if step.energy_delta <= self._cfg.convergence_energy_delta:
                self.refine()
                converged = True
                break
            self.refine()
        result = ResearchResult(
            hypotheses=list(self.hypotheses),
            steps=list(self._steps),
            converged=converged,
            refined_instructions=dict(self.instructions),
        )
        if self._cfg.instruction_persist_path:
            self._save_instructions(self._cfg.instruction_persist_path)
        return result

    # --- persistence ---------------------------------------------------------
    def _save_instructions(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"instructions": self.instructions}, f)

    def _load_instructions(self, path: str) -> None:
        """Load refined instructions; fall back to defaults on missing/corrupt sidecar."""
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            loaded = data.get("instructions")
            if isinstance(loaded, dict):
                self.instructions.update({k: v for k, v in loaded.items() if isinstance(v, str)})
        except (json.JSONDecodeError, OSError, ValueError):
            self.instructions = dict(_DEFAULT_INSTRUCTIONS)
