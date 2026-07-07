"""KnowledgeBase mixin: outcome learning, consolidation, health diagnostics, and dispel."""
from __future__ import annotations

import dataclasses
import math

from .kb_types import ConsolidateReport, DiagnoseReport, DispelReport, FailureMode


class DiagnosticsMixin:
    """Usage-driven learning loop, self-consolidation, and KB health reporting."""

    def record_outcome(self, query: str, useful_texts: list[str],
                       useless_texts: list[str] | None = None) -> dict[str, int]:
        """Feed back which retrieved texts proved useful; usage counts steer future ranking."""
        with self._lock:
            self._metrics["record_outcomes"] += 1
            known = set(self._texts)
            applied = 0
            for t in useful_texts:
                if t in known:
                    self._usage[t] = self._usage.get(t, 0) + 1
                    applied += 1
            for t in (useless_texts or []):
                if t in known and self._usage.get(t, 0) > 0:
                    self._usage[t] -= 1
            return {"applied": applied, "ignored": len(useful_texts) - applied}

    def consolidate(self) -> ConsolidateReport:
        """Execute the dispel plan: merge redundancy, reparent contradiction, widen degenerate pairs."""
        if self._pipeline is None:
            return ConsolidateReport(changed=False, nodes_before=0, nodes_after=0,
                                     merges=0, aperture_updates=0, dispel_count=0)
        with self._lock:
            self._metrics["consolidations"] += 1
            engine = self._pipeline.engine
            store = self._pipeline.store
            nodes = store.all_nodes()
            nodes_before = len(nodes)
            scan = engine.tension_scan(nodes, top_n=len(nodes))
            thr = self._settings.agent.consolidate_tension
            eligible = {(a, b) for a, b, t, _kind in scan if t >= thr}
            plan = [row for row in engine.dispel_plan(scan) if (row[1], row[2]) in eligible]
            merges = reparents = aperture_updates = 0
            for op, a_id, b_id in plan:
                try:
                    a, b = store.get(a_id), store.get(b_id)
                except KeyError:
                    continue  # consumed by an earlier op this pass
                if a.parent == b.id or b.parent == a.id:
                    continue  # never collapse a tree edge; that is structure, not tension
                if op == "merge":
                    merged = engine.merge_nodes(a, b)
                    store.upsert(merged)
                    if str(b.id) != str(merged.id):
                        store.delete(b.id)
                    merges += 1
                elif op == "reparent":
                    candidates = [n for n in store.nodes_at(b.prefix) if n.id not in (a.id, b.id)]
                    new_parent = engine.reparent(b, candidates)
                    if new_parent is not None and new_parent != b.parent:
                        store.upsert(dataclasses.replace(b, parent=new_parent))
                        reparents += 1
                elif op == "summarize":
                    umbrella = engine.summarize_cluster([a, b])
                    if umbrella is not None:
                        store.upsert(umbrella)
                        if str(b.id) != str(umbrella.id):
                            store.delete(b.id)
                        aperture_updates += 1
            nodes_after = len(store.all_nodes())
        executed = merges + reparents + aperture_updates
        return ConsolidateReport(
            changed=executed > 0, nodes_before=nodes_before, nodes_after=nodes_after,
            merges=merges, aperture_updates=aperture_updates, dispel_count=executed)

    def diagnose(self) -> DiagnoseReport:
        """Health snapshot an agent reads to decide when to consolidate or diversify ingest."""
        if self._pipeline is None:
            return DiagnoseReport(0, 0, len(self._texts), len(self._memory.facts()),
                                  0.0, 0.0, 0.0, 0)
        engine = self._pipeline.engine
        nodes = [n for n in self._pipeline.store.all_nodes() if n.members]
        octaves = len({n.prefix for n in nodes})
        mean_ap = sum(n.aperture for n in nodes) / len(nodes) if nodes else 0.0
        scan = engine.tension_scan(nodes, top_n=len(nodes))
        mean_t = sum(t for _, _, t, _ in scan) / len(scan) if scan else 0.0
        redundant = sum(1 for _, _, _, kind in scan if kind in ("redundancy", "contradiction"))
        energy = engine.context_energy(nodes[:32])
        # entropy divergence: variance of per-octave aperture means
        octave_aps: dict[int, list[float]] = {}
        for n in nodes:
            octave_aps.setdefault(int(n.prefix), []).append(n.aperture)
        oct_means = [sum(v) / len(v) for v in octave_aps.values()]
        global_mean = sum(oct_means) / len(oct_means) if oct_means else 0.0
        entropy_div = math.sqrt(sum((m - global_mean) ** 2 for m in oct_means) / max(len(oct_means), 1))
        failure = FailureMode.NONE
        suggestions: list[str] = []
        if mean_ap > 1.2:
            failure = FailureMode.OUTSIDE_CONE
            suggestions.append("ingest more focused texts to tighten cone apertures")
        elif mean_t > 0.7 and redundant > len(nodes) // 4:
            failure = FailureMode.BOUNDARY_AMBIGUOUS
            suggestions.append("call consolidate() to resolve boundary tension between overlapping cones")
        elif octaves > 0 and len(nodes) / max(octaves, 1) < 2:
            failure = FailureMode.OVER_COMPRESSED
            suggestions.append("ingest more diverse texts across octaves to restore hierarchy depth")
        elif entropy_div > 0.4:
            failure = FailureMode.OCTAVE_MISMATCH
            suggestions.append("check for missing octave levels; use deep_search() for cross-octave queries")
        reason = self._pipeline.encoder_fallback_reason
        if reason is not None:
            suggestions.append("real encoder unavailable; rankings are RandomEncoder noise -- fix the encoder install")
        return DiagnoseReport(
            nodes=len(nodes), octaves=octaves, texts=len(self._texts),
            facts=len(self._memory.facts()), mean_aperture=float(mean_ap),
            mean_tension=float(mean_t), total_energy=float(energy), redundant_pairs=redundant,
            entropy_divergence=float(entropy_div),
            failure_mode=failure,
            recovery_suggestions=tuple(suggestions),
            degraded=reason is not None,
            fallback_reason=reason,
        )

    def metrics(self) -> dict[str, int]:
        """Usage counters for agent monitoring."""
        m = dict(self._metrics)
        m.update({"nodes": len(self._pipeline.store.all_nodes()) if self._pipeline else 0,
                  "n_texts": len(self._texts), "n_facts": len(self._memory.facts())})
        return m

    def entropy_dispel(self, entropy_ceiling: float = 2.0) -> "DispelReport":
        """Remove nodes whose entropy proxy exceeds ceiling; report before/after mean entropy."""
        if self._pipeline is None:
            return DispelReport((), 0.0, 0.0)
        with self._lock:
            store_nodes = {n.id: n for n in self._pipeline.store.all_nodes()}
            if not store_nodes:
                return DispelReport((), 0.0, 0.0)
            def node_entropy(node):
                aperture = getattr(node, 'aperture', 0.5) or 0.5
                n = max(1, len(getattr(node, 'members', []) or []))
                return math.log(1.0 + aperture * n)
            entropies = {nid: node_entropy(n) for nid, n in store_nodes.items()}
            before = sum(entropies.values()) / (len(entropies) + 1e-9)
            to_dispel = tuple(nid for nid, e in entropies.items() if e > entropy_ceiling)
            for nid in to_dispel:
                self._pipeline.store.delete(nid)
            after_entropies = {nid: e for nid, e in entropies.items() if nid not in to_dispel}
            after = sum(after_entropies.values()) / (len(after_entropies) + 1e-9) if after_entropies else 0.0
            return DispelReport(to_dispel, before, after)
