"""Transactional, reversible projections over immutable identity evidence."""
from collections import Counter
from dataclasses import asdict
import json

from pipeline.identity.gallery_memory import MemoryPolicy, plan_memory


class MemoryStore:
    def memory_records(self):
        players = {r["id"]: dict(r) for r in self.db.execute("SELECT * FROM players")}
        states = {r["id"]: json.loads(r["data"]) for r in self.db.execute("SELECT * FROM memory_state")}
        result = []
        for row in self.db.execute("SELECT * FROM observations ORDER BY rowid"):
            node = json.loads(row["payload"])
            player = players[row["player"]]
            result.append({"node": node, "player": row["player"],
                "features": json.loads(row["features"]), "visibility": json.loads(row["visibility"]),
                "locked": bool(row["detached"] or player["status"] in {"manual_grouping", "non_player"}),
                "detached": bool(row["detached"]), "non_player": player["status"] == "non_player",
                "established": player["status"] == "candidate" and states.get(row["id"], {}).get("status") == "accepted",
                "active_template_keys": [t["key"] for t in states.get(row["id"], {}).get("templates", [])],
                "rejected": states.get(row["id"], {}).get("manual_rejected", False),
                "match_dependencies": states.get(row["id"], {}).get("match_dependencies", {}),
                "matched_against_revision": states.get(row["id"], {}).get("matched_against_revision"),
                "dependency_invalidated": states.get(row["id"], {}).get("dependency_invalidated", False)})
        return result

    def _invalidate_dependents(self, roots):
        """Invalidate the transitive evidence graph, retaining raw observations."""
        states = {r["id"]: json.loads(r["data"]) for r in self.db.execute("SELECT * FROM memory_state")}
        affected = set(roots)
        while True:
            added = {key for key, state in states.items() if key not in affected and
                     affected.intersection(state.get("match_dependencies", {}).get("observations", []))}
            if not added:
                break
            affected.update(added)
        for key in affected - set(roots):
            state = states[key]
            state.update(dependency_invalidated=True, invalidated_by=sorted(roots))
            self.db.execute("UPDATE memory_state SET data=? WHERE id=?", (json.dumps(state), key))
        return sorted(affected - set(roots))

    def _memory_projection(self):
        return {"players": [dict(r) for r in self.db.execute("SELECT * FROM players ORDER BY id")],
            "observations": [dict(r) for r in self.db.execute("SELECT id,player,gallery,detached FROM observations ORDER BY id")],
            "states": [dict(r) for r in self.db.execute("SELECT * FROM memory_state ORDER BY id")],
            "policy": self.meta("memory_policy")}

    def _save_memory_update(self, before, operation, detail=None):
        revision = self._record(operation, detail or {})
        self.db.execute("INSERT INTO memory_updates VALUES(?,?,?)",
                        (revision, json.dumps(before, sort_keys=True), operation))
        return revision

    def refresh_memory(self, policy=None, *, evidence=None, expected_revision=None):
        """Explicit CPU reprocessing/migration; never silently resets a library."""
        with self.db:
            self.db.execute("UPDATE metadata SET value=value WHERE key='match_id'")
            if expected_revision != self.meta("revision"):
                raise RuntimeError("Player library changed; refresh before editing")
            before = self._memory_projection()
            self._refresh_memory(policy, evidence=evidence)
            if before == self._memory_projection():
                return self.meta("revision")
            return self._save_memory_update(before, "memory_reprocess")

    def _refresh_memory(self, policy=None, *, evidence=None, incremental=False):
        policy = policy or MemoryPolicy(**json.loads(self.meta("memory_policy") or "{}"))
        policy.validate()
        if evidence:
            known = {r[0] for r in self.db.execute("SELECT id FROM observations")}
            if set(evidence)-known:
                raise ValueError("Curation references unknown observations")
            for key, value in evidence.items():
                previous = self.db.execute("SELECT data FROM memory_evidence WHERE id=?", (key,)).fetchone()
                old = json.loads(previous[0]) if previous else {}
                if old.get("source_fingerprint") and old["source_fingerprint"] != value.get("source_fingerprint"):
                    raise ValueError("Registered image/sample features changed; use a new run instead of replacing identity evidence")
                self.db.execute("INSERT OR REPLACE INTO memory_evidence VALUES(?,?)", (key, json.dumps(value, allow_nan=False)))
        evidence = {r["id"]: json.loads(r["data"]) for r in self.db.execute("SELECT * FROM memory_evidence")}
        records = self.memory_records()
        plan = plan_memory(records, evidence, policy, incremental=incremental)
        used = set()
        # Preserve the larger established group first if an update splits a
        # previous automatic identity. Manual groups always retain their ID.
        ordered = sorted(plan["groups"], key=lambda g: (-any(records[i]["locked"] for i in g["members"]),
            -g["promoted"], -len(g["members"]), min(g["members"])))
        assignments = {}
        for group in ordered:
            members = group["members"]
            forced = {records[i]["player"] for i in members if records[i]["locked"]}
            counts = Counter(records[i]["player"] for i in members)
            candidates = sorted(counts, key=lambda p: (
                not any(records[i].get("established") for i in members if records[i]["player"] == p),
                -counts[p], min(i for i in members if records[i]["player"] == p)))
            pid = next(iter(forced)) if forced else next((p for p in candidates if p not in used
                and self.db.execute("SELECT status FROM players WHERE id=?", (p,)).fetchone()[0] not in {"manual_grouping", "non_player"}), None)
            pid = pid or self._new_player("candidate" if group["promoted"] else "needs_review")
            used.add(pid)
            if not forced:
                self.db.execute("UPDATE players SET alias_of=NULL,status=? WHERE id=?",
                                ("candidate" if group["promoted"] else "needs_review", pid))
            for i in members:
                assignments[i] = pid
                plan["states"][i]["novel_by_distance"] = group["novel_by_distance"]
                plan["states"][i]["nearest_other_distance"] = group["nearest_other_distance"]
        for i, record in enumerate(records):
            state = plan["states"][i]
            if i in plan["duplicates"]:
                target = plan["duplicates"][i]
                if target in assignments: assignments[i] = assignments[target]
            if i not in assignments:
                old = record["player"]
                pid = old if (old not in used or record["locked"] or record["non_player"]) else self._new_player("needs_review")
                assignments[i] = pid; used.add(pid)
                if not record["locked"] and not record["non_player"]:
                    self.db.execute("UPDATE players SET status='needs_review',alias_of=NULL WHERE id=?", (pid,))
            pid = assignments[i]
            state.pop("feature_indices", None)
            state["planner_version"] = plan["planner_version"]
            state["manual_rejected"] = record["rejected"]
            state["matched_against_revision"] = (record.get("matched_against_revision")
                if state.get("match_dependencies") == record.get("match_dependencies")
                else self.meta("revision")) if state.get("match_dependencies") else None
            state["decision"] = ("memory_matched" if state["status"] == "accepted" else state["status"])
            self.db.execute("UPDATE observations SET player=?,gallery=? WHERE id=?",
                            (pid, int(bool(state["templates"]) and not record["detached"]), record["node"]["node_id"]))
            self.db.execute("INSERT OR REPLACE INTO memory_state VALUES(?,?)",
                            (record["node"]["node_id"], json.dumps(state, allow_nan=False)))
        # Aliases only for a whole old group moving to one successor. Never
        # delete an ID, or alias a split group to an arbitrary child.
        successors = {}
        for i, record in enumerate(records): successors.setdefault(record["player"], set()).add(assignments[i])
        for old, targets in successors.items():
            if old not in used and len(targets) == 1:
                new = next(iter(targets))
                if new != old:
                    self.db.execute("UPDATE players SET alias_of=? WHERE id=? AND label IS NULL", (new, old))
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES('memory_policy',?)", (json.dumps(asdict(policy), sort_keys=True),))

    def _rollback_memory(self):
        row = self.db.execute("SELECT * FROM memory_updates WHERE revision=?", (self.meta("revision"),)).fetchone()
        if row is None:
            raise ValueError("Only the current memory update can be rolled back; later edits must be reviewed first")
        before = json.loads(row["before_state"])
        # Ingested raw observations/events remain in place; the saved projection
        # is taken after raw registration and before assigning/updating memory.
        for player in before["players"]:
            self.db.execute("UPDATE players SET label=?,status=?,alias_of=? WHERE id=?",
                (player["label"], player["status"], player["alias_of"], player["id"]))
        previous = {p["id"] for p in before["players"]}
        for player in self.db.execute("SELECT id FROM players").fetchall():
            if player[0] not in previous:
                self.db.execute("UPDATE players SET alias_of=NULL,status='needs_review' WHERE id=?", (player[0],))
        for item in before["observations"]:
            self.db.execute("UPDATE observations SET player=?,gallery=?,detached=? WHERE id=?",
                (item["player"], item["gallery"], item["detached"], item["id"]))
        self.db.execute("DELETE FROM memory_state")
        self.db.executemany("INSERT INTO memory_state VALUES(?,?)", [(s["id"],s["data"]) for s in before["states"]])
        if before["policy"] is None:
            self.db.execute("DELETE FROM metadata WHERE key='memory_policy'")
        else:
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES('memory_policy',?)", (before["policy"],))
        return self._record("memory_rollback", {"undone_revision": row["revision"]})
