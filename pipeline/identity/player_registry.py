"""Transactional match-scoped identity memory; KPR weights remain frozen.

Model evidence is immutable after registration. Reviews change assignments,
never source artifacts. A rejected observation is not a new training class.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sqlite3
import uuid

import numpy as np

from pipeline.identity.cross_clip import part_distances


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class PlayerRegistry:
    def __init__(self, path: Path, match_id: str):
        if not match_id or len(match_id) > 120:
            raise ValueError("match_id must contain 1–120 characters")
        path = Path(path)
        if path.is_symlink():
            raise ValueError("Player database cannot be a symlink")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS players(
                id TEXT PRIMARY KEY, label TEXT, status TEXT NOT NULL, alias_of TEXT REFERENCES players(id));
            CREATE TABLE IF NOT EXISTS observations(
                id TEXT PRIMARY KEY, clip TEXT NOT NULL, player TEXT NOT NULL REFERENCES players(id),
                auto_player TEXT NOT NULL REFERENCES players(id), fingerprint TEXT NOT NULL,
                payload TEXT NOT NULL, features TEXT NOT NULL, visibility TEXT NOT NULL,
                gallery INTEGER NOT NULL, detached INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, clip TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS history(
                revision TEXT PRIMARY KEY, operation TEXT NOT NULL, payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        """)
        current = self.meta("match_id")
        if current is not None and current != match_id:
            self.db.close()
            raise ValueError("Database belongs to a different match")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES('match_id', ?)", (match_id,))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    def meta(self, key):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _record(self, operation, payload):
        revision = uuid.uuid4().hex
        self.db.execute("INSERT INTO history(revision, operation, payload) VALUES(?,?,?)",
                        (revision, operation, json.dumps(payload, ensure_ascii=False)))
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES('revision', ?)", (revision,))
        return revision

    def _new_player(self, status):
        pid = "PL-" + uuid.uuid4().hex[:20]
        self.db.execute("INSERT INTO players VALUES(?,?,?,NULL)", (pid, None, status))
        return pid

    @staticmethod
    def _conflict(left, right):
        a, b = left.get("jersey") or {}, right.get("jersey") or {}
        if (a.get("status") == b.get("status") == "candidate_consensus"
                and a.get("number") != b.get("number")):
            return "different_reliable_jersey_numbers"
        if left["clip_name"] == right["clip_name"]:
            # No evidence of temporal separation means abstain, not assume separation.
            intervals_a, intervals_b = left.get("intervals", []), right.get("intervals", [])
            if not intervals_a or not intervals_b or any(
                    a[0] < b[1] and b[0] < a[1] for a in intervals_a for b in intervals_b):
                return "overlapping_tracks_in_same_clip"
        return None

    def ingest(self, nodes, features, visibility, events, contract, *, max_distance=.2,
               novelty_distance=.5, min_margin=.05, min_common_parts=2, gallery_limit=12):
        if (max_distance is not None and (not math.isfinite(max_distance) or max_distance < 0)):
            raise ValueError("Invalid match distance")
        if (not math.isfinite(novelty_distance) or novelty_distance <= (max_distance or 0)
                or not math.isfinite(min_margin) or min_margin < 0 or gallery_limit < 1):
            raise ValueError("Require novelty_distance > match distance, nonnegative margin and gallery limit")
        if nodes:
            part_distances(np.stack(features), np.stack(visibility), min_common_parts=min_common_parts)
            if not contract:
                raise ValueError("KPR feature provenance is required")
        changed = []
        with self.db:
            if contract:
                serialized = json.dumps(contract, sort_keys=True)
                existing = self.meta("feature_contract")
                if existing and existing != serialized:
                    raise ValueError("Incompatible KPR feature space; re-extract using the registered model")
                self.db.execute("INSERT OR REPLACE INTO metadata VALUES('feature_contract',?)", (serialized,))
            for node in nodes:
                number_contract = (node.get("jersey") or {}).get("provenance")
                if number_contract and not number_contract.get("provenance", {}).get("empty_input"):
                    serialized_numbers = json.dumps(number_contract, sort_keys=True)
                    previous_numbers = self.meta("number_contract")
                    if previous_numbers and previous_numbers != serialized_numbers:
                        raise ValueError("Incompatible Qwen model or number acceptance policy in this match")
                    self.db.execute("INSERT OR REPLACE INTO metadata VALUES('number_contract',?)", (serialized_numbers,))
                payload = {key: value for key, value in node.items() if key != "feature_indices"}
                indices = node["feature_indices"]
                vectors = np.stack([features[i] for i in indices]).tolist()
                weights = np.stack([visibility[i] for i in indices]).tolist()
                stamp = digest([payload, vectors, weights])
                existing = self.db.execute("SELECT fingerprint FROM observations WHERE id=?", (node["node_id"],)).fetchone()
                if existing:
                    if existing[0] != stamp:
                        raise ValueError("Registered identity evidence changed; use a new match/run database rather than overwrite identity history")
                    continue
                candidates, comparable = [], []
                players = self.db.execute("SELECT * FROM players WHERE alias_of IS NULL AND status NOT IN ('needs_review','non_player') ORDER BY id").fetchall()
                for player in players:
                    members = self.db.execute("SELECT * FROM observations WHERE player=?", (player["id"],)).fetchall()
                    conflict = any(self._conflict(payload, json.loads(m["payload"])) for m in members)
                    galleries = [m for m in members if m["gallery"] and not m["detached"]]
                    distances = []
                    for member in galleries:
                        old = np.asarray(json.loads(member["features"]), dtype=np.float32)
                        old_vis = np.asarray(json.loads(member["visibility"]), dtype=np.float32)
                        current = np.asarray(vectors, dtype=np.float32)
                        matrix = part_distances(np.concatenate([current, old]),
                            np.concatenate([np.asarray(weights), old_vis]), min_common_parts=min_common_parts)
                        # Every constituent current track needs a comparable exemplar.
                        distances.append(float(np.max(np.min(matrix[:len(current), len(current):], axis=1))))
                    distance = min(distances, default=math.inf)
                    if math.isfinite(distance):
                        comparable.append(distance)
                        if not conflict:
                            candidates.append((distance, player["id"]))
                candidates.sort()
                best = candidates[0][0] if candidates else math.inf
                second = candidates[1][0] if len(candidates) > 1 else math.inf
                reasons = list(payload.get("hold_reasons", []))
                if (not reasons and max_distance is not None and best <= max_distance
                        and second - best >= min_margin):
                    pid, decision = candidates[0][1], "matched"
                elif not reasons and (not players or (comparable and min(comparable) >= novelty_distance)):
                    pid, decision = self._new_player("candidate"), "new_candidate"
                else:
                    reasons.append("ambiguous_or_uncomparable_identity")
                    payload["hold_reasons"] = reasons
                    pid, decision = self._new_player("needs_review"), "pending"
                payload["decision"] = decision
                payload["match_evidence"] = {
                    "candidates": [{"player_id": pid, "distance": d} for d, pid in candidates[:3]],
                    "match_distance": max_distance, "novelty_distance": novelty_distance,
                    "min_margin": min_margin,
                    "best_comparable_distance": min(comparable) if comparable else None,
                }
                # Fingerprint uses original evidence, not derived decision/reasons.
                gallery_count = self.db.execute("SELECT COUNT(*) FROM observations WHERE player=? AND gallery=1", (pid,)).fetchone()[0]
                gallery = int(decision != "pending" and gallery_count < gallery_limit)
                self.db.execute("INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,0)",
                    (node["node_id"], node["clip_name"], pid, pid, stamp,
                     json.dumps(payload, ensure_ascii=False), json.dumps(vectors), json.dumps(weights), gallery))
                changed.append(node["node_id"])
            for event in events:
                eid = "GE-" + digest([event["clip_name"], event["event_id"]])[:20]
                serialized = json.dumps(event, sort_keys=True)
                old = self.db.execute("SELECT payload FROM events WHERE id=?", (eid,)).fetchone()
                if old and old[0] != serialized:
                    raise ValueError("Registered event evidence changed; use a new run database")
                if not old:
                    self.db.execute("INSERT INTO events VALUES(?,?,?)", (eid, event["clip_name"], serialized))
                    changed.append(eid)
            if changed:
                self._record("ingest", {"added": changed})
        return self.snapshot()

    def review(self, operation, *, expected_revision=None, person_id=None, node_id=None, label=None):
        with self.db:
            # Acquire the write lock before checking revision across worker processes.
            self.db.execute("UPDATE metadata SET value=value WHERE key='match_id'")
            if self.meta("revision") != expected_revision:
                raise RuntimeError("Player library changed; refresh before editing")
            if operation in {"label_group", "confirm"}:
                player = self.db.execute("SELECT * FROM players WHERE id=? AND alias_of IS NULL", (person_id,)).fetchone()
                if not player:
                    raise FileNotFoundError("Unknown player")
                value = (label or "").strip()
                if len(value) > 120:
                    raise ValueError("Label too long")
                if value:
                    other = self.db.execute("SELECT * FROM players WHERE label=? AND id!=? AND alias_of IS NULL", (value, person_id)).fetchone()
                    if other:
                        left = self.db.execute("SELECT payload FROM observations WHERE player=?", (person_id,)).fetchall()
                        right = self.db.execute("SELECT payload FROM observations WHERE player=?", (other["id"],)).fetchall()
                        if any(self._conflict(json.loads(a[0]), json.loads(b[0])) for a in left for b in right):
                            raise ValueError("Cannot merge conflicting number or simultaneous track evidence")
                        self.db.execute("UPDATE observations SET player=? WHERE player=?", (other["id"], person_id))
                        self.db.execute("UPDATE players SET alias_of=? WHERE id=?", (other["id"], person_id))
                        person_id = other["id"]
                self.db.execute("UPDATE players SET label=?, status=? WHERE id=?",
                    (value or None, "manual_grouping" if value or operation == "confirm" else "candidate", person_id))
                # A review must not automatically add previously held samples to the gallery.
            elif operation == "exclude":
                player = self.db.execute("SELECT id FROM players WHERE id=? AND alias_of IS NULL", (person_id,)).fetchone()
                if not player:
                    raise FileNotFoundError("Unknown player")
                self.db.execute("UPDATE players SET status='non_player' WHERE id=?", (person_id,))
                self.db.execute("UPDATE observations SET gallery=0 WHERE player=?", (person_id,))
            elif operation in {"detach", "restore"}:
                row = self.db.execute("SELECT * FROM observations WHERE id=?", (node_id,)).fetchone()
                if not row:
                    raise FileNotFoundError("Unknown observation")
                if operation == "detach":
                    if row["detached"]:
                        raise ValueError("Observation is already detached")
                    pid = self._new_player("needs_review")
                    self.db.execute("UPDATE observations SET player=?, detached=1 WHERE id=?", (pid, node_id))
                else:
                    if not row["detached"]:
                        raise ValueError("Observation is not detached")
                    pid = row["auto_player"]
                    seen = set()
                    while True:
                        if pid in seen:
                            raise ValueError("Corrupt player alias cycle")
                        seen.add(pid)
                        alias = self.db.execute("SELECT alias_of FROM players WHERE id=?", (pid,)).fetchone()[0]
                        if alias is None:
                            break
                        pid = alias
                    members = self.db.execute("SELECT payload FROM observations WHERE player=? AND id!=?", (pid, node_id)).fetchall()
                    if any(self._conflict(json.loads(row["payload"]), json.loads(m[0])) for m in members):
                        raise ValueError("Restoring this observation conflicts with current player evidence")
                    self.db.execute("UPDATE observations SET player=?, detached=0 WHERE id=?", (pid, node_id))
            else:
                raise ValueError("Unsupported review operation; persistent identity history cannot be reset")
            revision = self._record(operation, {"player": person_id, "node": node_id, "label": label})
        return revision

    def snapshot(self):
        members, node_people, local_people, track_people = {}, {}, {}, {}
        for row in self.db.execute("SELECT * FROM observations ORDER BY clip,id"):
            node = json.loads(row["payload"])
            node["manually_detached"] = bool(row["detached"])
            node["gallery_eligible"] = bool(row["gallery"] and not row["detached"])
            members.setdefault(row["player"], []).append(node)
            node_people[node["node_id"]] = row["player"]
            local_people[(node["clip_name"], node["local_person_id"])] = row["player"]
            for tid in node["raw_track_ids"]:
                track_people[(node["clip_name"], tid)] = row["player"]
        events = []
        for row in self.db.execute("SELECT * FROM events ORDER BY clip,id"):
            event = json.loads(row["payload"])
            owners = {track_people[(row["clip"], tid)] for tid in event.get("raw_track_ids", []) if (row["clip"], tid) in track_people}
            pid = next(iter(owners)) if len(owners) == 1 else None
            if not event.get("raw_track_ids"):
                pid = local_people.get((row["clip"], event.get("local_person_id")))
            events.append({**event, "event_id": row["id"], "source_event_id": event["event_id"],
                           "id": pid, "global_person_id": pid, "identity_status": "mapped" if pid else "unmapped"})
        people, mappings = [], []
        for player in self.db.execute("SELECT * FROM players WHERE alias_of IS NULL ORDER BY id"):
            group = members.get(player["id"], [])
            if not group:
                continue
            cover = max(group, key=lambda n: float(n.get("cover", {}).get("archive_quality_score", 0)))
            own_events = [e for e in events if e["global_person_id"] == player["id"]]
            numbers = {n["jersey"]["number"] for n in group if (n.get("jersey") or {}).get("number") is not None}
            jersey = {"number": next(iter(numbers)) if len(numbers) == 1 else None,
                      "candidate_numbers": sorted(numbers), "status": "candidate_consensus" if len(numbers) == 1 else "conflict" if numbers else "unreadable"}
            counts = {}
            for e in own_events:
                counts[e["event"]] = counts.get(e["event"], 0) + 1
            people.append({"global_person_id": player["id"], "identity_label": player["label"],
                "status": player["status"], "members": group, "cover_node_id": cover["node_id"],
                "clip_count": len({n["clip_name"] for n in group}), "local_archive_count": len(group),
                "manual_member_count": len(group) if player["status"] == "manual_grouping" else 0,
                "event_count": len(own_events), "event_types": counts, "jersey": jersey})
            for node in group:
                mappings.append({"global_person_id": player["id"], **{k: node[k] for k in
                    ("node_id", "clip_name", "video_id", "local_person_id", "raw_track_ids", "intervals")}})
        aliases = {}
        for row in self.db.execute("SELECT id,alias_of FROM players WHERE alias_of IS NOT NULL"):
            pid, seen = row["alias_of"], {row["id"]}
            while pid not in seen:
                seen.add(pid)
                parent = self.db.execute("SELECT alias_of FROM players WHERE id=?", (pid,)).fetchone()[0]
                if parent is None:
                    aliases[row["id"]] = pid
                    break
                pid = parent
            else:
                raise ValueError("Corrupt player alias cycle")
        return {"people": people, "mappings": mappings, "events": events, "aliases": aliases, "revision": self.meta("revision"),
                "match_id": self.meta("match_id"), "feature_contract": self.meta("feature_contract")}
