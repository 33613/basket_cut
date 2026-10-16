"""Synthetic CPU checks; no model inference or identity accuracy claims."""

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from analysis.bootstrap_audit import audit, constrained_groups, decision_reasons
from pipeline.identity.player_registry import PlayerRegistry
from pipeline.media import atomic_json, sha256_file
from workflows.person_library import export_library


class BootstrapAuditChecks(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix="basket-bootstrap-audit-")
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.run=self.root/"results"
        self.seed=self.run/"bootstrap"
        (self.seed/"media").mkdir(parents=True)
        source=self.root/"source.mp4";source.write_bytes(b"synthetic source evidence, never decoded")
        samples=[]
        for i in range(4):
            relative=f"media/{i}.png"
            Image.new("RGB",(30,60),("navy","white")[i%2]).save(self.seed/relative)
            samples.append({"sample_id":f"photo-{i//2}-person-{i%2}","frame_id":f"photo-{i//2}",
                "frame_idx":(i//2)*25,"timestamp_s":float(i//2),"crop_path":relative,"torso_path":relative,
                "crop_path_sha256":sha256_file(self.seed/relative),"torso_path_sha256":sha256_file(self.seed/relative),
                "bbox_xyxy":[0,0,30,60],"det_score":.9,"sharpness":100.,"quality_score":.8})
        photos={"status":"completed","protocol":{"source":{"path":str(source),"sha256":sha256_file(source)}},
            "samples":samples,"frames":[{"frame_id":"photo-0"},{"frame_id":"photo-1"}]}
        atomic_json(self.seed/"photos.json",photos)
        self.vectors=np.array([[[math.cos(a),math.sin(a)]]*2 for a in (0,.5,.02,.52)],dtype=np.float32)
        self.visible=np.ones((4,2),dtype=np.float32)
        np.savez_compressed(self.seed/"features.npz",embeddings=self.vectors,visibility_scores=self.visible)
        contract={"metadata":{"checkpoint_sha256":"synthetic","prompt_mode":"none","part_count":2,
            "feature_dimension":2,"test_embeddings":["parts"]},"shape":[2,2]}
        atomic_json(self.seed/"feature_contract.json",{**contract,"photos_sha256":sha256_file(self.seed/"photos.json"),
            "features_sha256":sha256_file(self.seed/"features.npz")})
        atomic_json(self.seed/"numbers.json",{"status":"completed","signature":{"photos_sha256":sha256_file(self.seed/"photos.json")},
            "readings":{s["sample_id"]:{"number":"23","raw_response":'{"number":"23","readable":true}'} for s in samples}})
        nodes=[{"node_id":s["sample_id"],"clip_name":s["frame_id"],"video_id":"source",
            "local_person_id":s["sample_id"],"raw_track_ids":[],"intervals":[],"cover":{},
            "evidence_kind":"bootstrap_photo","feature_indices":[i],"hold_reasons":[]} for i,s in enumerate(samples)]
        with PlayerRegistry(self.run/"library/players.sqlite3","synthetic-match") as registry:
            registry.ingest(nodes,self.vectors,self.visible,[],contract)
            export_library(registry,self.run,self.run/"library")

    def test_read_only_audit_exposes_gallery_lockout_and_exports_images(self):
        before={p:sha256_file(p) for p in self.run.rglob("*") if p.is_file()}
        report,files=audit(self.run,self.root/"audit")
        self.assertEqual(report["crop_count"],4)
        self.assertEqual(report["decision_counts"],{"new_candidate":1,"pending":2,"matched":1})
        self.assertEqual(report["pending_with_other_cross_frame_pending_within_0_2"],2)
        self.assertEqual(report["pending_gate_counts"]["novelty_threshold_not_reached"],2)
        self.assertEqual(report["pending_gate_counts"]["no_nonconflicting_registered_candidate"],2)
        self.assertFalse(report["accuracy_evaluated"])
        self.assertTrue(Path(files["archive"]).is_file())
        self.assertEqual(len(list((self.root/"audit/images").glob("*.jpg"))),4)
        self.assertTrue((self.root/"audit/evidence/features.npz").is_file())
        self.assertTrue(all(sha256_file(p)==digest for p,digest in before.items()))
        self.assertIn("跨帧近邻",Path(files["html"]).read_text())
        with self.assertRaises(FileExistsError): audit(self.run,self.root/"audit")

    def test_complete_link_prevents_transitive_same_photo_merges(self):
        matrix=np.array([[0,.01,.2],[.01,0,.01],[.2,.01,0]])
        groups=constrained_groups(matrix,["frame-0","frame-1","frame-0"],.3)
        self.assertEqual(sorted(map(len,groups)),[1,2])
        for group in groups:
            self.assertLessEqual(len({i for i in group if i in (0,2)}),1)
        separated=constrained_groups(np.array([[0,.1,.9],[.1,0,.1],[.9,.1,0]]),["a","b","c"],.2)
        self.assertEqual(len(separated),2)
        self.assertEqual(constrained_groups(np.empty((0,0)),[],.2),[])

    def test_saved_gates_distinguish_margin_and_distance(self):
        value={"decision":"pending","match_evidence":{"candidates":[{"distance":.1},{"distance":.12}],
            "best_comparable_distance":.1,"match_distance":.2,"novelty_distance":.5,"min_margin":.05}}
        self.assertIn("nearest_player_margin_too_small",decision_reasons(value))
        self.assertNotIn("match_threshold_exceeded",decision_reasons(value))
        value["match_evidence"]["candidates"][0]["distance"]=.3
        self.assertIn("match_threshold_exceeded",decision_reasons(value))

    def test_changed_features_and_overlapping_output_are_rejected(self):
        with self.assertRaises(ValueError): audit(self.run,self.run/"audit")
        (self.seed/"features.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"hashes/status"): audit(self.run,self.root/"changed")


def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BootstrapAuditChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__=="__main__":
    main()
