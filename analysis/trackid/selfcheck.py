"""Native video/GT conversion and real TrackEval checks; no real GPU model."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from analysis.trackid.dataset import SPLIT, parse_gt, prepare, sha256, sources
from analysis.trackid.benchmark import evaluate, infer, preview, status
from analysis.teamtrack.selfcheck import TeamTrackChecks


class TrackIDChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.raw=self.root/'raw'; self.raw.mkdir()
        self.rows=[]; self.files=[]
        for view,name,frames,fps in [('Indoor','basket_S1T1_pre',4,19.98),('Outdoor','IMG_0104',6,29.97)]:
            self.add_sequence(view,name,frames,fps)
        self.write_manifest()
        self.output=self.root/'out'; self.checkpoint=self.root/'model.pth'; self.checkpoint.write_bytes(b'synthetic')

    def add_sequence(self,view,name,frames,fps):
        video=self.raw/f'videos/{view}/{name}.mp4'; video.parent.mkdir(parents=True,exist_ok=True)
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),fps,(40,32))
        self.assertTrue(writer.isOpened())
        for i in range(frames):writer.write(np.full((32,40,3),i*20,dtype=np.uint8))
        writer.release()
        gt=self.raw/f'annotations/original/{view}/{name}.txt'; gt.parent.mkdir(parents=True,exist_ok=True)
        gt.write_text(''.join(f'{i},1,2,2,8,8,-1,-1,-1,-1\n' for i in range(1,frames+1)))
        self.rows.append(dict(subset=view,sequence=name,video_path=str(video.relative_to(self.raw)),
            gt_path=str(gt.relative_to(self.raw)),width=40,height=32,fps_average=str(fps),
            actual_decoded_frames=frames,gt_rows=frames,gt_annotated_frames=frames,identity_count=1,check_status='verified'))
        self.files.extend([video,gt])

    def write_manifest(self):
        path=self.raw/'manifest.csv'
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=self.rows[0]); writer.writeheader(); writer.writerows(self.rows)
        (self.raw/'checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.relative_to(self.raw)}\n' for p in [*self.files,path]))

    def mock_model(self,fail_on_frame=None):
        return TeamTrackChecks.fake_model_modules(self,fail_on_frame)

    def test_placeholders_become_evaluable_players_without_changing_raw(self):
        before=[p.read_bytes() for p in self.files]
        manifest=prepare(self.raw,self.output)
        self.assertEqual(sum(s['gt_boxes'] for s in manifest['sequences']),10)
        self.assertEqual(before,[p.read_bytes() for p in self.files])
        self.assertEqual([s['frames'] for s in manifest['sequences']],[4,6])
        self.assertAlmostEqual(manifest['sequences'][0]['fps'],19.98,places=2)
        self.assertAlmostEqual(manifest['sequences'][1]['fps'],29.97,places=2)
        for seq in manifest['sequences']:
            row=(self.output/SPLIT/'gt'/seq['sequence']/'gt/gt.txt').read_text().splitlines()[0]
            self.assertEqual(row,'1,1,2.0,2.0,8.0,8.0,1,1,1')
        self.assertEqual(prepare(self.raw,self.output),manifest)
        self.assertEqual(len(list((self.output/SPLIT/'alignment').rglob('gt-*.jpg'))),6)

    def test_missing_macos_metadata_is_not_an_evaluation_input(self):
        checksum = self.raw/'checksums.sha256'
        with checksum.open('a') as stream:
            for name in ('logs/.DS_Store','videos/Indoor/._basket_S1T1_pre.mp4',
                         '__MACOSX/metadata.txt'):
                stream.write('0'*64+'  '+name+'\n')
        result=prepare(self.raw,self.output)
        self.assertEqual(len(result['sequences']),2)
        self.assertEqual(sum(s['gt_boxes'] for s in result['sequences']),10)
        self.assertFalse((self.raw/'logs/.DS_Store').exists())

    def test_missing_real_gt_or_non_macos_sidecar_still_fails(self):
        self.files[1].unlink()
        with self.assertRaisesRegex(ValueError,'Missing dataset checksum file'):
            sources(self.raw)
        self.files[1].write_text('restored but invalid')
        self.write_manifest()
        with (self.raw/'checksums.sha256').open('a') as stream:
            stream.write('0'*64+'  metadata/missing-real-report.json\n')
        with self.assertRaisesRegex(ValueError,'Missing dataset checksum file'):
            sources(self.raw)

    def test_macos_named_checksum_cannot_escape_dataset_root(self):
        with (self.raw/'checksums.sha256').open('a') as stream:
            stream.write('0'*64+'  ../.DS_Store\n')
        with self.assertRaisesRegex(ValueError,'unsafe'):
            sources(self.raw)

    def test_upload_warning_label_does_not_replace_actual_validation(self):
        self.rows[1]['check_status']='verified_with_warnings'
        self.write_manifest()
        result=prepare(self.raw,self.output)
        outdoor=next(s for s in result['sequences'] if s['view']=='Outdoor')
        self.assertEqual(outdoor['source_manifest_row']['check_status'],'verified_with_warnings')
        self.assertEqual(outdoor['frames'],6)
        self.assertAlmostEqual(outdoor['fps'],29.97,places=2)
        self.assertEqual(outdoor['gt_boxes'],6)

    def test_warning_status_cannot_hide_invalid_gt_or_missing_frames(self):
        self.rows[1]['check_status']='warning'
        self.rows[1]['actual_decoded_frames']=7
        self.write_manifest()
        with self.assertRaisesRegex(ValueError,'decoded 6'):
            prepare(self.raw,self.output)
        self.assertFalse((self.output/SPLIT/'dataset.json').exists())
        self.rows[1]['actual_decoded_frames']=6
        self.files[3].write_text('1,1,0,0,-1,1,-1,-1,-1,-1\n')
        self.write_manifest()
        with self.assertRaisesRegex(ValueError,'geometry'):
            prepare(self.raw,self.output)

    def test_selected_explicit_failures_and_error_counts_are_rejected(self):
        self.rows[1]['check_status']='failed: invalid decode'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError,'inspection reports'):
            sources(self.raw)
        # A failing unselected view must not block a valid Indoor-only run.
        self.assertEqual(len(sources(self.raw,subset='Indoor')),1)
        self.rows[1]['check_status']='warning'
        for row in self.rows:row['decode_errors']=0
        self.rows[1]['decode_errors']=1
        self.write_manifest()
        with self.assertRaisesRegex(ValueError,'decode_errors=1'):
            sources(self.raw)

    def test_inspect_reports_original_status_without_gpu_or_output_dir(self):
        self.rows[1]['check_status']='verified_with_warnings'
        self.write_manifest()
        result=subprocess.run([sys.executable,'-m','cli.benchmark_trackid',
            '--dataset-root',str(self.raw),'--stage','inspect'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(result.stdout)
        self.assertEqual(data['rows'],2)
        self.assertEqual(data['status_counts']['verified_with_warnings'],1)
        self.assertEqual(data['non_verified_rows'][0]['sequence'],'IMG_0104')
        self.assertFalse(self.output.exists())

    def make_outdoor_native_mot9(self):
        path=self.files[3]
        rows=[]
        for i,line in enumerate(path.read_text().splitlines()):
            row=line.split(',')[:6]+(['0'] if i == 0 else ['0.83'])+['-1','-1']
            rows.append(','.join(row))
        path.write_text('\n'.join(rows)+'\n')
        for row in self.rows:row['gt_columns']=10 if row['subset']=='Indoor' else 9
        self.write_manifest()

    def test_native_outdoor_mot9_score_is_not_a_class_or_filter(self):
        self.make_outdoor_native_mot9()
        before=self.files[3].read_bytes()
        result=prepare(self.raw,self.output)
        outdoor=next(s for s in result['sequences'] if s['view']=='Outdoor')
        self.assertEqual(outdoor['gt_boxes'],6)
        self.assertEqual(outdoor['gt_format']['source_columns'],9)
        self.assertEqual(outdoor['gt_format']['source_zero_score_rows_retained'],1)
        self.assertEqual(self.files[3].read_bytes(),before)
        gt=self.output/SPLIT/'gt'/outdoor['sequence']/'gt/gt.txt'
        self.assertTrue(all(line.endswith(',1,1,1') for line in gt.read_text().splitlines()))
        modules,_=self.mock_model()
        with patch.dict(sys.modules,modules):infer(self.output,self.checkpoint,device='cpu')
        self.assertEqual(evaluate(self.output)[-1]['TP'],10)

    def test_official_fixed_revision_annotation_examples(self):
        fixture=Path(__file__).parent/'fixtures/official-layout-examples.json'
        data=json.loads(fixture.read_text())
        for example in data['examples']:
            path=self.root/'official.txt'
            path.write_text(''.join(','.join(row)+'\n' for row in example['first_rows']))
            grouped,metadata=parse_gt(path,2,include_metadata=True)
            self.assertEqual(len(grouped[1]),2)
            self.assertEqual(grouped[1][0][1:],tuple(map(float,example['first_rows'][0][2:6])))
            self.assertEqual(metadata['source_columns'],len(example['first_rows'][0]))

    def test_invalid_unknown_and_mixed_layouts_are_not_silently_truncated(self):
        path=self.root/'bad-layout.txt'
        bad=('1,1,1,1,10,10,0.8,-1\n','1,1,1,1,10,10,0.8,-1,-1,extra,extra\n',
             '1,1,1,1,10,10,nan,-1,-1\n','1,1,1,1,10,10,1.2,-1,-1\n',
             '1,1,1,1,10,10,0.8,1,-1\n',
             '1,1,1,1,10,10,-1,-1,-1,-1\n2,1,1,1,10,10,0.8,-1,-1\n')
        for text in bad:
            path.write_text(text)
            with self.subTest(text=text),self.assertRaises(ValueError):parse_gt(path,2)

    def test_manifest_column_count_must_match_actual_native_layout(self):
        self.make_outdoor_native_mot9()
        self.rows[1]['gt_columns']=10
        self.write_manifest()
        with self.assertRaisesRegex(ValueError,'GT column count'):
            prepare(self.raw,self.output)

    def test_selection_uses_shortest_complete_sequence_per_view(self):
        self.add_sequence('Indoor','short',2,20); self.write_manifest()
        result=sources(self.raw,limit_per_view=1)
        self.assertEqual([s['source_sequence'] for s in result],['short','IMG_0104'])
        self.assertEqual(len(sources(self.raw,subset='Outdoor')),1)
        with self.assertRaises(ValueError):sources(self.raw,limit_per_view=-1)

    def test_duplicate_ids_wrong_frame_nonfinite_and_unknown_tail_fail(self):
        path=self.root/'bad.txt'
        for content in ('0,1,0,0,1,1,-1,-1,-1,-1\n','1,1,0,0,0,1,-1,-1,-1,-1\n',
            '1,1,nan,0,1,1,-1,-1,-1,-1\n','1,1,0,0,1,1,1,1,1,-1\n',
            '1,1,0,0,1,1,-1,-1,-1,-1\n'*2,'3,1,0,0,1,1,-1,-1,-1,-1\n'):
            path.write_text(content)
            with self.subTest(content=content),self.assertRaises(ValueError):parse_gt(path,2)

    def test_empty_gt_frames_and_out_of_image_boxes_are_not_filled_or_clipped(self):
        path=self.root/'gt.txt'; path.write_text('2,0,-1,2,10,8,-1,-1,-1,-1\n')
        grouped=parse_gt(path,3)
        self.assertEqual(set(grouped),{2}); self.assertEqual(grouped[2][0][1],-1)

    def test_manifest_frame_mismatch_fails_before_evaluation(self):
        self.rows[0]['actual_decoded_frames']=5; self.write_manifest()
        with self.assertRaisesRegex(ValueError,'decoded 4'):
            prepare(self.raw,self.output)
        self.assertFalse((self.output/SPLIT/'dataset.json').exists())

    def test_checksum_source_and_prepared_gt_mutation_are_refused(self):
        manifest=prepare(self.raw,self.output)
        gt=self.output/SPLIT/'gt'/manifest['sequences'][0]['sequence']/'gt/gt.txt'
        gt.write_text('changed')
        with self.assertRaisesRegex(ValueError,'Prepared GT changed'):prepare(self.raw,self.output)
        self.files[1].write_text('changed')
        with self.assertRaisesRegex(ValueError,'checksum'):sources(self.raw)

    def test_duplicate_selection_and_path_escape_are_refused(self):
        self.rows.append(dict(self.rows[0])); self.write_manifest()
        with self.assertRaisesRegex(ValueError,'duplicate'):sources(self.raw)
        self.rows.pop(); self.rows[0]['video_path']='../outside.mp4'; self.write_manifest()
        with self.assertRaisesRegex(ValueError,'unsafe'):sources(self.raw)

    def test_full_inference_evaluation_and_native_previews(self):
        prepare(self.raw,self.output); modules,calls=self.mock_model()
        with patch.dict(sys.modules,modules):infer(self.output,self.checkpoint,device='cpu')
        self.assertEqual((calls['loads'],calls['trackers']),(1,2))
        self.assertFalse(modules['torch'].backends.cuda.matmul.allow_tf32)
        self.assertFalse(modules['torch'].backends.cudnn.allow_tf32)
        results=evaluate(self.output)
        self.assertEqual({r['view'] for r in results if r['scope']=='view'},{'Indoor','Outdoor'})
        overall=results[-1]
        self.assertEqual((overall['TP'],overall['FP'],overall['FN']),(10,0,0))
        for key in ('HOTA','DetA','AssA','MOTA','IDF1','recall','precision'):self.assertAlmostEqual(overall[key],100)
        self.assertEqual(status(self.output)['completed_sequences'],2)
        previews=preview(self.output)
        self.assertEqual(sum(len(p['images']) for p in previews),6)
        self.assertTrue(all(Path(i).is_file() for p in previews for i in p['images']))
        with (self.output/SPLIT/'metrics/hota_thresholds.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))),19*5)
        with (self.output/SPLIT/'metrics/sequences/Outdoor__IMG_0104/clear_frames.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))),6)

    def test_resume_reuses_model_results_but_rechecks_raw_gt(self):
        prepare(self.raw,self.output); modules,calls=self.mock_model()
        with patch.dict(sys.modules,modules):
            infer(self.output,self.checkpoint,device='cpu')
            infer(self.output,self.checkpoint,device='cpu')
        self.assertEqual(calls['loads'],1)
        self.files[1].write_text('changed')
        with patch.dict(sys.modules,modules),self.assertRaisesRegex(ValueError,'source GT changed'):
            infer(self.output,self.checkpoint,device='cpu')
        with self.assertRaisesRegex(ValueError,'original video/GT changed'):evaluate(self.output)

    def test_incomplete_results_are_never_evaluated(self):
        prepare(self.raw,self.output); modules,_=self.mock_model(fail_on_frame=2)
        with patch.dict(sys.modules,modules),self.assertRaisesRegex(RuntimeError,'interrupted'):
            infer(self.output,self.checkpoint,device='cpu')
        self.assertEqual(status(self.output)['completed_sequences'],0)
        with self.assertRaises(FileNotFoundError):evaluate(self.output)
        self.assertFalse((self.output/'status.json').is_file())

    def test_changed_completed_prediction_or_summary_fails_status(self):
        manifest=prepare(self.raw,self.output); modules,_=self.mock_model()
        with patch.dict(sys.modules,modules):infer(self.output,self.checkpoint,device='cpu')
        evaluate(self.output)
        summary=self.output/SPLIT/'metrics/summary.csv'; original=summary.read_bytes()
        summary.write_text('changed')
        with self.assertRaisesRegex(ValueError,'result checksum'):status(self.output)
        summary.write_bytes(original)
        pred=self.output/SPLIT/'predictions/motip/data'/f'{manifest["sequences"][0]["sequence"]}.txt'
        pred.write_text('changed')
        with self.assertRaisesRegex(ValueError,'completion evidence'):status(self.output)

    def test_cli_prepare_and_status_are_cpu_only(self):
        result=subprocess.run([sys.executable,'-m','cli.benchmark_trackid','--dataset-root',str(self.raw),
            '--output-dir',str(self.output),'--stage','prepare','--limit-per-view','1'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run([sys.executable,'-m','cli.benchmark_trackid','--output-dir',str(self.output),
            '--stage','status'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['selected_sequences'],2)


def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TrackIDChecks))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
