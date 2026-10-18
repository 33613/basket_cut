"""TrackID3x3 Indoor/Outdoor, original video frames and SportsMOT checkpoint."""
import argparse
import json
from pathlib import Path
import subprocess

from contracts.execution import ExecutionSettings, normalize_omp_threads
from analysis.trackid.dataset import SPLIT, prepare
from analysis.teamtrack.dataset import save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--env-file',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--subset',choices=('all','Indoor','Outdoor'),default='all')
    p.add_argument('--limit-per-view',type=int,default=0,help='0=all; 1=shortest complete sequence per selected view for alignment')
    p.add_argument('--stage',choices=('prepare','infer','evaluate','preview','status','all'),default='all')
    args=p.parse_args()
    if args.limit_per_view<0:p.error('--limit-per-view must be nonnegative')
    if args.stage=='status':
        from analysis.trackid.benchmark import status
        print(json.dumps(status(args.output_dir),ensure_ascii=False,indent=2)); return
    if args.env_file:
        from cli.serve_web import load_environment
        load_environment(args.env_file)
    settings=ExecutionSettings(); normalize_omp_threads()
    output=args.output_dir.resolve()
    if args.stage in ('evaluate','all'):
        from analysis.teamtrack.metrics import trackeval_module
        trackeval_module()
    if args.stage in ('prepare','all'):
        if args.dataset_root is None:p.error('--dataset-root is required for preparation')
        prepare(args.dataset_root,output,subset=args.subset,limit_per_view=args.limit_per_view)
        save_json(output/'status.json',{'status':'prepared','dataset':'TrackID3x3','split':SPLIT})
    try:
        if args.stage=='infer':
            from analysis.trackid.benchmark import infer
            if args.device.startswith('cuda'):
                import os
                query=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                                     check=True,capture_output=True,text=True)
                busy=[p.strip() for p in query.stdout.splitlines() if p.strip() and p.strip()!=str(os.getpid())]
                if busy:raise RuntimeError('GPU occupied; finish initialization first. Active PIDs: '+', '.join(busy))
            infer(output,args.checkpoint or settings.motip_checkpoint,args.device)
        elif args.stage=='all':
            save_json(output/'status.json',{'status':'running','dataset':'TrackID3x3'})
            command=[str(settings.motip_python),'-u','-m','cli.benchmark_trackid',
                     '--output-dir',str(output),'--stage','infer',
                     '--checkpoint',str((args.checkpoint or settings.motip_checkpoint).resolve()),'--device',args.device]
            subprocess.run(command,cwd=settings.repository_root,check=True)
        if args.stage in ('evaluate','all'):
            from analysis.trackid.benchmark import evaluate
            rows=evaluate(output)
            print(json.dumps([r for r in rows if r['scope']!='sequence'],ensure_ascii=False,indent=2))
        if args.stage in ('preview','all'):
            from analysis.trackid.benchmark import preview
            preview(output)
            print('Alignment images:',output/SPLIT/'alignment')
    except Exception as exc:
        # Never erase a previous completed evaluation merely because a later
        # optional preview failed; record the operation failure separately.
        save_json(output/'last_error.json',{'stage':args.stage,'error':str(exc)})
        raise


if __name__=='__main__':main()
