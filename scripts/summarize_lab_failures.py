"""Group failure observations; labels describe outcomes rather than proven causes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--reference-log',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
report=json.loads((args.run/'summary.json').read_text())
reference={int(n):outcome=='SUCCESS' for n,outcome in
           re.findall(r'Episode (\d+) finished: (SUCCESS|FAIL)',args.reference_log.read_text())}
rows=report['episodes']
if len(rows)!=100 or len(reference)!=100:
    raise SystemExit('Expected the complete 100-episode confirmation replay and log.')
if any(reference[r['episode']]!=r['success'] for r in rows):
    raise SystemExit('Replayed outcomes differ from the confirmation log; investigate before grouping.')
failures=[]
for row in rows:
    if row['success']:
        continue
    cube,goal=row['final_cube_xyz'],row['goal_xyz']
    near=max(abs(cube[i]-goal[i]) for i in [0,1])<.05
    if row['max_cube_height_m']<.045:
        group='no_sustained_lift'
    elif near and cube[2]>.06:
        group='ends_elevated_near_bin'
    else:
        group='other'
    failures.append({**row,'observed_outcome':group})
result={'seed':report['seed'],'checkpoint_sha256':report['checkpoint_sha256'],
        'episodes':len(rows),'successes':sum(r['success'] for r in rows),
        'all_episode_outcomes_match_reference':True,
        'rules':{'no_sustained_lift':'Maximum sampled cube z < 45 mm',
                 'ends_elevated_near_bin':'Lifted; final XY within 50 mm on each bin axis and cube z > 60 mm',
                 'other':'All remaining failures'},
        'limitations':'Outcome labels are diagnostic thresholds, not causal explanations. Uses previously evaluated layouts; no new generalization claim.',
        'failure_counts':dict(Counter(r['observed_outcome'] for r in failures)),
        'failures':failures}
with args.out.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='failures'},indent=2))
