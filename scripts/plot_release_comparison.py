"""Plot the successful and stalled releases from diagnostic episodes 2 and 3."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True,layout='constrained')
for episode,label in [(2,'Episode 2: success'),(3,'Episode 3: stalled')]:
    with np.load(args.run/f'episode_{episode:03d}.npz') as d:
        trace,actions=d['trace'],d['actions']
    time=np.arange(1,len(trace)+1)*.1
    axes[0].plot(time,trace[:,2]*100,label=label)
    axes[1].plot(time,actions[:,3],label=label)
    axes[2].plot(time,np.linalg.norm(trace[:,:3]-trace[:,3:6],axis=1)*100,label=label)
axes[0].set_ylabel('Cube height (cm)')
axes[1].set_ylabel('Requested jaw target (rad)')
axes[2].set_ylabel('Cube–hand distance (cm)')
axes[2].set_xlabel('Simulation time (seconds)')
axes[0].legend()
for ax in axes:ax.grid(alpha=.25)
fig.suptitle('Frozen BC + DAgger policy: release comparison (seed 987)')
args.out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.out,dpi=150)
