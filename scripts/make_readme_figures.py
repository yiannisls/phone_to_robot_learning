"""Create README figures from recorded footage and saved experiment summaries."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
root=args.root
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.facecolor':'#f8fafc','axes.facecolor':'#f8fafc',
                     'text.color':'#172b4d','axes.labelcolor':'#172b4d','xtick.color':'#52637a','ytick.color':'#52637a'})

def frame(path,second):
    cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_MSEC,second*1000)
    ok,bgr=cap.read();cap.release()
    if not ok:raise RuntimeError(f'Cannot read {path} at {second}s')
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)

phone=frame(root/'data/raw/IMG_5539.mp4',4.5)
robot=frame(root/'results/bc_v2_validation/validation.avi',8.5)
fig=plt.figure(figsize=(12,6),layout='constrained')
grid=fig.add_gridspec(1,2,width_ratios=[1,2.2])
for ax,img,title in [(fig.add_subplot(grid[0]),phone,'01  Personally recorded phone footage'),
                     (fig.add_subplot(grid[1]),robot,'02  Learned policy in simulation')]:
    ax.imshow(img);ax.axis('off');ax.set_title(title,loc='left',fontsize=12,fontweight='bold',pad=14)
fig.savefig(args.out/'phone-to-robot.png',dpi=145)
plt.close(fig)

transport=np.genfromtxt(root/'data/processed/IMG_5539_transport.csv',delimiter=',',names=True)
fig,ax=plt.subplots(figsize=(10,3.5),layout='constrained')
ax.fill_between(transport['time_seconds'],transport['height_m']*100,color='#d8e9ef')
ax.plot(transport['time_seconds'],transport['height_m']*100,color='#007f89',lw=3)
ax.scatter([transport['time_seconds'][0],transport['time_seconds'][-1]],
           [transport['height_m'][0]*100,transport['height_m'][-1]*100],color='#007f89',zorder=3)
ax.set(xlabel='Time after retiming (seconds)',ylabel='Adapted lift (cm)',ylim=(-.5,11))
ax.set_title('From a phone recording to a robot transport path',loc='left',fontweight='bold',pad=17)
ax.grid(axis='y',alpha=.18)
fig.savefig(args.out/'phone-transport.png',dpi=145);plt.close(fig)

summary=json.loads((root/'docs/reproduction-results.json').read_text())
# The summary includes episode-level evidence as well as the aggregate groups.
summary=summary['summary']
if 'normal' not in summary:
    print(list(summary))
    raise RuntimeError('Unexpected summary structure')
fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True,layout='constrained')
for ax,key,title in zip(axes,['normal','opening_pulse'],['Ordinary execution','One artificial early-opening request']):
    counts=[summary[key][name]['task_successes'] for name in ['bc','wm']]
    bars=ax.bar(['BC','BC + world model'],counts,width=.55,color=['#7087a0','#007f89'])
    for bar,n in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,n+.35,f'{n}/20',ha='center',fontweight='bold')
    ax.set_title(title,fontsize=11,pad=16);ax.set_ylim(0,23);ax.set_yticks([0,5,10,15,20]);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
axes[0].set_ylabel('Successful transfers')
fig.savefig(args.out/'control-results.png',dpi=145);plt.close(fig)
manifest={'figures':{'gripper-comparison.gif':'results/world_model_control_comparison.mp4; ffmpeg fps=5, width=800, 96-colour palette; saved-state replay.','phone-to-robot.png':'IMG_5539 at 4.5s; results/bc_v2_validation/validation.avi at 8.5s; frames are separate demonstrations, not synchronized.',
                    'phone-transport.png':'Saved IMG_5539_transport.csv; smoothed, retargeted and retimed lift, not unmodified physical reconstruction.',
                    'control-results.png':'docs/reproduction-results.json; 80 reruns on saved development layouts.'}}
(args.out/'figure-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
