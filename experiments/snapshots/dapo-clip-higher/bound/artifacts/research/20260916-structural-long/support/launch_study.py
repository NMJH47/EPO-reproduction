import hashlib,json,os,subprocess
from pathlib import Path
from datetime import datetime,timezone
root=Path('/home/ubuntu22/EPO-experiments/frontier-20260914')
out=root/'artifacts/research/20260916-structural-long'
plan=out/'study_plan.json'
expected='fd6b372cac8ae991a3a0b06b8b1e3fb6fd23bd8165ea53fd117452f6ce2229eb'
assert hashlib.sha256(plan.read_bytes()).hexdigest()==expected
cmd=['/home/ubuntu22/anaconda3/envs/epo/bin/python','-B','-u',str(out/'support/run_structural_study.py'),'--plan',str(plan),'--plan-sha256',expected,'--execute']
env=dict(os.environ);env.update(PATH='/home/ubuntu22/anaconda3/envs/epo/bin:'+env.get('PATH',''),LD_LIBRARY_PATH='/home/ubuntu22/anaconda3/envs/epo/lib',PYTHONPATH=str(root/'rl_games')+':'+str(root),CUDA_VISIBLE_DEVICES='0',PYTORCH_NO_CUDA_MEMORY_CACHING='1',PYTHONUNBUFFERED='1')
for k in ('WANDB_RUN_ID','WANDB_RESUME','WANDB_MODE'):env.pop(k,None)
def write(record):
 tmp=out/'launcher_receipt.tmp';tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(out/'launcher_receipt.json')
with (out/'controller_stdout.log').open('x') as f:
 child=subprocess.Popen(cmd,cwd=str(root),env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 fields=Path('/proc/%d/stat'%child.pid).read_text().rsplit(')',1)[1].split()
 r=dict(pid=child.pid,start_ticks=int(fields[19]),command=cmd,started_at_utc=datetime.now(timezone.utc).isoformat(),status='running',exit_code=None);write(r)
 r.update(exit_code=child.wait(),actual_exit_code_source='subprocess.wait',ended_at_utc=datetime.now(timezone.utc).isoformat());r['status']='completed' if r['exit_code']==0 else 'failed';write(r)
