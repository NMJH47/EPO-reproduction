#!/usr/bin/env python3
"""Fixed DAPO study queue. Default inert; no resume, retry or score-selected run."""
import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

PYTHON = '/home/ubuntu22/anaconda3/envs/epo/bin/python'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/linked input: ' + str(path))
    before = path.stat()
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''):
            sha.update(block); md5.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'File changed: ' + str(path))
    return {'path': str(path), 'sha256': sha.hexdigest(), 'md5': base64.b64encode(md5.digest()).decode(), 'size': after.st_size}


def digest(path):
    return identity(path)['sha256']


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def process_identity(pid):
    try:
        path = Path('/proc') / str(pid)
        fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z': return None
        return {'pid': pid, 'start_ticks': int(fields[19]), 'uid': path.stat().st_uid,
                'argv': [v.decode() for v in (path/'cmdline').read_bytes().split(b'\0') if v]}
    except FileNotFoundError:
        return None


def environment(root):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('EPO_CLIP_') or key in ('WANDB_RUN_ID', 'WANDB_RESUME', 'WANDB_MODE', 'CUDA_LAUNCH_BLOCKING', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE'):
            env.pop(key, None)
    prefix = Path(PYTHON).parent
    env.update(PATH=str(prefix)+':'+env.get('PATH',''), LD_LIBRARY_PATH=str(prefix.parent/'lib'),
               PYTHONPATH=str(root/'rl_games')+':'+str(root), CUDA_VISIBLE_DEVICES='0',
               PYTORCH_NO_CUDA_MEMORY_CACHING='1', PYTHONUNBUFFERED='1')
    return env


def command(entry, plan_path):
    return [PYTHON, '-B', '-u', str(entry), '--plan', str(plan_path), '--plan-sha256', digest(plan_path), '--execute']


def validate(study, path):
    root, output = Path(study['execution_root']), Path(study['output'])
    require(study['purpose'] == 'dapo_transfer_study' and type(study['execution_ready']) is bool, 'Wrong/unbound study')
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'Invalid isolated execution root')
    require(output.resolve() == output and root/'artifacts' in output.parents and not output.exists(), 'Fresh isolated controller output required')
    for name in ('run_id','entity','project','group'):
        require(isinstance(study[name],str) and study[name] and '/' not in study[name], 'Explicit identity required: '+name)
    files = study['files_sha256']
    roles = ('train_entry','evaluation_entry','analysis_entry','monitor_entry','monitor_source','nvml_source','base_source','evaluator_source')
    required = {str(Path(__file__).resolve())} | {study[x] for x in roles}
    require(all(root in Path(study[x]).resolve().parents for x in roles), 'Executing source outside isolated root')
    require(str(Path(path).resolve()) not in files, 'Study plan may not pin itself')
    jobs = study['train_jobs']
    selected = study['pilot_jobs'] + study['formal_jobs']
    require(len(set(selected)) == len(selected) == 8 and set(selected) <= set(jobs), 'Exactly two pilot and six formal jobs required')
    orders = [('e51','candidate',35001,306),('e51','control',35001,306),
              ('fresh','candidate',35011,15259),('fresh','control',35011,15259),
              ('fresh','control',35012,15259),('fresh','candidate',35012,15259),
              ('fresh','candidate',35013,15259),('fresh','control',35013,15259)]
    require(len(study['pilot_jobs']) == 2 and len(study['formal_jobs']) == 6, 'Phase order differs')
    identities = [study['run_id']]; outputs = [str(output)]
    for key, expected in zip(selected, orders):
        require(key and Path(key).name == key, 'Unsafe phase key')
        job = jobs[key]; required.add(job['plan_path'])
        require(files.get(job['plan_path']) == job['plan_sha256'] == digest(job['plan_path']), 'Training plan hash differs')
        plan = load(job['plan_path'])
        require((plan['mode'],plan['arm'],plan['seed'],plan['epochs']) == expected and plan['execution_root']==str(root)
                and plan['execution_ready'] is True, 'Fixed training grid/ready/root differs: '+key)
        require(plan['files_sha256'].get(study['train_entry']) == files[study['train_entry']], 'Training entry SHA not bound')
        identities.extend([plan['run_id'],job['monitor_run_id']]); outputs.extend([plan['output'],job['monitor_output']])
        specs = study['evaluation_specs'][key]
        require([s['eval_seed'] for s in specs] == ([15001,15002] if expected[0]=='e51' else [16001,16002]), 'Fixed eval grid differs')
        for item in specs:
            identities.extend([item['run_id'],item['monitor_run_id']]); outputs.extend([item['output'],item['monitor_output']])
    require(len(study['qualification_completed']) == 4, 'Three training and one evaluator qualification required')
    kinds=[]
    for item in study['qualification_completed']:
        require(item['kind'] in ('training','evaluation'), 'Unknown qualification kind')
        required.update([item['plan_path'],item['launch_receipt'],item['verification_receipt'],item['monitor_wait_receipt']])
        require(files.get(item['plan_path']) == item['plan_sha256'] == digest(item['plan_path']), 'Qualification plan differs')
        q=load(item['plan_path']);kinds.append((item['kind'],q.get('mode'),q.get('arm')))
        if item['kind']=='training': require(q['epochs']==1,'Qualification must have one epoch')
    unused={jobs[key]['plan_path'] for key in set(jobs)-set(selected)}
    qualified_training={item['plan_path'] for item in study['qualification_completed'] if item['kind']=='training'}
    require(unused <= qualified_training, 'Unused nonqualification training job is not allowed')
    require(kinds == [('training','e51','candidate'),('training','fresh','control'),
                      ('training','fresh','candidate'),('evaluation',None,None)], 'Fixed qualification sequence differs')
    fixture=study['analysis_fixture'];required.add(fixture['summary_path'])
    require(files.get(fixture['summary_path']) == fixture['summary_sha256'], 'Analysis fixture binding differs')
    for stage in ('pilot','formal'):
        identities.append(study['analysis_specs'][stage]['run_id']);outputs.append(study['analysis_specs'][stage]['output'])
    require(len(identities)==len(set(identities)) and all(isinstance(x,str) and x.isalnum() for x in identities), 'Duplicate/invalid run ID')
    require(len(outputs)==len(set(outputs)), 'Output collision')
    for candidate in outputs:
        p=Path(candidate)
        require(p.is_absolute() and p.resolve()==p and root/'artifacts' in p.parents and not p.exists(), 'Existing/invalid future output: '+candidate)
    for name,source in study.get('extra_plan_files',{}).items():
        require(Path(name).name==name and name not in ('study_plan.json','analysis_fixture_summary.json'), 'Unsafe prospective attachment name')
        required.add(source)
    require(required <= set(files), 'Study lacks frozen executed source/plan/qualification evidence')
    for source,sha in files.items():
        require(digest(source)==sha, 'Study input changed: '+source)
    return {'status':'prepared','gpu_started':False,'api_called':False,'training_jobs':8,
            'qualification_records':4,'pilot_evaluations':4,'formal_evaluations':12}


def normalized_record(item):
    return {'path':item.get('path',item.get('local_path')), 'sha256':item['sha256'],
            'md5':item.get('md5',item.get('md5_base64')), 'size':item['size']}


def verify_remote(wandb, study, run_id, proofs):
    """Fresh run lookup, exact actual inventories; no checkpoint download/copy."""
    run=wandb.Api(timeout=60).run('/'.join([study['entity'],study['project'],run_id]))
    require(run.state=='finished','Remote run is not finished: '+run_id)
    logged=list(run.logged_artifacts());verified=[]
    for proof in proofs:
        name=proof.get('artifact',proof.get('name'));kind=proof['type']
        require(proof['state']=='COMMITTED','Local receipt did not commit')
        matches=[x for x in logged if x.name==name and x.type==kind]
        require(len(matches)==1 and matches[0].state=='COMMITTED','Missing/ambiguous remote artifact '+str(name))
        entries=matches[0].manifest.entries;records=proof['files']
        require(set(entries)==set(records),'Remote inventory differs')
        for key,original in records.items():
            record=normalized_record(original);actual=identity(record['path'])
            require(actual==record,'Local original differs: '+key)
            require(entries[key].size==record['size'] and entries[key].digest==record['md5'],'Remote bytes differ: '+key)
        verified.append({'artifact':name,'type':kind,'state':'COMMITTED','file_count':len(records),
                         'bytes':sum(x['size'] for x in records.values())})
    return {'run_id':run_id,'state':run.state,'verified_at':now(),'artifacts':verified}


def verify_training(wandb, study, plan, wait):
    require(wait['exit_code']==0 and wait['actual_exit_code_source']=='subprocess.wait','Actual training wait0 required')
    out=Path(plan['output']);summary=load(out/'summary.json');finish=load(out.with_name(out.name+'-finish-attempt.json'))
    require(load(out/'plan.json')==plan, 'Training executed a different resolved plan')
    config_record=load(out/'executed_training_config_manifest.json')['files']
    live_cfg=[Path(p) for p in config_record if p.endswith('/config.yaml')]
    require(len(live_cfg)==1, 'One actual native config path required')
    native_wb=load(live_cfg[0].parent/'wandb_run.json')
    require(native_wb['id']==plan['run_id'] and native_wb['state']=='finished' and native_wb['exit_code']==0
            and native_wb['mode']=='online', 'Native training run receipt differs')
    require(summary['status']=='completed' and summary['effective_exit_code']==0 and summary['mode']==plan['mode']
            and summary['arm']==plan['arm'] and summary['seed']==plan['seed'],'Training scope/status differs')
    require(finish['run_finish_entered']==finish['run_finish_returned']==1 and finish['native_observer_finish_returned']
            and not finish['errors'],'Original owned finish incomplete')
    counts=summary['counts'];epochs=plan['epochs']
    for name,per in (('epoch_entries',1),('epoch_returns',1),('actor_attempts',8),('actor_returns',8),
                     ('env_step_attempts',16),('env_step_returns',16),('scheduler_calls',2),('scheduler_returns',2),
                     ('lr_returns',2),('history_releases',1)):
        require(counts[name]==epochs*per,'Training terminal quota differs: '+name)
    require(summary['physical_frames']==epochs*327680 and counts['adam_entered']==counts['adam_returned']
            and counts['adam_exceptions']==0,'Physical/actual Adam accounting differs')
    if epochs==1: require(counts['adam_returned']>0,'Technical qualification did not execute Adam')
    require(load(out/'terminal-finite.json')['status']=='completed','Live terminal finite check failed')
    checkpoint=summary['terminal_checkpoint'];original=identity(checkpoint['path'])
    require({k:original[k] for k in ('sha256','md5','size')}==checkpoint['identity'],'Checkpoint differs')
    start=5000069120 if plan['mode']=='e51' else 0;start_epoch=15259 if plan['mode']=='e51' else 0
    require(checkpoint['frame']==start+epochs*327680 and checkpoint['epoch']==start_epoch+epochs,'Endpoint clock differs')
    config=out/'executed_training_config.yaml';manifest=load(out/'executed_training_config_manifest.json')['files']
    for path,expected in manifest.items():
        require({k:v for k,v in identity(path).items() if k!='path'}==expected,'Executed config changed')
    proofs=[load(out/'plan-artifact-wait.json'),load(out/'result-artifact-wait.json')]
    require([p['type'] for p in proofs]==['dapo-training-plan','dapo-training-result']
            and all(p['run_id']==plan['run_id'] for p in proofs),'Wrong training artifacts')
    bound={normalized_record(x)['path']:normalized_record(x) for x in proofs[1]['files'].values()}
    for path in (str(config),checkpoint['path'],str(out/'summary.json'),str(out/'terminal-finite.json')):
        require(path in bound and identity(path)==bound[path],'Terminal evidence not in original training artifact: '+path)
    remote=verify_remote(wandb,study,plan['run_id'],proofs)
    return {'status':'completed','run_id':plan['run_id'],'checkpoint':checkpoint['path'],
            'checkpoint_sha256':original['sha256'],'train_config':str(config),'train_config_sha256':digest(config),
            'training_seed':plan['seed'],'arm':plan['arm'],'remote':remote,'source_summary_sha256':digest(out/'summary.json')}


def verify_evaluation(wandb,study,plan,wait):
    require(wait['exit_code']==0 and wait['actual_exit_code_source']=='subprocess.wait','Actual evaluation wait0 required')
    out=Path(plan['output']);receipt=load(out.with_name(out.name+'-adapter-receipt.json'));native=load(out/'wandb_run.json')
    require(load(out/'evaluation_plan.json')==plan and receipt['plan_sha256']==digest(out/'evaluation_plan.json'), 'Evaluation resolved plan differs')
    require(receipt['status']=='completed' and receipt['exit_code']==0 and receipt['run_id']==plan['run_id']
            and receipt['owned_finish_returned'] and receipt['plan_waited_before_environment'],'Adapter did not complete')
    require(native['id']==plan['run_id'] and native['state']=='finished' and native['exit_code']==0 and native['mode']=='online','Native evaluation receipt differs')
    counts=receipt['counts'];steps=counts['env_step_returns']
    require(counts['resource_journal_closed'] and counts['env_step_attempts']==steps==counts['global_control_steps']
            and 0<steps<=30000 and counts['actual_environment_control_frames']==947*steps,'Actual evaluation physical count differs')
    proofs=receipt['artifacts'];require(sorted(p['type'] for p in proofs)==['dapo-evaluation-plan','evaluation'],'Wrong evaluation artifacts')
    native_proof=next(p for p in proofs if p['type']=='evaluation')
    required_native={str(out/name) for name in ('episodes.csv','summary.json','protocol.json','evaluation_resources.jsonl')}
    bound_native={normalized_record(item)['path'] for item in native_proof['files'].values()}
    require(required_native <= bound_native, 'Original evaluation artifact omits necessary raw evidence')
    remote=verify_remote(wandb,study,plan['run_id'],proofs)
    summary=load(out/'summary.json');protocol=load(out/'protocol.json')
    require(summary['episode_count']==947 and summary['normalizer_unchanged'] and summary['gene_assignment_validated'],'Native eval controls failed')
    require(protocol['checkpoint_sha256']==plan['checkpoint_sha256'] and protocol['training_config_sha256']==plan['train_config_sha256'], 'Evaluated endpoint differs')
    return {'status':'completed','run_id':plan['run_id'],'checkpoint_sha256':plan['checkpoint_sha256'],
            'episodes_csv':str(out/'episodes.csv'),'summary_json':str(out/'summary.json'),
            'protocol_json':str(out/'protocol.json'),'training_run_id':plan['source_training_run_id'],
            'eval_seed':plan['eval_seed'],'remote':remote,'actual_environment_control_frames':947*steps}


def monitor_plan(study, entry, target, item):
    sources=[study['monitor_entry'],study['monitor_source'],entry]
    return {'purpose':'dapo_resource_monitor','execution_ready':True,'execution_root':study['execution_root'],
            'output':item['monitor_output'],'monitor_source':study['monitor_source'],'entry':entry,
            'pid':target['pid'],'start_ticks':target['start_ticks'],'run_id':item['monitor_run_id'],
            'entity':study['entity'],'project':study['project'],'group':study['group'],
            'files_sha256':{p:study['files_sha256'][p] for p in sources}}


def wait_preserving(child, record, path, remember, monitor=None, heartbeat=None):
    """Always obtain the actual child's wait, even after repeated control signals."""
    if child is None: return None
    stop_sent=False
    while True:
        try:
            code=child.wait(timeout=60)
            record.update(exit_code=code,ended_at_utc=now(),status='completed' if code==0 else 'failed',
                          actual_exit_code_source='subprocess.wait')
            try: write(path,record)
            except BaseException as exc: remember(exc,'write_actual_wait')
            return record
        except subprocess.TimeoutExpired:
            if heartbeat is not None:
                try:heartbeat(record)
                except BaseException as exc:remember(exc,'heartbeat')
            if monitor is not None and monitor.poll() is not None and child.poll() is None and not stop_sent:
                remember(RuntimeError('Passive monitor exited before target; stopping own target, no retry'),'monitor_lost')
                try:
                    terminate_owned(child,record)
                    stop_sent=True
                except BaseException as exc:remember(exc,'stop_unmonitored_target')
            continue
        except (KeyboardInterrupt,InterruptedError) as exc:
            remember(exc,'interrupted_wait_preserved')
            if isinstance(exc,KeyboardInterrupt) and not stop_sent:
                try:
                    terminate_owned(child,record)
                    stop_sent=True
                except BaseException as secondary:remember(secondary,'stop_interrupted_owned_child')
        except BaseException as exc:
            remember(exc,'wait_error')
            # A real unavailable wait cannot be replaced by a guessed exit code.
            return record



def terminate_owned(child,record):
    if child is None or child.poll() is not None: return
    actual=process_identity(child.pid)
    require(actual is not None and actual['pid']==record['pid'] and actual['start_ticks']==record['start_ticks']
            and actual['argv']==record['command'], 'Refusing signal to an uncertain/reused target PID')
    child.terminate()  # Native SIGTERM/finally can preserve failed raw; never respawn.

def launch_phase(study, key, entry, plan_path, monitor=None, heartbeat=None):
    root=Path(study['execution_root']);directory=Path(study['output'])/'phases'/key
    directory.mkdir(parents=True)
    target=watcher=None;target_record=watcher_record=None;streams=[];first=[];errors=[]
    cmd=command(entry,plan_path)
    def remember(exc,where):
        if not first:first.append((exc,exc.__traceback__))
        errors.append({'where':where,'error':repr(exc)})
        try:write(directory/'control_errors.json',{'primary':repr(first[0][0]),'errors':errors})
        except BaseException as secondary:print('Secondary controller journal error: '+repr(secondary),file=sys.stderr)
    try:
        write(directory/'intent.json',{'utc':now(),'command':cmd,'plan_sha256':digest(plan_path),'no_retry':True})
        stream=(directory/'stdout.log').open('x');streams.append(stream)
        target=subprocess.Popen(cmd,cwd=str(root),env=environment(root),stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        target_record={'pid':target.pid,'command':cmd,'started_at_utc':now(),'status':'running','exit_code':None}
        ident=process_identity(target.pid)
        require(ident is not None and ident['argv']==cmd and ident['uid']==os.getuid(),'Launched target identity uncertain; never respawn')
        target_record.update(start_ticks=ident['start_ticks'],uid=ident['uid'])
        write(directory/'launch_exit.json',target_record)
        if heartbeat is not None:heartbeat(target_record)
        if monitor is not None:
            mp=monitor_plan(study,entry,ident,monitor);mpath=directory/'monitor_plan.json';write(mpath,mp)
            mcmd=command(study['monitor_entry'],mpath)
            stream=(directory/'monitor_stdout.log').open('x');streams.append(stream)
            watcher=subprocess.Popen(mcmd,cwd=str(root),env=environment(root),stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            watcher_record={'pid':watcher.pid,'command':mcmd,'started_at_utc':now(),'status':'running','exit_code':None}
            mi=process_identity(watcher.pid)
            require(mi is not None and mi['argv']==mcmd and mi['uid']==os.getuid(),'Monitor identity uncertain')
            watcher_record.update(start_ticks=mi['start_ticks'],uid=mi['uid']);write(directory/'monitor_exit.json',watcher_record)
    except BaseException as exc:
        remember(exc,'spawn_or_control_setup')
        if target is not None and target_record is not None:
            try:terminate_owned(target,target_record)
            except BaseException as secondary:remember(secondary,'stop_target_after_control_failure')
    finally:
        if target is not None:
            if target_record is None:target_record={'pid':target.pid,'command':cmd,'exit_code':None}
            wait_preserving(target,target_record,directory/'launch_exit.json',remember,watcher,heartbeat)
        if watcher is not None:
            if watcher_record is None:watcher_record={'pid':watcher.pid,'exit_code':None}
            wait_preserving(watcher,watcher_record,directory/'monitor_exit.json',remember)
        for stream in streams:
            try:stream.close()
            except BaseException as exc:remember(exc,'stdout_close')
    if target_record is None or target_record.get('exit_code')!=0:
        remember(RuntimeError('Target did not return actual wait0'),'target_exit')
    if monitor is not None and (watcher_record is None or watcher_record.get('exit_code')!=0):
        remember(RuntimeError('Monitor did not return actual wait0; see target and monitor exit records'),'monitor_exit')
    if first:
        exc,tb=first[0];raise exc.with_traceback(tb)
    return target_record,watcher_record


def verify_monitor(wandb,study,item,wait,target_wait,entry):
    require(wait is not None and wait['exit_code']==0 and wait['actual_exit_code_source']=='subprocess.wait','Actual monitor wait0 required')
    output=Path(item['monitor_output']);m=load(output/'wandb_run.json');summary=load(output/'summary.json')
    require(m['id']==item['monitor_run_id'] and m['state']=='finished' and m['exit_code']==0 and
            summary['status']=='completed' and summary['sample_count']>0,'Monitor completion differs')
    protocol=load(output/'protocol.json');target=protocol['target_identity']
    require(target['pid']==target_wait['pid'] and target['start_ticks']==target_wait['start_ticks']
            and target['uid']==os.getuid() and entry in target['argv']
            and target['argv']==target_wait['command'], 'Monitor observed a different target identity')
    require(protocol['target_entry_sha256']==study['files_sha256'][entry], 'Monitored entry source differs')
    files={p.name:identity(p) for p in output.iterdir() if p.is_file() and p.name!='wandb_run.json'}
    require(set(files)=={'monitor_source.py','protocol.json','driver_resources.csv','summary.json'},'Original monitor four-file contract differs')
    proof={'artifact':m['artifact'],'type':'resource-telemetry','state':m['artifact_state'],'files':files}
    return verify_remote(wandb,study,m['id'],[proof])


def verify_qualifications(wandb,study):
    records=[]
    for item in study['qualification_completed']:
        plan=load(item['plan_path']);wait=load(item['launch_receipt']);review=load(item['verification_receipt'])
        require(review['status']=='completed' and review['actual_wait0'] is True and
                review['plan_sha256']==item['plan_sha256'] and review['run_id']==plan['run_id'],'Prior qualification review differs')
        if item['kind']=='training':result=verify_training(wandb,study,plan,wait)
        else:result=verify_evaluation(wandb,study,plan,wait)
        entry=study['train_entry'] if item['kind']=='training' else study['evaluation_entry']
        result['monitor']=verify_monitor(wandb,study,item,load(item['monitor_wait_receipt']),wait,entry)
        records.append(result)
    fixture=load(study['analysis_fixture']['summary_path'])
    require(fixture['status']=='completed' and fixture['gpu_validated'] is True and fixture['analyzer_sha256']==study['files_sha256'][study['analysis_entry']]
            and fixture['cases_passed'],'Exact analysis CUDA fixture incomplete')
    return records


def evaluation_plan(study,spec,endpoint):
    bindings={p:study['files_sha256'][p] for p in (study['evaluation_entry'],study['evaluator_source'],study['nvml_source'])}
    bindings[endpoint['train_config']]=endpoint['train_config_sha256']
    return {'purpose':'dapo_frozen_master_evaluation','execution_ready':True,'execution_root':study['execution_root'],
            'output':spec['output'],'evaluator_source':study['evaluator_source'],'nvml_source':study['nvml_source'],
            'checkpoint':endpoint['checkpoint'],'checkpoint_sha256':endpoint['checkpoint_sha256'],
            'train_config':endpoint['train_config'],'train_config_sha256':endpoint['train_config_sha256'],
            'run_id':spec['run_id'],'entity':study['entity'],'project':study['project'],'group':study['group'],
            'eval_seed':spec['eval_seed'],'source_training_run_id':endpoint['run_id'],'files_sha256':bindings}


def analysis_plan(study,stage,evaluations):
    keys=study['pilot_jobs'] if stage=='pilot' else study['formal_jobs'];by={}
    for key in keys:
        train=load(study['train_jobs'][key]['plan_path'])
        for endpoint in evaluations[key]:
            pair=by.setdefault((train['seed'],endpoint['eval_seed']),{})
            pair[train['arm']]={k:endpoint[k] for k in ('episodes_csv','summary_json','protocol_json','checkpoint_sha256','run_id','training_run_id')}
    comparisons=[dict(training_seed=a,eval_seed=b,**pair) for (a,b),pair in sorted(by.items())]
    paths={study['analysis_entry'],study['base_source']}
    for item in comparisons:
        for mode in ('control','candidate'):
            paths.update(item[mode][k] for k in ('episodes_csv','summary_json','protocol_json'))
            paths.add(str(Path(item[mode]['episodes_csv']).parent/'wandb_run.json'))
    spec=study['analysis_specs'][stage]
    return {'stage':stage,'execution_root':study['execution_root'],'output':spec['output'],'run_id':spec['run_id'],
            'entity':study['entity'],'project':study['project'],'group':study['group'],
            'files_sha256':{p:digest(p) for p in sorted(paths)},'comparisons':comparisons}


def verify_analysis(wandb,study,plan,wait):
    require(wait['exit_code']==0 and wait['actual_exit_code_source']=='subprocess.wait','Analysis actual wait0 required')
    out=Path(plan['output']);receipt=load(out/'publication.json');summary=load(out/'summary.json')
    expected_plan=Path(study['output'])/'resolved_plans'/('analysis-'+plan['stage']+'.json')
    require(load(expected_plan)==plan and receipt['plan_sha256']==digest(expected_plan) and
            receipt['entry_sha256']==study['files_sha256'][study['analysis_entry']], 'Analysis plan/source binding differs')
    require(receipt['status']=='completed' and receipt['exit_code']==0 and receipt['state']=='finished'
            and receipt['run_id']==plan['run_id'],'Analysis publication incomplete')
    require(summary['status']=='completed' and summary['gpu_validated'] is True and summary['stage']==plan['stage'],'CUDA analysis incomplete')
    proofs=[receipt['plan_publication'],receipt['raw_publication']]
    require([p['type'] for p in proofs]==['dapo-pair-analysis-plan','dapo-pair-analysis-results'],'Wrong analysis artifact types')
    bound={normalized_record(item)['path'] for item in proofs[1]['files'].values()}
    require(str(out/'summary.json') in bound, 'Analysis summary is not in raw artifact')
    remote=verify_remote(wandb,study,plan['run_id'],proofs)
    gate=summary['pilot_gate_passed']
    require((type(gate)is bool) if plan['stage']=='pilot' else gate is None,'Wrong registered gate type')
    return {'status':'completed','run_id':plan['run_id'],'pilot_gate_passed':gate,'summary_sha256':digest(out/'summary.json'),'remote':remote}




def payload_budget(files, limit):
    total=sum(p.stat().st_size for p in files.values())
    require(total <= limit, 'Prospective payload exceeds fixed byte budget')
    return total

def closed_controller_files(output, journal_closed=True):
    files={}
    for p in output.rglob('*'):
        if not p.is_file() or p.is_symlink() or 'wandb' in p.relative_to(output).parts: continue
        if p.name=='phases.jsonl' and not journal_closed: continue
        if p.name in ('stdout.log','monitor_stdout.log'):
            wait_path=p.with_name('launch_exit.json' if p.name=='stdout.log' else 'monitor_exit.json')
            if not wait_path.is_file(): continue
            wait=load(wait_path)
            if wait.get('actual_exit_code_source')!='subprocess.wait' or wait.get('exit_code') is None: continue
        files[str(p.relative_to(output))]=p
    return files

def execute(study,path):
    root=Path(study['execution_root']);output=Path(study['output']);output.mkdir(parents=True)
    shutil.copy2(path,output/'study_plan.json')
    sources=output/'source';sources.mkdir()
    for name in ('train_entry','evaluation_entry','evaluator_source','analysis_entry','monitor_entry','monitor_source','nvml_source','base_source'):
        shutil.copy2(study[name],sources/(name+'.py'))
    shutil.copy2(__file__,sources/'controller.py')
    base=module('dapo_controller_publication',study['base_source'])
    import wandb
    require(wandb.__version__=='0.24.2' and wandb.run is None,'Fresh controller SDK context required')
    run=None;first=[];completed=[];endpoints={};evaluations={};controller_status='failed'
    journal=(output/'phases.jsonl').open('x')
    def remember(exc,where):
        if not first:first.append((exc,exc.__traceback__))
        try:
            with output.with_name(output.name+'-errors.jsonl').open('a') as stream:
                stream.write(json.dumps({'utc':now(),'where':where,'error':repr(exc),'primary':repr(first[0][0])})+'\n')
        except BaseException as secondary:print('Secondary controller error record: '+repr(secondary),file=sys.stderr)
    def mark(kind,key,result):
        completed.append({'kind':kind,'key':key,'result':result})
        if run is not None:
            run.log({'study/phase':key,'study/phase_status':'completed','study/completed_phases':len(completed)})
        journal.write(json.dumps(completed[-1],sort_keys=True)+'\n');journal.flush()
        write(output/'state.json',{'status':'running','utc':now(),'completed_phase_count':len(completed),'last_phase':key})
    def inputs_unchanged():
        for p,s in study['files_sha256'].items():require(digest(p)==s,'Frozen study input changed: '+p)
        require(digest(path)==digest(output/'study_plan.json'),'Study plan changed')
    def heartbeat(key,record):
        progress={'status':'running','active_phase':key,'utc':now(),'pid':record['pid'],
                  'start_ticks':record.get('start_ticks'),'completed_phase_count':len(completed)}
        write(output/'state.json',progress)
        run.summary.update({'study/active_phase':key,'study/heartbeat_utc':progress['utc'],'study/active_pid':record['pid']})
        run.log({'study/phase':key,'study/phase_status':'running','study/heartbeat_unix':datetime.now(timezone.utc).timestamp()})
    def train_one(key):
        inputs_unchanged();job=study['train_jobs'][key];plan=load(job['plan_path'])
        wait,mwait=launch_phase(study,'train-'+key,study['train_entry'],Path(job['plan_path']),job,lambda rec:heartbeat('train-'+key,rec))
        endpoint=verify_training(wandb,study,plan,wait);monitor=verify_monitor(wandb,study,job,mwait,wait,study['train_entry'])
        mark('training',key,dict(endpoint,monitor=monitor));endpoints[key]=endpoint
    def evaluate_one(key,spec):
        inputs_unchanged();plan=evaluation_plan(study,spec,endpoints[key])
        p=output/'resolved_plans'/('eval-'+key+'-'+str(spec['eval_seed'])+'.json');p.parent.mkdir(exist_ok=True);write(p,plan)
        wait,mwait=launch_phase(study,'eval-'+key+'-'+str(spec['eval_seed']),study['evaluation_entry'],p,spec,lambda rec:heartbeat('eval-'+key+'-'+str(spec['eval_seed']),rec))
        result=verify_evaluation(wandb,study,plan,wait);monitor=verify_monitor(wandb,study,spec,mwait,wait,study['evaluation_entry'])
        result['monitor']=monitor;evaluations.setdefault(key,[]).append(result);mark('evaluation',key+'-'+str(spec['eval_seed']),result)
    def analyze(stage):
        inputs_unchanged();plan=analysis_plan(study,stage,evaluations)
        p=output/'resolved_plans'/('analysis-'+stage+'.json');write(p,plan)
        wait,_=launch_phase(study,'analysis-'+stage,study['analysis_entry'],p,heartbeat=lambda rec:heartbeat('analysis-'+stage,rec))
        result=verify_analysis(wandb,study,plan,wait);mark('analysis',stage,result);return result
    try:
        run=wandb.init(id=study['run_id'],entity=study['entity'],project=study['project'],group=study['group'],
                       job_type='dapo-study-controller',mode='online',resume='never',force=True,dir=str(output),config=study)
        require(run is not None and run.id==study['run_id'] and not run.offline and not run.resumed,'Wrong own online controller run')
        prefiles={'study_plan.json':output/'study_plan.json'}
        prefiles.update({'source/'+p.name:p for p in sources.iterdir()})
        # Exact static train plans and completed qualifications are small references; no checkpoint copies.
        for i,item in enumerate(study['train_jobs'].values()):prefiles['train_plans/'+str(i)+'.json']=Path(item['plan_path'])
        for i,item in enumerate(study['qualification_completed']):
            for key in ('plan_path','launch_receipt','verification_receipt','monitor_wait_receipt'):prefiles['qualifications/'+str(i)+'-'+key+'.json']=Path(item[key])
        text_suffixes={'.py','.json','.yaml','.yml','.md','.diff','.patch','.sh'}
        for i,(source,sha) in enumerate(sorted(study['files_sha256'].items())):
            if Path(source).suffix in text_suffixes:
                prefiles['bound_inputs/'+str(i)+'_'+Path(source).name]=Path(source)
        payload_budget(prefiles,64*2**20)
        prefiles['analysis_fixture_summary.json']=Path(study['analysis_fixture']['summary_path'])
        for name,source in study.get('extra_plan_files',{}).items():
            require(name not in prefiles, 'Prospective attachment collision')
            prefiles[name]=Path(source)
        payload_budget(prefiles,512*2**20)
        proof=base.publish(run,prefiles,'dapo-study-plan');write(output/'plan-artifact-wait.json',proof)
        mark('qualification','completed-before-queue',verify_qualifications(wandb,study))
        for key in study['pilot_jobs']:train_one(key)
        # Fixed seed-major order, then the predeclared candidate/control pilot order.
        for index in range(2):
            for key in study['pilot_jobs']:evaluate_one(key,study['evaluation_specs'][key][index])
        pilot=analyze('pilot')
        if pilot['pilot_gate_passed']:
            for key in study['formal_jobs']:
                train_one(key)
                for item in study['evaluation_specs'][key]:evaluate_one(key,item)
            analyze('formal');controller_status='completed_formal'
        else:
            controller_status='completed_pilot_negative'
        inputs_unchanged()
    except BaseException as exc:
        if run is None:
            candidate=wandb.run
            if candidate is not None and candidate.id==study['run_id']:
                run=candidate
        remember(exc,'study_phase')
    finally:
        try:journal.close()
        except BaseException as exc:remember(exc,'phase_journal_close')
        try:
            write(output/'summary.json',{'status':'failed' if first else controller_status,'utc':now(),
                  'completed_phase_count':len(completed),'first_error':repr(first[0][0]) if first else None,
                  'automatic_retry':False,'formal_started':any(p.is_file() and load(p).get('pid',0)>0 for p in
                     (output/'phases'/('train-'+key)/'launch_exit.json' for key in study['formal_jobs'])),
                  'scope':'Only completed fixed phases; negative pilot stops formal allocation. Actual child/monitor wait and fresh manifests independently checked.'})
            if run is not None and run.id==study['run_id']:
                files=closed_controller_files(output,journal.closed)
                result=base.publish(run,files,'dapo-study-result');write(output/'result-artifact-wait.json',result)
        except BaseException as exc:remember(exc,'closed_controller_raw_publication')
        if run is not None and run.id==study['run_id']:
            try:run.finish(exit_code=1 if first else 0)
            except BaseException as exc:remember(exc,'own_controller_finish')
        if not first and run is not None:
            try:
                remote=verify_remote(wandb,study,study['run_id'],[load(output/'plan-artifact-wait.json'),load(output/'result-artifact-wait.json')])
                write(output.with_name(output.name+'-publication.json'),{'status':'completed','state':'finished','run_id':study['run_id'],
                      'plan_sha256':digest(path),'remote':remote,'scope':controller_status})
            except BaseException as exc:remember(exc,'fresh_controller_readback')
        if first:
            try:write(output.with_name(output.name+'-failed.json'),{'status':'failed','run_id':study['run_id'],'first_error':repr(first[0][0]),'closed_raw_retained':True})
            except BaseException as secondary:print('Failed receipt write: '+repr(secondary),file=sys.stderr)
    if first:
        exc,tb=first[0];raise exc.with_traceback(tb)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--execute',action='store_true')
    args=ap.parse_args();require(digest(args.plan)==args.plan_sha256,'Study plan SHA differs')
    study=load(args.plan);report=validate(study,args.plan)
    if not args.execute:print(json.dumps(report,indent=2));return
    require(study['execution_ready'] is True,'Unbound study cannot execute')
    require(Path.cwd()==Path(study['execution_root']),'Controller cwd differs')
    require(shutil.disk_usage(study['execution_root']).free >= 4*2**30, 'Controller startup disk below4GiB')
    previous=signal.getsignal(signal.SIGTERM)
    def terminate(_sig,_frame):raise KeyboardInterrupt('Study SIGTERM; preserve active child wait then stop')
    with open('/tmp/epo-dapo-study-controller.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,terminate)
        try:execute(study,args.plan)
        finally:signal.signal(signal.SIGTERM,previous)


if __name__=='__main__':
    main()
