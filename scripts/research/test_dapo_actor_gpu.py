"""Default inert. --execute imports IsaacGym BEFORE Torch, then tiny CUDA leaves.

No model or optimizer. Tests native/asymmetric actor, logratio-leaf coefficients
and frozen off-support effects. Run only under root's reviewed GPU ownership.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

LAST_EVIDENCE = None


def run(t):
    global LAST_EVIDENCE
    source = Path(__file__).with_name('dapo_actor.py')
    spec = importlib.util.spec_from_file_location('dapo_actor_candidate', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'rl_games'))
    from rl_games.common import common_losses
    result = {'status': 'running', 'gpu_validated': False,
              'scope': 'Synthetic actor/logratio leaf only; no model or learning proof',
              'dapo_actor_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'torch_version': t.__version__, 'cuda_runtime': t.version.cuda,
              'gpu': t.cuda.get_device_name(0), 'model_forwards': 0,
              'optimizer_steps': 0, 'autograd_calls': 0, 'rows': []}
    LAST_EVIDENCE = result
    assert t.__version__ == '1.8.1', 'Version-specific max/clamp derivative contract'
    assert t.cuda.is_available()
    def check(ok, message):
        if not ok:
            result.update(status='failed', failure=message)
            raise RuntimeError(message)
    def grad(loss, leaf):
        out, = t.autograd.grad(loss.sum(), leaf)
        result['autograd_calls'] += 1
        return out
    with t.enable_grad():
        x=t.tensor([1.],device='cuda:0',requires_grad=True)
        y=t.tensor([1.],device='cuda:0',requires_grad=True)
        gx,gy=t.autograd.grad(t.max(x,y).sum(),(x,y));result['autograd_calls']+=1
        row={'case':'actual_max_tie','left':float(gx.item()),'right':float(gy.item())};result['rows'].append(row)
        check(row['left']==0 and row['right']==1,'Actual max tie differs from second-input convention')
        layouts=((t.float16,t.float16),(t.float32,t.float32),(t.float64,t.float64),
                 (t.float16,t.float32),(t.float32,t.float16))
        for dtype, adtype in layouts:
            # Exact binary boundaries qualify inclusive clamp/max dispatch.
            # Production decimal bounds qualify actual representable neighbors.
            for low,high in ((.125,.25),(.1,.2)):
                with t.no_grad():
                    edges=t.tensor([1-low,1+low,1+high],device='cuda:0',dtype=dtype)
                    eps=t.finfo(dtype).eps
                    points=t.cat((edges-eps,edges,edges+eps,
                                  t.tensor([.25,.5,1.,1.5,2.],device='cuda:0',dtype=dtype)))
                    ratio_values=points.repeat(3)
                    A=t.cat((t.ones_like(points,dtype=adtype),-t.ones_like(points,dtype=adtype),t.zeros_like(points,dtype=adtype)))
                ratio=ratio_values.detach().requires_grad_()
                raw=-A*ratio
                clipped=-A*t.clamp(ratio,1-low,1+high)
                original=t.max(raw,clipped)
                actual_dr=grad(original,ratio)
                with t.no_grad():
                    expected=module._branch_arrays(t,ratio,A,low,high)
                    exact=t.equal(original,expected['loss']) and t.equal(actual_dr*ratio,expected['dlogratio'])
                    result['rows'].append({'case':'ratio_leaf_boundaries','ratio_dtype':str(dtype),'A_dtype':str(adtype),
                        'low':low,'high':high,'n':ratio.numel(),'exact':bool(exact),'ties':int(expected['tie'].sum().item())})
                    check(exact,'Ratio-leaf boundary analytic coefficient differs')
                # Compare unchanged original actor vs symmetric adapter on the
                # original neglogp leaf, including mixed A/product dtypes.
                with t.no_grad():
                    ell=t.log(ratio_values)
                    old=t.zeros_like(ell)
                n1=(-ell).detach().requires_grad_();n2=(-ell).detach().requires_grad_()
                orig=common_losses.actor_loss(old,n1,A,True,low)
                sym=module.actor_loss_asymmetric(t,old,n2,A,True,low,low)
                g1=grad(orig,n1);g2=grad(sym,n2)
                same=t.equal(orig,sym) and t.equal(g1,g2)
                result['rows'].append({'case':'symmetric_native_exact','ratio_dtype':str(dtype),'A_dtype':str(adtype),
                                      'clip':low,'loss_exact':bool(t.equal(orig,sym)),'gradient_exact':bool(t.equal(g1,g2))})
                check(same,'Symmetric adapter differs from original actor')
                l0=ell.detach().requires_grad_();l1=ell.detach().requires_grad_()
                native=module.actor_loss_asymmetric(t,old,-l0,A,True,low,low)
                candidate=module.actor_loss_asymmetric(t,old,-l1,A,True,low,high)
                gn=grad(native,l0);gc=grad(candidate,l1)
                with t.no_grad():
                    shadow=module.actor_shadow(t,old,-ell,A,native_clip=low,clip_low=low,clip_high=high)
                    exact=(t.equal(native,shadow['native']['loss']) and t.equal(candidate,shadow['candidate']['loss'])
                           and t.equal(gn,shadow['native']['dlogratio']) and t.equal(gc,shadow['candidate']['dlogratio']))
                    result['rows'].append({'case':'logratio_leaf_shadow_exact','ratio_dtype':str(dtype),'A_dtype':str(adtype),
                                          'low':low,'high':high,'exact':bool(exact)})
                    check(exact,'Detached logratio shadow differs from leaf autograd')
        # Away from rounding boundaries, only positive-A newly unclipped rows
        # change; old safe off-support makes their effective coefficients zero.
        ell=t.log(t.tensor([.8,1.,1.05,1.15,1.19,1.3]*3,device='cuda:0',dtype=t.float32))
        old=t.zeros_like(ell)
        A=t.tensor([1.]*6+[-1.]*6+[0.]*6,device='cuda:0')
        with t.no_grad():
            s=module.actor_shadow(t,old,-ell,A)
            expected=(A>0)&(s['ratio']>1.1)&(s['ratio']<=1.2)
            check(t.equal(s['coefficient_changed'],expected),'Interior changes outside expected positive-A band')
            safe=t.exp(t.nan_to_num(ell,nan=0.,posinf=20.,neginf=-20.).clamp(-20.,20.))
            w=((safe>=.9)&(safe<=1.1)).float()
            equal=t.equal(s['native']['dlogratio']*w,s['candidate']['dlogratio']*w)
            result['rows'].append({'case':'positive_band_and_fixed_off_support','changed':int(expected.sum().item()),'off_weighted_coefficients_exact':bool(equal)})
            check(equal,'Frozen off support did not zero newly recovered band')
            # Deliberate invalid/extreme rows remain present and marked; no
            # claim to a meaningful piecewise derivative for NaN/Inf products.
            invalid=t.tensor([float('nan'),float('inf'),-float('inf'),100.,-100.],device='cuda:0')
            bad=module.actor_shadow(t,t.zeros_like(invalid),-invalid,t.tensor([1.,1.,0.,0.,1.],device='cuda:0'))
            result['rows'].append({'case':'nonfinite_retained','n':invalid.numel(),
                                   'finite_comparable':int(bad['finite'].sum().item()),
                                   'raw_ratio_nonfinite':int((~t.isfinite(bad['ratio'])).sum().item())})
            check(bad['finite'].numel()==5 and not bool(bad['finite'][:4].any().item()),'Invalid rows silently treated as qualified finite')
    result.update(status='completed',gpu_validated=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--output')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'status':'dry_run','Torch_imported':False,'GPU_used':False}));return
    if not args.output:raise SystemExit('--execute requires new --output JSON path')
    output=Path(args.output)
    if output.exists():raise SystemExit('Refuse overwrite')
    output.parent.mkdir(parents=True,exist_ok=True)
    try:
        import isaacgym  # Must initialize Isaac bindings before importing Torch.
        import torch
        result=run(torch)
    except BaseException as exc:
        result=LAST_EVIDENCE or {'status':'failed','gpu_validated':False,'rows':[]}
        result.update(status='failed',gpu_validated=False,error=repr(exc))
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        raise
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'autograd_calls':result['autograd_calls']}))


if __name__=='__main__':main()
