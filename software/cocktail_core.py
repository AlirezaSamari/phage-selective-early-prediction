
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def schedule_mask(minutes,hour):
    t=torch.as_tensor(minutes)
    mask=(t<=hour*60)[:,None].expand(-1,3).clone()
    phage=((t==0)|(t==120)|(t==360))&(t<=hour*60)
    mask[:,1:]=phage[:,None]
    return mask

def outcomes(full,minutes):
    b=full[:,:,0]; late=torch.as_tensor(minutes)>=21*60
    sustained=b[:,late].amax(1)<=0.
    rebound=(b.amin(1)<=0.)&(b[:,-1]>=2.)&~sustained
    partial=(b[:,0]-b[:,-1]>=2.)&~sustained&~rebound
    y=torch.full((len(b),),3,dtype=torch.long)
    y[sustained]=0; y[rebound]=1; y[partial]=2
    return y

def observe(clean,cfg,seed,scenario=None,augmented=False):
    scenario=scenario or {}; g=torch.Generator().manual_seed(int(seed)); n=len(clean)
    if augmented:
        # Most cases remain nominal; a fixed subset receives bounded assay augmentation.
        use=(torch.rand(n,1,1,generator=g)<float(cfg['training_augmentation_fraction'])).float()
        noise=1.+use*.5*torch.rand(n,1,1,generator=g)
        missing=.05+use*.10*torch.rand(n,1,1,generator=g)
        shift=use*.5*torch.rand(n,1,1,generator=g)
    else:
        noise=float(scenario.get('noise',1.))
        missing=float(scenario.get('missing',.05))
        shift=float(scenario.get('lod_shift',0.))
    lod=(torch.tensor(cfg['log10_lod']).view(1,1,3)+shift).expand(n,clean.shape[1],3).clone()
    sigma=torch.tensor(cfg['sigma_log10']).view(1,1,3)
    latent=clean+torch.randn(clean.shape,generator=g)*sigma*noise
    observed=torch.rand(clean.shape,generator=g)>=missing
    censored=observed&(latent<=lod)
    values=torch.where(observed,torch.maximum(latent,lod),torch.zeros_like(latent))
    return dict(values=values,observed=observed,censored=censored,lod=lod)

def features(obs,cfg,hour,mean,std):
    available=schedule_mask(cfg['master_minutes'],hour)[None]
    seen=obs['observed']&available; cens=obs['censored']&seen
    z=(obs['values']-mean[None,None])/std[None,None]
    z=torch.where(seen,z,torch.zeros_like(z))
    lod=(obs['lod']-mean[None,None])/std[None,None]
    lod=torch.where(available,lod,torch.zeros_like(lod))
    t=(torch.tensor(cfg['master_minutes']).float()/360.)[None,:,None].expand(len(z),-1,-1)
    base=torch.cat((z,seen.float(),cens.float(),lod,t),-1)
    previous=torch.cat((torch.zeros_like(z[:,:1]),z[:,:-1]),1)
    previous_seen=torch.cat((torch.zeros_like(seen[:,:1]),seen[:,:-1]),1)
    pair=seen&previous_seen
    delta=torch.where(pair,z-previous,torch.zeros_like(z))
    contrasts=torch.stack((z[:,:,0]-z[:,:,1],z[:,:,0]-z[:,:,2],z[:,:,1]-z[:,:,2]),-1)
    contrasts=torch.where(seen.all(-1,keepdim=True),contrasts,torch.zeros_like(contrasts))
    return torch.cat((base,delta,contrasts),-1).float()

class Model(nn.Module):
    def __init__(self,width=72,components=5,input_dim=13):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(input_dim,width),nn.GELU(),nn.LayerNorm(width))
        self.cls=nn.Parameter(torch.zeros(1,1,width))
        layer=nn.TransformerEncoderLayer(d_model=width,nhead=4,dim_feedforward=width*4,
            dropout=.08,activation='gelu',batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=2)
        self.classifier=nn.Linear(width,4); self.mixture=nn.Linear(width,components*3)
        self.action=nn.Linear(width,1)
    def forward(self,x):
        z=self.project(x); z=torch.cat((self.cls.expand(len(x),-1,-1),z),1)
        h=self.encoder(z)[:,0]; logits=self.classifier(h)
        w,m,raw=self.mixture(h).chunk(3,-1)
        return logits,w,m,F.softplus(raw)+.03,self.action(h).squeeze(-1),h

def mixture_nll(w,m,s,target):
    comp=-.5*((target[:,None]-m)/s)**2-s.log()-.5*math.log(2*math.pi)
    return -torch.logsumexp(F.log_softmax(w,-1)+comp,-1).mean()

def cascade(action_prob,thresholds,cfg,active_actions,domain=None):
    pred=torch.full((len(action_prob[0]),),-1,dtype=torch.long); stage=pred.clone()
    for k,allowed in enumerate(cfg['allowed_classes']):
        enabled=bool(active_actions.get(str(allowed),False))
        take=(pred<0)&enabled&(action_prob[k]>=float(thresholds[f'stage_{k}']))
        if domain is not None: take &= domain[k]
        pred[take]=allowed; stage[take]=k
    return pred,stage

PHYS_DEFAULT = dict(ks=5., c_in=100., c0=None, eps=2e-6,
                    resistant_init=(1e-7, 1e-7, 1e-14), growth_ratio=(1., .9, .9, .8),
                    mutation=True, superinfection=True)

@torch.no_grad()
def simulate(theta,minutes,device='cpu',dt_min=1.,dtype=torch.float32,phys=None,return_states=False):
    p={**PHYS_DEFAULT,**(phys or {})}
    theta=theta.to(device=device,dtype=dtype)
    assert theta.ndim==2 and theta.shape[1]==15
    assert dt_min in (1.,.5,.25)
    times=np.asarray(minutes,dtype=float)
    assert times[0]==0 and np.all(np.diff(times)>0)
    sample_steps=np.rint(times/dt_min).astype(int)
    assert np.allclose(sample_steps*dt_min,times)
    n=len(theta); qlen=int(60/dt_min)
    la=(theta[:,4].round().clamp(10,60)/dt_min).long()
    lb=(theta[:,5].round().clamp(10,60)/dt_min).long()
    ba,bb=theta[:,6],theta[:,7]
    mua,mub=10.**theta[:,8],10.**theta[:,9]
    if not p['mutation']:
        mua,mub=torch.zeros_like(mua),torch.zeros_like(mub)
    flow,da,db=theta[:,10],theta[:,11],theta[:,12]
    aa,ab=10.**theta[:,2],10.**theta[:,3]
    pa,pb=10.**theta[:,13],10.**theta[:,14]
    sup=1. if p['superinfection'] else 0.
    ks,c_in,eps=float(p['ks']),float(p['c_in']),float(p['eps'])
    growth=theta[:,1,None]*theta.new_tensor(list(p['growth_ratio']))
    ra,rb,rab=p['resistant_init']
    active=10.**theta[:,0,None]*theta.new_tensor([1.-ra-rb-rab,ra,rb,rab])
    nutrient=theta.new_full((n,),c_in if p['c0'] is None else float(p['c0']))
    q={key:theta.new_zeros((n,qlen)) for key in ('a','b','ar','br','aw','bw','tie')}
    dt_h=dt_min/60.; head=0; offsets=torch.arange(qlen,device=device); result=[]; states=[]
    def add(key,value,delay):
        target=(head+delay-1)%qlen
        q[key].scatter_add_(1,target[:,None],value[:,None])
    def record():
        infected=sum(v.sum(1) for v in q.values())
        total=active.sum(1)+infected
        result.append(torch.stack((total,pa,pb),-1).clamp_min(1e-30).log10())
        if return_states:
            states.append(torch.cat((active,infected[:,None]),1).clamp_min(1e-30).log10())
    record(); cursor=1
    for step in range(int(sample_steps[-1])):
        nutrient=nutrient+dt_h*flow*(c_in-nutrient)
        monod=nutrient/(nutrient+ks).clamp_min(1e-30)
        new=active*growth*monod[:,None]*dt_h
        active=active+new-active*flow[:,None]*dt_h
        nutrient=(nutrient-eps*new.sum(1)).clamp_min(0.)
        ma,mb=new[:,0]*mua,new[:,0]*mub; mab=new[:,0]*mua*mub
        scale=torch.minimum(torch.ones_like(ma),new[:,0]/(ma+mb+mab).clamp_min(1e-30))
        ma,mb,mab=ma*scale,mb*scale,mab*scale
        sa,sb=new[:,1]*mub,new[:,2]*mua
        active=torch.stack((active[:,0]-ma-mb-mab,active[:,1]+ma-sa,
                            active[:,2]+mb-sb,active[:,3]+mab+sa+sb),1).clamp_min(0.)
        survival=(1.-flow*dt_h).clamp_min(0.)
        for v in q.values(): v.mul_(survival[:,None])
        due={key:v[:,head].clone() for key,v in q.items()}
        for v in q.values(): v[:,head]=0.
        pa=pa+ba*(due['a']+due['ar']+due['aw']+.5*due['tie'])
        pb=pb+bb*(due['b']+due['br']+due['bw']+.5*due['tie'])
        pa=pa*(1.-(da+flow)*dt_h).clamp_min(0.)
        pb=pb*(1.-(db+flow)*dt_h).clamp_min(0.)
        head=(head+1)%qlen
        ia,ib=q['a'].sum(1),q['b'].sum(1)
        infect_a=active[:,0]+active[:,2]+sup*ib; infect_b=active[:,0]+active[:,1]+sup*ia
        raw_a,raw_b=aa*infect_a*pa*dt_min,ab*infect_b*pb*dt_min
        bound_a,bound_b=torch.minimum(raw_a,pa),torch.minimum(raw_b,pb)
        hit_a=torch.minimum(bound_a,infect_a).clamp_min(0.)
        hit_b=torch.minimum(bound_b,infect_b).clamp_min(0.)
        pa,pb=(pa-bound_a).clamp_min(0.),(pb-bound_b).clamp_min(0.)
        a_s=hit_a*active[:,0]/infect_a.clamp_min(1e-30)
        a_r=hit_a*active[:,2]/infect_a.clamp_min(1e-30)
        a_b=hit_a*sup*ib/infect_a.clamp_min(1e-30)
        b_s=hit_b*active[:,0]/infect_b.clamp_min(1e-30)
        b_r=hit_b*active[:,1]/infect_b.clamp_min(1e-30)
        b_a=hit_b*sup*ia/infect_b.clamp_min(1e-30)
        both=a_s*b_s/active[:,0].clamp_min(1e-30)
        only_a,only_b=(a_s-both).clamp_min(0.),(b_s-both).clamp_min(0.)
        factor=torch.minimum(torch.ones_like(both),active[:,0]/(only_a+only_b+both).clamp_min(1e-30))
        only_a,only_b,both=only_a*factor,only_b*factor,both*factor
        a_r,b_r=torch.minimum(a_r,active[:,2]),torch.minimum(b_r,active[:,1])
        # Superinfection transfers only pre-existing cohorts.
        slots=((head+offsets)%qlen)[None].expand(n,-1); remaining=offsets[None]+1
        for old,amount,total,latency,new_winner,old_winner in (
            ('b',a_b,ib,la,'aw','bw'),('a',b_a,ia,lb,'bw','aw')):
            fraction=(amount/total.clamp_min(1e-30)).clamp(0.,1.)
            moved=q[old].gather(1,slots)*fraction[:,None]
            q[old].mul_((1.-fraction)[:,None])
            earlier=latency[:,None]<remaining
            equal=latency[:,None]==remaining
            later=latency[:,None]>remaining
            add(new_winner,(moved*earlier).sum(1),latency)
            q[old_winner].scatter_add_(1,slots,moved*later)
            q['tie'].scatter_add_(1,slots,moved*equal)
        active=torch.stack((active[:,0]-only_a-only_b-both,active[:,1]-b_r,
                            active[:,2]-a_r,active[:,3]),1).clamp_min(0.)
        add('a',only_a,la); add('b',only_b,lb); add('ar',a_r,la); add('br',b_r,lb)
        add('aw',both*(la<lb),la); add('bw',both*(lb<la),lb)
        add('tie',both*(la==lb),torch.minimum(la,lb))
        if cursor<len(sample_steps) and step+1==sample_steps[cursor]:
            record(); cursor+=1
    assert len(result)==len(minutes)
    out=torch.stack(result,1)
    if not torch.isfinite(out).all(): raise FloatingPointError('Nonfinite simulator output')
    return (out,torch.stack(states,1)) if return_states else out
