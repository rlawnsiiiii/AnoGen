"""E2b: leave the last part of the trajectory unguided (guidance_t_window) — level-shift slice, exact denoisers."""
import json
from pathlib import Path
import numpy as np
from anogen.testbed.gauss import (BlockEncoder, GPChannel, GuideTerm, LinearDenoiser, Schedule, Shell, guided_ddim,
                                  level_shift, proto_energy)
from anogen.testbed.metrics import gallery_report
W,N=512,256
ch,sch=GPChannel(),Schedule(); rng=np.random.default_rng(0)
train=ch.sample(4096,W,rng); lo,hi=float(train.min()),float(train.max())
enc=BlockEncoder(W,8); shell=Shell.from_nominal(enc,ch.sample(512,W,rng),ch.sample(512,W,rng))
donor=ch.sample(N,W,rng)
protos=enc.encode(level_shift(ch.sample(6,W,rng),np.array([90,150,210,270,330,400]),np.full(6,0.25)))
pidx=np.random.default_rng(17).integers(0,6,N)
terms=[GuideTerm(0.3,lambda x: shell.band(x)),GuideTerm(1.0,lambda x: proto_energy(enc,x,protos[pidx]))]
res={}
for kind in ("flip_ramp","bidir"):
  den=LinearDenoiser(ch,W,sch,kind)
  for space in ("x0",):
    for final,tw in ((1.0,(0,1)),(0.25,(0,1)),(1.0,(0.1,1)),(1.0,(0.2,1)),(0.0,(0.1,1))):
      x=guided_ddim(den,donor,terms,rng=np.random.default_rng(1),space=space,final_scale=final,t_window=tw)
      r=gallery_report(x,lo,hi); z=enc.encode(x); pd=np.linalg.norm(z-protos[pidx],axis=1).mean()
      print(kind,space,"final",final,"window",tw,"dp999 %.3f start %.2f end %.2f protodist %.2f"%(r['diff_p999'],r['peak_at_start'],r['peak_at_end'],pd),flush=True)
      res[f"{kind} {space} final {final} window {tw}"]={**r,"proto_dist":float(pd)}
Path("results/testbed/e2b_twindow.json").write_text(json.dumps(res,indent=1))
