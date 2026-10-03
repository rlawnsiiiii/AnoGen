"""E8a: train the tiny numpy ε-net (causal or bidirectional) on the testbed GP."""
import pickle, sys, time
from pathlib import Path
import numpy as np
from anogen.testbed.gauss import GPChannel, Schedule
from anogen.testbed.npnet import TinyEpsNet, train_eps_net

causal = sys.argv[1] == "causal"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
out = Path("results/testbed/nets"); out.mkdir(parents=True, exist_ok=True)
ch, sch = GPChannel(), Schedule()
chol = np.linalg.cholesky(ch.cov(512))
fn = lambda n, rng: ch.mean + (chol @ rng.standard_normal((512, n))).T  # noqa: E731
net = TinyEpsNet(hidden=32, k=5, dilations=(1, 2, 4, 8, 16), causal=causal, seed=0)
t0 = time.time()
hist = train_eps_net(net, fn, sch, steps=steps, batch=64, mean=ch.mean, seed=1, log_every=250)
for layer in net.layers():
    layer.cache = None
net._pre = net._h_in = None
net._tf = None
pickle.dump({"net": net, "hist": hist, "seconds": time.time() - t0}, open(out / f"{sys.argv[1]}.pkl", "wb"))
print("done", sys.argv[1], time.time() - t0)
