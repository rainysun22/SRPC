"""Debug4: 去重绑定 + 更高维 对解对率与 acc(8) 的影响。"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2_reasoning import make_vocab, bind, unbind, sample_grid, colored_output, _expose, ALL_PERMS

class Mem:
    def __init__(self,V): self.V=V; self.d=V.shape[1]; self.S=np.zeros(self.d); self.cov=np.zeros(5,bool)
    def store(self,c,cp):
        if self.cov[c-1]: return      # 去重
        self.S+=bind(self.V[c-1],self.V[cp-1]); self.cov[c-1]=True
    def map(self,c):
        if not self.cov[c-1]: return 0
        q=unbind(self.S,self.V[c-1]); q=q/np.linalg.norm(q)
        return int((self.V@q).argmax())+1

def run(d,trials,k):
    V=make_vocab(d,seed=0); hit=0
    for t in range(trials):
        r=ALL_PERMS[np.random.default_rng(41+t).integers(len(ALL_PERMS))]
        rng=np.random.default_rng(42+t); mem=Mem(V)
        for _ in range(k):
            g,_=sample_grid(rng)
            for s,ds in _expose(g,colored_output(g,r)):
                mem.store(s,ds)
        g_new,act=sample_grid(rng)
        b=colored_output(g_new,r); rm={c:mem.map(c) for c in act}
        if all(rm[c]!=0 for c in act):
            bg=colored_output(g_new,{c:rm[c] for c in act})
        else:
            bg=colored_output(g_new,{act[0]:act[0],act[1]:act[1]})
        hit+=int(bg.tolist()==b.tolist())
    return hit/trials

for d in [2048,4096,8192,16384]:
    print(f"d={d:6d}  acc(8)={run(d,800,8):.3f}  acc(3)={run(d,800,3):.3f}  acc(1)={run(d,800,1):.3f}")