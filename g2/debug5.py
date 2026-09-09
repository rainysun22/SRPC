"""Debug5: 去重+高维下，*覆盖条件* 解绑对率（探针色⊆已覆盖）分离覆盖与解绑。"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2_reasoning import make_vocab, bind, unbind, sample_grid, colored_output, _expose, ALL_PERMS

class Mem:
    def __init__(s,V): s.V=V; s.d=V.shape[1]; s.S=np.zeros(s.d); s.cov=np.zeros(5,bool)
    def store(s,c,cp):
        if s.cov[c-1]: return
        s.S+=bind(s.V[c-1],s.V[cp-1]); s.cov[c-1]=True
    def map(s,c): return 0 if not s.cov[c-1] else int((s.V@(lambda q:q/np.linalg.norm(q))(unbind(s.S,s.V[c-1]))).argmax())+1

def covered_cond(d,trials,k):
    V=make_vocab(d,seed=0); both_ok=0; both_n=0; single_ok=0; single_n=0
    for t in range(trials):
        r=ALL_PERMS[np.random.default_rng(51+t).integers(len(ALL_PERMS))]
        rng=np.random.default_rng(52+t); mem=Mem(V)
        for _ in range(k):
            g,_=sample_grid(rng)
            for s,ds in _expose(g,colored_output(g,r)): mem.store(s,ds)
        # 直接评估每色解对率
        for c in range(1,6):
            single_n+=1; single_ok+=int(mem.map(c)==r[c])
        # 双色条件(探针从已覆盖色取, 简单取覆盖全时)
        if mem.cov.all():
            act=list(rng.choice(COL:=range(1,6),2,replace=False)); both_n+=1
            both_ok+=int(all(mem.map(c)==r[c] for c in act))
    return single_ok/single_n, (both_ok/both_n if both_n else float('nan'))

for d in [2048,4096]:
    single,both=covered_cond(d,1500,8)
    print(f"d={d:6d}  覆盖条件单色对率={single:.3f}  双色={both:.3f}  (过covered占比={both>0})")