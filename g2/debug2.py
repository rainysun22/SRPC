"""G2a 快速 debug：单色 map_color 对率拆解（确认 cleanup 是否真能解对）。"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2_reasoning import (make_vocab, VsaColorMemory, bind, unbind,
                          sample_grid, colored_output, _expose, ALL_PERMS)

d = 2048
V = make_vocab(d, seed=0)
tri=0; ok=0; covered_tri=0; covered_ok=0
per_color_ok=[0,0,0,0,0]; per_color_n=[0,0,0,0,0]
for trial in range(600):
    r=ALL_PERMS[np.random.default_rng(trial).integers(len(ALL_PERMS))]
    mem=VsaColorMemory(V)
    rng=np.random.default_rng(90+trial)
    for _ in range(8):
        g,_=sample_grid(rng)
        go=colored_output(g,r)
        for s,ds in _expose(g,go):
            mem.store_pair(s,ds)
    for c in range(1,6):
        pred=mem.map_color(c)
        per_color_n[c-1]+=1
        if pred==r[c]:
            per_color_ok[c-1]+=1
            if mem.covered[c-1]:
                covered_tri+=1; covered_ok+=1
        else:
            if mem.covered[c-1]:
                covered_tri+=1
print("covered 色解对率: %.4f  (%d/%d)"%(covered_ok/max(covered_tri,1),covered_ok,covered_tri))
print("推理前精度 baseline cover rate check:")
print("per-color acc:", [round(per_color_ok[i]/max(per_color_n[i],1),3) for i in range(5)])
print("covered flags count per trial =", [per_color_n[i] for i in range(5)])