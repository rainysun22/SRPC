"""Debug: 绑定叠加数量对单绑定解绑对率的影响（定位 0.70 上限来源）。"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2_reasoning import make_vocab, VsaColorMemory, bind, unbind

d=2048; V=make_vocab(d,seed=0)
def clean_trial(V, keyc, dstc):
    mem=VsaColorMemory(V)
    mem.store_pair(keyc,dstc)
    return mem.map_color(keyc)

# 1) 单绑定
ok=sum(int(clean_trial(V,(k%5)+1,((k+1)%5)+1)==((k+1)%5)+1) for k in range(2000))
print("单绑定解对率:", ok/2000)

# 2) 随绑定叠加数(n_pair)与重复次数，固定一目标键的长尾对率
for n_pair in [2,4,8,12,16,24]:
    okc=0; n=0
    for t in range(400):
        rng=np.random.default_rng(t)
        mem=VsaColorMemory(V)
        for _ in range(n_pair):
            k=(rng.integers(5))%5+1; d_=(rng.integers(5))%5+1
            mem.store_pair(k,d_)
        # 测第一键的解对：跟踪它最后绑定的 dst
        # 简化：重放第一键绑定的 dst 作为真值
        mem2=VsaColorMemory(V)
        k0=1; dst0=None
        for _ in range(n_pair):
            k=(rng.integers(5))%5+1; dd=(rng.integers(5))%5+1
        # 上面消耗了统一rng，重做一次带真值
        rng2=np.random.default_rng(t)
        for j in range(n_pair):
            k=int(rng2.integers(5))+1; dd=int(rng2.integers(5))+1
            if j==0: k0=k; dst0=dd
            mem2.store_pair(k,dd)
        if dst0 is None: continue
        pred=mem2.map_color(k0)
        okc+=int(pred==dst0); n+=1
    print(f"n_pair={n_pair:3d}  解对率={okc/max(n,1):.3f} (干扰键可能重复→真值仅记录首个)")