"""E2 集成验证：读出头路径 30k 步曲线（下穿 unigram 4.797 且持续下降）。"""
import time
import numpy as np
from srpc.config import E2Config
from srpc.lm import ByteCorpus, train_lmpcn

cfg = E2Config()
cfg.eval_every = 5000
corpus = ByteCorpus(cfg)
print(f"unigram BPC: {corpus.unigram_bpc:.3f}")
t0 = time.time()
r = train_lmpcn(cfg, 768, 30000, corpus, eta_w=0.01, iters=24, seed=0)
print(f"final: bpc={r['bpc']:.3f} acc={r['acc']:.3f} tau={r['tau']} "
      f"wall={r['wall']}s")
for pt in r["curve"]:
    print(f"  step={pt['step']}: bpc={pt['bpc']:.3f} acc={pt['acc']:.3f} "
          f"tau={pt['tau']} event={pt['event_rate']:.3f}")
