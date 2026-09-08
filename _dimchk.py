import torch, time
sd = torch.load("/root/srpc_e2/srpc_src/results_e2_gpu/diag_345000.pt",
                map_location="cpu", weights_only=False)["model"]
for k in ["W1c", "W2", "W3", "W_out", "m2"]:
    print(k, tuple(sd[k].shape), sd[k].dtype)
W2 = sd["W2"].cuda()
# time an SVD to gauge per-step cap cost
for _ in range(3):
    t0 = time.time(); u, s, v = torch.linalg.svd(W2, full_matrices=False)
    torch.cuda.synchronize()
print("SVD time(ms): %.2f" % ((time.time() - t0) * 1e3))
print("smax=%.3f" % float(s[0]))