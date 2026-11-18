#!/usr/bin/env python3
"""Check a converted MXFP4 GGUF against a reference GGUF of the same model (e.g. Q8_0):
same tensor names and shapes, and each MXFP4 tensor decoded with ggml's formula must point the same way as the
reference weights (cosine similarity per tensor; a wrong bit layout or row order gives about 0).
Usage: verify-gguf.py <gguf-py dir> NEW.gguf REF.gguf [max tensors to decode]"""
import sys; sys.path.insert(0, sys.argv[1])
import numpy as np
from gguf import GGUFReader, quants, GGMLQuantizationType as T
new, ref = GGUFReader(sys.argv[2]), GGUFReader(sys.argv[3]); lim = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
K = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)
def dq_mxfp4(t):
    rows = int(t.shape[1]) if len(t.shape) > 1 else 1; cols = int(t.shape[0])
    b = np.asarray(t.data).reshape(rows, cols // 32, 17)
    d = np.ldexp(np.float32(1), b[:, :, 0].astype(np.int32) - 128)[..., None]       # 0.5 * 2^(e-127)
    q = b[:, :, 1:]
    return (np.concatenate([K[q & 15], K[q >> 4]], -1) * d).reshape(rows, cols)
def dq(t):
    if t.tensor_type == T.MXFP4: return dq_mxfp4(t)
    if t.tensor_type in (T.F32, T.F16, T.BF16): return np.asarray(t.data, dtype=np.float32)
    return quants.dequantize(np.asarray(t.data), t.tensor_type)
R = {t.name: t for t in ref.tensors}; N = {t.name: t for t in new.tensors}
print("tensors: new %d, ref %d; only in new: %s; only in ref: %s" % (len(N), len(R), sorted(set(N) - set(R))[:6], sorted(set(R) - set(N))[:6]))
bad_shape = [(n, list(N[n].shape), list(R[n].shape)) for n in N if n in R and list(N[n].shape) != list(R[n].shape)]
print("shape mismatches:", bad_shape[:6])
import collections; print("types:", dict(collections.Counter(t.tensor_type.name for t in new.tensors)))
worst = []; done = 0; kinds = collections.defaultdict(list)
for n, t in N.items():
    if n not in R or t.tensor_type != T.MXFP4: continue
    kind = n.split(".", 2)[2]
    if done >= lim and len(kinds[kind]) >= 2: continue
    a = dq(t).reshape(-1).astype(np.float64); b = dq(R[n]).reshape(-1).astype(np.float64)
    a2 = dq(t); b2 = dq(R[n]).reshape(a2.shape)
    rowcos = (a2 * b2).sum(1) / (np.linalg.norm(a2, axis=1) * np.linalg.norm(b2, axis=1) + 1e-30)
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)); kinds[kind].append((c, float(rowcos.min()))); worst.append((c, n)); done += 1
for k, v in sorted(kinds.items()):
    print("  %-26s n=%3d  cosine min %.4f mean %.4f   worst single row %.3f" % (k, len(v), min(x[0] for x in v), sum(x[0] for x in v) / len(v), min(x[1] for x in v)))
print("lowest:", [(round(c, 4), n) for c, n in sorted(worst)[:4]])
for n in ("output_norm.weight", "blk.0.attn_norm.weight", "blk.0.ssm_a", "blk.0.ssm_dt.bias", "blk.0.ssm_norm.weight", "blk.0.ssm_conv1d.weight", "blk.3.attn_k_norm.weight", "token_embd.weight", "output.weight"):
    if n in N and n in R:
        a = dq(N[n]).reshape(-1).astype(np.float64); b = dq(R[n]).reshape(-1).astype(np.float64)
        print("  %-26s %-5s cosine %.5f  max abs diff %.4g" % (n, N[n].tensor_type.name, a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), np.abs(a - b).max()))
