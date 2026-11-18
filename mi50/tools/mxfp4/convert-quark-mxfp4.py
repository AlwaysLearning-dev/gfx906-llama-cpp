#!/usr/bin/env python3
"""Convert an AMD Quark MXFP4 checkpoint (safetensors with packed 4-bit weights) to GGUF without re-quantizing.

Quark's "real_quantized" export stores every quantized linear weight as
  <name>.weight        uint8 [rows, cols/2]   two E2M1 codes per byte, element 2i in the low nibble
  <name>.weight_scale  uint8 [rows, cols/32]  one E8M0 exponent per 32 values
which is the same information as ggml's block_mxfp4, in a different order. This wrapper runs llama.cpp's own
convert_hf_to_gguf.py and adds the one missing piece: moving those bits into GGML_TYPE_MXFP4 tensors (using the
converter's existing lossless repack_mxfp4_blocks), with the same value-head reorder the converter applies to the
float weights of Qwen3.5-style linear attention. Everything else (norms, embeddings, tokenizer, metadata) goes
through the stock converter.

Usage: convert-quark-mxfp4.py <llama.cpp source dir> <convert_hf_to_gguf.py arguments...>
"""
import sys, runpy
SRC = sys.argv.pop(1)
sys.path.insert(0, SRC); sys.path.insert(0, SRC + "/gguf-py")
import torch, gguf
from conversion.base import ModelBase, LazyTorchTensor, logger

_orig_dequant = ModelBase.dequant_model

def _reorder_linear_attn(self, name, weight, scale):
    """Value-head reorder (grouped -> tiled) on packed codes; mirrors _LinearAttentionVReorderBase for 32-value groups."""
    hp = self.hparams
    nk, nv = hp.get("linear_num_key_heads", 0), hp.get("linear_num_value_heads", 0)
    if not (nk > 0 and nv > 0 and nk != nv and ".linear_attn." in name):
        return weight, scale
    dk, dv = hp["linear_key_head_dim"], hp["linear_value_head_dim"]
    per_k = nv // nk
    def perm(n_heads_dim):  # index permutation over nv * n_heads_dim positions
        idx = torch.arange(nv * n_heads_dim, dtype=torch.long).reshape(nk, per_k, n_heads_dim)
        return idx.permute(1, 0, 2).contiguous().reshape(-1)
    def rows(w, s, head_dim):
        p = perm(head_dim); return w.index_select(0, p), s.index_select(0, p)
    if name.endswith(".in_proj_qkv.weight"):
        qk = 2 * dk * nk
        vw, vs = rows(weight[qk:], scale[qk:], dv)
        return torch.cat([weight[:qk], vw], 0), torch.cat([scale[:qk], vs], 0)
    if name.endswith(".in_proj_z.weight"):
        return rows(weight, scale, dv)
    if name.endswith((".in_proj_a.weight", ".in_proj_b.weight")):
        return rows(weight, scale, 1)
    if name.endswith(".out_proj.weight"):
        p = perm(dv)                                   # column permutation, moves whole heads
        assert dv % 32 == 0 and p.numel() == weight.shape[1] * 2
        g = p.reshape(-1, 32)
        assert torch.equal(g, g[:, :1] + torch.arange(32)) and torch.all(g[:, 0] % 32 == 0)
        codes = torch.stack((weight & 0x0F, weight >> 4), -1).reshape(weight.shape[0], -1).index_select(1, p)
        codes = codes.reshape(weight.shape[0], -1, 2)
        return (codes[..., 0] | (codes[..., 1] << 4)).contiguous(), scale.index_select(1, g[:, 0] // 32)
    return weight, scale

def _dequant_model(self):
    qc = self.hparams.get("quantization_config")
    if not (isinstance(qc, dict) and qc.get("quant_method") == "quark"):
        return _orig_dequant(self)
    w = (qc.get("global_quant_config") or {}).get("weight") or {}
    ex = qc.get("export") or {}
    if not (w.get("dtype") == "fp4" and w.get("group_size") == 32 and w.get("scale_format") == "e8m0"
            and w.get("qscheme") == "per_group" and ex.get("weight_format") == "real_quantized"):
        raise NotImplementedError(f"Quark export is not packed MXFP4 (group 32, E8M0): {w} {ex}")
    if qc.get("layer_quant_config") or qc.get("layer_type_quant_config"):
        raise NotImplementedError("per-layer Quark configs are not handled")
    consumed, n = [], 0
    for name in list(self.model_tensors.keys()):
        sname = name[:-len(".weight")] + ".weight_scale" if name.endswith(".weight") else None
        if sname is None or sname not in self.model_tensors:
            continue
        weight = LazyTorchTensor.to_eager(self.model_tensors[name]())
        scale = LazyTorchTensor.to_eager(self.model_tensors[sname]())
        if weight.dtype != torch.uint8 or scale.dtype != torch.uint8 or weight.ndim != 2 or scale.ndim != 2:
            raise ValueError(f"{name}: expected packed uint8 weight and scale, got {weight.dtype}{tuple(weight.shape)} / {scale.dtype}{tuple(scale.shape)}")
        weight, scale = _reorder_linear_attn(self, name, weight, scale)
        raw = self.repack_mxfp4_blocks(weight, scale)
        new_name = self.map_tensor_name(name)
        logger.info(f"Repacked {new_name} [{weight.shape[0]}, {weight.shape[1] * 2}] as MXFP4 (bits moved, not re-quantized)")
        self.gguf_writer.add_tensor(new_name, raw, raw_dtype=gguf.GGMLQuantizationType.MXFP4)
        consumed += [name, sname]; n += 1
    for name in consumed:
        self.model_tensors.pop(name, None)
    left = [k for k in self.model_tensors if k.endswith(".weight_scale")]
    if left:
        raise ValueError(f"unhandled scale tensors: {left[:5]}")
    logger.info(f"Quark MXFP4: repacked {n} tensors")

ModelBase.dequant_model = _dequant_model
sys.argv[0] = SRC + "/convert_hf_to_gguf.py"
runpy.run_path(sys.argv[0], run_name="__main__")
