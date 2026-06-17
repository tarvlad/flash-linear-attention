import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Step 1: p pre-computed by torch (softmax), passed to kernel.  No exp2 in Triton.
# Isolates: tl.dot, tl.load/store, block_ptr, arithmetic, accumulation.
# ---------------------------------------------------------------------------

@triton.jit
def bwd_kernel_preprocess(o, do, delta, B: tl.constexpr, V: tl.constexpr):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V
    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    tl.store(delta + i_n, tl.sum(b_o * b_do).to(delta.dtype.element_ty))


@triton.jit
def bwd_kernel_dq(
    q, k, v, p, delta, do, dq,
    scale, T,
    B: tl.constexpr, H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T
    o_q = i_t * BT + tl.arange(0, BT)

    p_q  = tl.make_block_ptr(q  + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0),        (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0),        (BT, BK), (1, 0))
    p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dt = tl.make_block_ptr(delta + bos * HQ + i_hq,     (T,),  (HQ,),    (i_t * BT,),            (BT,),    (0,))

    b_q  = tl.load(p_q,  boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dt = tl.load(p_dt, boundary_check=(0,))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)

    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s),        (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
        p_p = tl.make_block_ptr(p + i_bh * T * T, (T, T), (T, 1), (i_t * BT, i_s), (BT, BS), (1, 0))
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_p = tl.load(p_p, boundary_check=(0, 1)).to(tl.float32)
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_dt[:, None])
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))

    for i_s in range(i_t * BT, tl.minimum((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s),        (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
        p_p = tl.make_block_ptr(p + i_bh * T * T, (T, T), (T, 1), (i_t * BT, i_s), (BT, BS), (1, 0))
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_p = tl.load(p_p, boundary_check=(0, 1)).to(tl.float32)
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_dt[:, None])
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))

    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(0)
    dtype = torch.float16
    B, T, H, HQ, D = 1, 63, 1, 1, 64
    scale = D ** -0.5

    device = "cuda" if torch.cuda.is_available() else "npu"

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    # --- torch forward ---
    q_f = q.float().squeeze(2)
    k_f = k.float().squeeze(2)
    v_f = v.float().squeeze(2)
    do_f = do.float().squeeze(2)

    s = (q_f * scale) @ k_f.mT
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    s = s.masked_fill(~mask, -float('inf'))

    p_f = torch.softmax(s, dim=-1)
    p = p_f.unsqueeze(1).expand(B, HQ, T, T).contiguous()
    o_f = (p_f @ v_f).unsqueeze(2)

    # --- delta (preprocess) ---
    delta = torch.empty(B * T * HQ, dtype=torch.float, device=device)
    bwd_kernel_preprocess[(delta.numel(),)](
        o_f, do, delta,
        B=triton.next_power_of_2(D), V=D,
    )
    delta = delta.view(B, T, HQ)

    # --- Triton dq ---
    BT, BS = 64, 32
    BK = max(triton.next_power_of_2(D), 16)
    BV = min(max(triton.next_power_of_2(D), 16), 64)
    G = HQ // H

    dq_triton = torch.empty(B, T, HQ, D, dtype=torch.float, device=device)
    bwd_kernel_dq[(triton.cdiv(D, BV), triton.cdiv(T, BT), B * HQ)](
        q=q, k=k, v=v, p=p, delta=delta, do=do, dq=dq_triton,
        scale=scale, T=T,
        B=B, H=H, HQ=HQ, G=G, K=D, V=D,
        BT=BT, BS=BS, BK=BK, BV=BV,
    )

    # --- torch dq ---
    dp = do_f @ v_f.mT
    dt_f = (o_f.squeeze(2) * do_f).sum(dim=-1)
    ds = p_f * (dp - dt_f.unsqueeze(-1))
    ds = ds.masked_fill(~mask, 0.)
    dq_torch = (scale * (ds @ k_f)).unsqueeze(2)

    print("o  diff:", (o_f - (p_f @ v_f).unsqueeze(2)).abs().max().item())
    print("dq diff:", (dq_triton - dq_torch).abs().max().item())


if __name__ == "__main__":
    main()
