import torch
import triton
import triton.language as tl

RCP_LN2 = 1.4426950216


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
    q, k, v, lse, delta, do, dq, p_noise,
    scale, T,
    W: tl.constexpr,
    B: tl.constexpr, H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, USE_WINDOW: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    if IS_VARLEN:
        bos, eos = 0, 0
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
    RCP_LN2: tl.constexpr = 1.4426950216

    p_q  = tl.make_block_ptr(q  + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0),        (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0),        (BT, BK), (1, 0))
    p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_ls = tl.make_block_ptr(lse + bos * HQ + i_hq,       (T,),  (HQ,),    (i_t * BT,),            (BT,),    (0,))
    p_dt = tl.make_block_ptr(delta + bos * HQ + i_hq,     (T,),  (HQ,),    (i_t * BT,),            (BT,),    (0,))

    b_q  = tl.load(p_q,  boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_ls = tl.load(p_ls, boundary_check=(0,))
    b_dt = tl.load(p_dt, boundary_check=(0,))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    o_q = i_t * BT + tl.arange(0, BT)

    for i_s in range(0, tl.minimum((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s),        (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
        p_n = tl.make_block_ptr(p_noise + i_bh * T * T, (T, T), (T, 1), (i_t * BT, i_s), (BT, BS), (1, 0))
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        b_p = tl.where((o_q[:, None] >= o_k[None, :]) & m_k[None, :],
                       tl.math.exp2(b_s - b_ls[:, None]), 0.0)
        tl.store(p_n, b_p, boundary_check=(0, 1))
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_dt[:, None])
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))

    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


torch.manual_seed(0)
dtype = torch.float16
B, T, H, HQ, D = 1, 63, 1, 1, 64
scale = D ** -0.5
device = "cuda" if torch.cuda.is_available() else "npu"
BT, BS = 64, 32
BK = max(triton.next_power_of_2(D), 16)
BV = min(max(triton.next_power_of_2(D), 16), 64)
G = HQ // H

q = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
k = torch.randn((B, T, H, D), dtype=dtype, device=device)
v = torch.randn((B, T, H, D), dtype=dtype, device=device)
do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
q_f = q.float().squeeze(2)
k_f = k.float().squeeze(2)
v_f = v.float().squeeze(2)
do_f = do.float().squeeze(2)

s = (q_f * scale) @ k_f.mT
mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
s = s.masked_fill(~mask, -float('inf'))
o_f = (torch.softmax(s, dim=-1) @ v_f).unsqueeze(2)
lse_b2 = (torch.logsumexp(s, dim=-1) * RCP_LN2).unsqueeze(-1)

delta = torch.empty(B * T * HQ, dtype=torch.float, device=device)
bwd_kernel_preprocess[(delta.numel(),)](
    o_f, do, delta,
    B=triton.next_power_of_2(D), V=D,
)
delta = delta.view(B, T, HQ)

dp = do_f @ v_f.mT
dt_f = (o_f.squeeze(2) * do_f).sum(dim=-1)
ds = (torch.softmax(s, dim=-1)) * (dp - dt_f.unsqueeze(-1))
ds = ds.masked_fill(~mask, 0.)
dq_torch = (scale * (ds @ k_f)).unsqueeze(2)

grid = (triton.cdiv(D, BV), triton.cdiv(T, BT), B * HQ)
p_noise = torch.empty(B * HQ, T, T, dtype=torch.float32, device=device)

dq_triton = torch.empty(B, T, HQ, D, dtype=torch.float, device=device)
bwd_kernel_dq[grid](
    q, k, v, lse_b2, delta, do, dq_triton, p_noise,
    scale, T, W=None,
    B=B, H=H, HQ=HQ, G=G, K=D, V=D,
    BT=BT, BS=BS, BK=BK, BV=BV,
    USE_G=False, USE_WINDOW=False, IS_VARLEN=False,
)

print("dq diff:", (dq_triton - dq_torch).abs().max().item())
