import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def gelu(input):
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * input))
    return cdf * input

dtype=torch.float16

@triton.autotune(
    configs=[
        triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 64, 'group_sz': 8}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 64, 'bsx': 256, 'bsk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 64, 'bsk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 128, 'bsk': 32, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 32, 'bsk': 32, 'group_sz': 8}, num_stages=5, num_warps=2),
        triton.Config({'bsy': 32, 'bsx': 64, 'bsk': 32, 'group_sz': 8}, num_stages=5, num_warps=2),
        triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 128, 'group_sz': 8}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 256, 'bsx': 128, 'bsk': 128, 'group_sz': 8}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 256, 'bsx': 64, 'bsk': 128, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 256, 'bsk': 128, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 128, 'bsk': 128, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 128, 'bsk': 64, 'group_sz': 8}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 32, 'bsk': 64, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 64, 'group_sz': 4}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 64, 'bsx': 256, 'bsk': 32, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 64, 'bsk': 32, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 128, 'bsk': 32, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 32, 'bsk': 32, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 32, 'bsx': 64, 'bsk': 32, 'group_sz': 4}, num_stages=5, num_warps=2),
        triton.Config({'bsy': 128, 'bsx': 256, 'bsk': 128, 'group_sz': 4}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 256, 'bsx': 128, 'bsk': 128, 'group_sz': 4}, num_stages=3, num_warps=8),
        triton.Config({'bsy': 64, 'bsx': 256, 'bsk': 128, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 128, 'bsk': 128, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 64, 'bsx': 128, 'bsk': 64, 'group_sz': 4}, num_stages=4, num_warps=4),
        triton.Config({'bsy': 128, 'bsx': 32, 'bsk': 64, 'group_sz': 4}, num_stages=4, num_warps=4)
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, O_ptr,
    A_stride_batch,
    A_stride_height, A_stride_width,
    B_stride_height, B_stride_width,
    O_stride_batch,
    O_stride_height, O_stride_width,
    M, N, K,
    bias_ptr,
    add_bias: tl.constexpr,
    apply_activation: tl.constexpr,
    activation: tl.constexpr,
    bsx: tl.constexpr, bsy: tl.constexpr, bsk: tl.constexpr, group_sz: tl.constexpr
):
    """
    Grouped Matmul
    O = A x B
    """
    batch_idx = tl.program_id(axis=0)
    row_idx = tl.program_id(axis=1)
    col_idx = tl.program_id(axis=2)

    num_row_programs = tl.num_programs(1)
    num_col_programs = tl.num_programs(2)

    row_idxnew, col_idxnew = tl.swizzle2d(row_idx, col_idx, num_row_programs, num_col_programs, group_sz)

    offset_batch = batch_idx * A_stride_batch

    acc = tl.zeros((bsy, bsx), dtype=tl.float32)    # accumulator

    for offset in range(0, K, bsk):
        offset_k = offset + tl.arange(0, bsk)

        # Read offsets from A_ptr
        offset_a = row_idxnew * bsy + tl.arange(0, bsy)
        offset_a = offset_a[:, None]*A_stride_height + offset_k[None, :]*A_stride_width  # bsy * bsk
        mask_a = row_idxnew * bsy + tl.arange(0, bsy)
        mask_a = (mask_a[:, None] < M) & (offset_k[None, :] < K)
        a = tl.load(A_ptr + offset_batch + offset_a, mask_a)

        # Read offset from B_ptr
        offset_b = col_idxnew * bsx + tl.arange(0, bsx)
        offset_b = offset_k[:, None]*B_stride_height + offset_b[None, :]*B_stride_width  # bsk * bsx
        mask_b = col_idxnew * bsx + tl.arange(0, bsx)
        mask_b = (offset_k[:, None] < K) & (mask_b[None, :] < N)
        b = tl.load(B_ptr + offset_b, mask_b)

        # acc += tl.dot(a, b, allow_tf32=True)  # triton old version
        acc = tl.dot(a, b, acc, allow_tf32=True)
        
        

    offset_batch_out = batch_idx * O_stride_batch
    offset_or = row_idxnew * bsy + tl.arange(0, bsy)
    offset_oc = col_idxnew * bsx + tl.arange(0, bsx)
    offset_o = offset_or[:, None]*O_stride_height+ offset_oc[None, :]*O_stride_width  # bsy * bsx
    mask_o = (offset_or[:, None] < M) & (offset_oc[None, :] < N)

    if add_bias:
        bias = tl.load(bias_ptr + offset_oc, offset_oc < N)
        acc += bias[None, :]

    if apply_activation:
        if activation == 'gelu':
            acc = gelu(acc)

    tl.store(O_ptr + offset_batch_out + offset_o, acc, mask_o)


def matmul_triton(A: torch.Tensor, B: torch.Tensor, bias: Optional[torch.Tensor] = None, activation: Optional[str] = None) -> torch.Tensor:
    """
    Implements matrix multiplication between input matrix A and B
    
    Args:
        - A {torch.Tensor}: Input matrix with shape (B, T, Cin) where B is the batch size, T is the sequence length, Cin is the input dimension
        - B {torch.Tensor}: Weight matrix with shape (Cin, Cout) where Cout is the hidden dimension
        - bias {torch.Tensor}: Optionally add a bias to the ouput, shape (1, Cout)
        - activation {str}: Optionally apply activation to the ouput

    Returns:
        - {torch.Tensor}: Output tensor with (B, T, Cout)
    """
    assert len(A.shape) == 3, f"First input matrix needs to have 3 dimensions (B, T, C) but {A.shape}"
    assert A.device == B.device and A.is_cuda, "Both matrix should be on GPU"

    if bias is not None:
        assert bias.is_cuda, "Bias is not on GPU"
        bias = bias.unsqueeze(0)
        assert bias.shape[1] == B.shape[1], "Bias shape does not match output feature dimension shape"

    if activation:
        assert activation in ["gelu"], f"Only GELU activation supported as of now! Provided: {activation}"

    batch_size, M, K = A.shape
    K, N = B.shape

    grid = lambda meta: (batch_size, triton.cdiv(M, meta["bsy"]), triton.cdiv(N, meta["bsx"]))

    O = torch.empty((batch_size, M, N), device=A.device, dtype=A.dtype)

    matmul_kernel[grid](
        A, B, O,
        A_stride_batch=A.stride(0),
        A_stride_height=A.stride(1), A_stride_width=A.stride(2),
        B_stride_height=B.stride(0), B_stride_width=B.stride(1),
        O_stride_batch=O.stride(0),
        O_stride_height=O.stride(1), O_stride_width=O.stride(2),
        M=M, N=N, K=K,
        bias_ptr=bias,
        add_bias=True if bias is not None else False,
        activation=activation,
        apply_activation=True if activation else False
    )

    return O