#include "common.cuh"
#include "cross-entropy-loss.cuh"
#include "sum.cuh"

#include <cmath>
#include <cstdint>

static __global__ void accuracy_count_f32(
    const float * logits, const float * labels, uint64_t * dst,
    const int64_t ncols, const int64_t nrows, const int64_t rows_per_block) {

    const int64_t row_start = (int64_t)blockIdx.x*rows_per_block;
    const int64_t row_stop  = min(row_start + rows_per_block, nrows);

    int64_t ncorrect = 0;

    for (int64_t row = row_start; row < row_stop; ++row) {
        float   max_logits    = -FLT_MAX;
        float   max_labels    = -FLT_MAX;
        int64_t argmax_logits = -1;
        int64_t argmax_labels = -1;

        for (int64_t col = threadIdx.x; col < ncols; col += WARP_SIZE) {
            {
                const float val        = logits[row*ncols + col];
                const int   bigger     = val > max_logits;
                const int   not_bigger = bigger ^ 0x00000001;

                max_logits    =    max_logits*not_bigger + val*bigger;
                argmax_logits = argmax_logits*not_bigger + col*bigger;
            }
            {
                const float val        = labels[row*ncols + col];
                const int   bigger     = val > max_labels;
                const int   not_bigger = bigger ^ 0x00000001;

                max_labels    =    max_labels*not_bigger + val*bigger;
                argmax_labels = argmax_labels*not_bigger + col*bigger;
            }
        }

#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float val        = __shfl_xor_sync(0xFFFFFFFF,    max_logits, mask, WARP_SIZE);
            const float col        = __shfl_xor_sync(0xFFFFFFFF, argmax_logits, mask, WARP_SIZE);
            const int   bigger     = val > max_logits;
            const int   not_bigger = bigger ^ 0x00000001;

            max_logits    =    max_logits*not_bigger + val*bigger;
            argmax_logits = argmax_logits*not_bigger + col*bigger;
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float val        = __shfl_xor_sync(0xFFFFFFFF,    max_labels, mask, WARP_SIZE);
            const float col        = __shfl_xor_sync(0xFFFFFFFF, argmax_labels, mask, WARP_SIZE);
            const int   bigger     = val > max_labels;
            const int   not_bigger = bigger ^ 0x00000001;

            max_labels    =    max_labels*not_bigger + val*bigger;
            argmax_labels = argmax_labels*not_bigger + col*bigger;
        }

        ncorrect += argmax_labels == argmax_logits;
    }

    atomicAdd(dst, ncorrect);
}

static __global__ void cross_entropy_loss_back_f32(const float * logits, const float * labels, const float * loss, float * dst, const int nclasses) {
    extern __shared__ float tmp[];

    float maxval = -INFINITY;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = logits[blockIdx.x*nclasses + i];
        maxval = fmaxf(maxval, val);
        tmp[i] = val;
    }
    maxval = warp_reduce_max(maxval);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = expf(tmp[i] - maxval);
        sum += val;
        tmp[i] = val;
    }
    sum = warp_reduce_sum(sum);
    const float sm_scale = 1.0f/sum;

    const float d_by_nrows = *loss/gridDim.x;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        dst[blockIdx.x*nclasses + i] = (tmp[i]*sm_scale - labels[blockIdx.x*nclasses + i])*d_by_nrows;
    }
}

void ggml_cuda_cross_entropy_loss(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

    const dim3 blocks_dim(CUDA_CROSS_ENTROPY_LOSS_BLOCK_SIZE, 1, 1);
    const dim3 blocks_num((nrows + CUDA_CROSS_ENTROPY_LOSS_BLOCK_SIZE - 1) / CUDA_CROSS_ENTROPY_LOSS_BLOCK_SIZE, 1, 1);
    const int shmem = 2*CUDA_CROSS_ENTROPY_LOSS_BLOCK_SIZE*ne00*sizeof(float);

    ggml_cuda_pool_alloc<float> dst_tmp(pool, blocks_num.x);

    cross_entropy_loss_f32<<<blocks_num, blocks_dim, shmem, stream>>>(src0_d, src1_d, dst_tmp.ptr, ne00, nrows);

    // Combine results from individual blocks:
    sum_f32_cuda(pool, dst_tmp.ptr, dst_d, blocks_num.x, stream);
}

void ggml_cuda_cross_entropy_loss_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * opt0 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(opt0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(opt0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * opt0_d = (const float *) opt0->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(nrows, 1, 1);
    const int shmem = ne00*sizeof(float);

    cross_entropy_loss_back_f32<<<blocks_num, blocks_dim, shmem, stream>>>(src0_d, src1_d, opt0_d, dst_d, ne00);
}
