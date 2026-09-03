/* ************************************************************************
 * Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_HIP_HIP_KERNELS_MCSR_HPP_
#define ROCALUTION_HIP_HIP_KERNELS_MCSR_HPP_

#include <hip/hip_runtime.h>

namespace rocalution
{

    template <unsigned int WF_SIZE, typename T, typename I, typename J>
    __global__ void kernel_mcsr_spmv(I nrow,
                                     const J* __restrict__ row_offset,
                                     const I* __restrict__ col,
                                     const T* __restrict__ val,
                                     const T* __restrict__ in,
                                     T* __restrict__ out)
    {
        const int gid    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        const int tid    = hipThreadIdx_x;
        const int laneid = tid & (WF_SIZE - 1);
        const int warpid = gid / WF_SIZE;
        const int nwarps = hipGridDim_x * hipBlockDim_x / WF_SIZE;

        for(I ai = warpid; ai < nrow; ai += nwarps)
        {
            J row_start = row_offset[ai];
            J row_end   = row_offset[ai + 1];

            T sum = static_cast<T>(0);

            for(J aj = row_start + laneid; aj < row_end; aj += WF_SIZE)
            {
                sum = sum + val[aj] * in[col[aj]];
            }

            wf_reduce_sum<WF_SIZE>(&sum);

            if(laneid == 0)
            {
                out[ai] = val[ai] * in[ai] + sum;
            }
        }
    }

    template <unsigned int WF_SIZE, typename T, typename I, typename J>
    __global__ void kernel_mcsr_add_spmv(I nrow,
                                         const J* __restrict__ row_offset,
                                         const I* __restrict__ col,
                                         const T* __restrict__ val,
                                         T scalar,
                                         const T* __restrict__ in,
                                         T* __restrict__ out)
    {
        const int gid    = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        const int tid    = hipThreadIdx_x;
        const int laneid = tid % WF_SIZE;
        const int warpid = gid / WF_SIZE;
        const int nwarps = hipGridDim_x * hipBlockDim_x / WF_SIZE;

        for(I ai = warpid; ai < nrow; ai += nwarps)
        {
            T sum = static_cast<T>(0);

            for(J aj = row_offset[ai] + laneid; aj < row_offset[ai + 1]; aj += WF_SIZE)
            {
                sum = sum + scalar * val[aj] * in[col[aj]];
            }

            wf_reduce_sum<WF_SIZE>(&sum);

            if(laneid == 0)
            {
                out[ai] = out[ai] + scalar * val[ai] * in[ai] + sum;
            }
        }
    }

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_KERNELS_MCSR_HPP_
