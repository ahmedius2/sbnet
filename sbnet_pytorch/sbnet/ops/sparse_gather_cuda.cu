/*

   Sparse Blocks Network
   Copyright (c) 2017, Uber Technologies, Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/

/*
From GitHub page:

Current code is not tuned for performance with non-square block sizes and has specialized implementations for a specific list of block sizes. This includes square blocks of sizes 1 to 34 and a few others. To achieve maximum performance for these sizes you would need to add your custom template instantiations by modifying SIZE_TEMPLATES macro in sparse_gather.cu
*/

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_runtime.h>
#include "sparse_blocks.cu.h"
#include "cuda_helpers.h"
#include "op_utils.h"

#define COMPUTE_R1(RR) ((RR) < 7 ? ((RR) == 1 ? 1 : 2) : 4)

namespace {
template<typename scalar_t>
struct LaunchParams {
    dim3 block, grid;
    int shmemSize;
    int bSzH1;
    int fittingC1;
    enum { MAX_SHMEM = 24*1024 };
    LaunchParams(int C, int bSzH, int bSzW, int numActive)
    {
        fittingC1 = std::min(32, C);
        bSzH1 = COMPUTE_R1(bSzH);
        while ((shmemSize = (fittingC1+1)*bSzH1*bSzW*sizeof(scalar_t)) > MAX_SHMEM)
            fittingC1--;
        assert(fittingC1 >= 1);
        assert(bSzH1*bSzW*(fittingC1+1)*sizeof(scalar_t) <= MAX_SHMEM);
        block = dim3(512, 1, 1);
        grid = dim3(numActive, DIVUP(C, fittingC1), DIVUP(bSzH, bSzH1));
    }
};
}

// Define the GPU implementation that launches the CUDA kernel.
torch::Tensor LaunchSparseGatherGPU(
        torch::Tensor x, int N, int H, int W, int C, torch::Tensor y,
        int bOffsH0, int bOffsW0, int bSzH, int bSzW, int bStrH, int bStrW,
        int numActive, torch::Tensor activeBlockIndices, bool transpose)
{
	bool hasInst = false;
	cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
#if 1
	#define CALL(RR, CC1, trans) \
		if (bSzH == RR && bSzW == RR && lp.fittingC1 == CC1) { \
			hasInst = true; \
			blockGatherTiled0<scalar_t, 512, RR, COMPUTE_R1(RR), RR, CC1, trans><<<lp.grid, lp.block, lp.shmemSize, stream>>>( \
				x.data_ptr<scalar_t>(), (const int16_t*)activeBlockIndices.data_ptr<int16_t>(), \
				y.data_ptr<scalar_t>(), N, H, W, C, bOffsH0, bOffsW0, bStrH, bStrW); \
		} else

	#define SIZE_TEMPLATES(transt, CCC) \
		CALL( 1, CCC, transt) \
		CALL( 2, CCC, transt) \
		CALL( 3, CCC, transt) \
		CALL( 4, CCC, transt) \
		CALL( 5, CCC, transt) \
		CALL( 6, CCC, transt) \
		CALL( 7, CCC, transt) \
		CALL( 8, CCC, transt) \
		CALL( 9, CCC, transt) \
		CALL(10, CCC, transt) \
		CALL(11, CCC, transt) \
		CALL(12, CCC, transt) \
		CALL(13, CCC, transt) \
		CALL(14, CCC, transt) \
		CALL(15, CCC, transt) \
		CALL(16, CCC, transt) \
		CALL(17, CCC, transt) \
		CALL(18, CCC, transt) \
		CALL(19, CCC, transt) \
		CALL(20, CCC, transt) \
		CALL(21, CCC, transt) \
		CALL(22, CCC, transt) \
		CALL(23, CCC, transt) \
		CALL(24, CCC, transt) \
		CALL(25, CCC, transt) \
		CALL(26, CCC, transt) \
		CALL(27, CCC, transt) \
		CALL(28, CCC, transt) \
		CALL(29, CCC, transt) \
		CALL(30, CCC, transt) \
		CALL(31, CCC, transt) \
		CALL(32, CCC, transt) \
		CALL(33, CCC, transt) \
		CALL(34, CCC, transt) \
		CALL(41, CCC, transt) \
		CALL(48, CCC, transt) \
		CALL(63, CCC, transt) \
		CALL(64, CCC, transt) \
		CALL(65, CCC, transt) \
		CALL(81, CCC, transt) \
		   { hasInst = false; }

	AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "SparseGatherGPU", ([&] {
		LaunchParams<scalar_t> lp(C, bSzH, bSzW, numActive);
		if (transpose) {
			if (lp.fittingC1 >= 32) {
				SIZE_TEMPLATES(true, 32)
			} else if (lp.fittingC1 == 16) {
				SIZE_TEMPLATES(true, 16)
			} else if (lp.fittingC1 == 24) {
				SIZE_TEMPLATES(true, 24)
			}
		} else {
			if (lp.fittingC1 >= 32) {
				SIZE_TEMPLATES(false, 32)
			} else if (lp.fittingC1 == 16) {
				SIZE_TEMPLATES(false, 16)
			} else if (lp.fittingC1 == 24) {
				SIZE_TEMPLATES(false, 24)
			}
		}
#endif
		if (!hasInst)
		{
			//printf("gather, C, bSzH, bSzW=%d, %d, %d, fittingC1=%d\n", C, bSzH, bSzW, lp.fittingC1);
			blockGatherTiled1<scalar_t, 512><<<lp.grid, lp.block, lp.shmemSize, stream>>>(
				x.data_ptr<scalar_t>(), (const int16_t*)activeBlockIndices.data_ptr<int16_t>(),
				y.data_ptr<scalar_t>(), N, H, W, C, bOffsH0, bOffsW0, bStrH, bStrW,
				bSzH, lp.bSzH1, bSzW, lp.fittingC1, transpose);
		}

	}));
	#undef SIZE_TEMPLATES
	#undef CALL
	gpuErrorCheck( cudaPeekAtLastError() );

	return y;
}


// Define the GPU implementation that launches the CUDA kernel.
//template <typename T> struct SparseScatterFunctor<GPUDevice, T> {
//    void operator()(
//        const GPUDevice& d,
torch::Tensor LaunchSparseScatterGPU(
        torch::Tensor x, int N, int H, int W, int C, torch::Tensor y,
        int bOffsH0, int bOffsW0, int bSzH, int bSzW, int bStrH, int bStrW,
        int numActive, torch::Tensor activeBlockIndices, bool add, bool transpose, bool atomic
)
{
	bool hasInst = false;
	cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
#if 1
	#define CALL(RR, CC1, addt, transt) \
		if (bSzH == RR && bSzW == RR && lp.fittingC1 == CC1 && atomic == false) { \
			hasInst = true; \
			blockScatterTiled0<scalar_t, 512, RR, COMPUTE_R1(RR), RR, CC1, addt, transt, false> \
				<<<lp.grid, lp.block, lp.shmemSize, stream>>>( \
					x.data_ptr<scalar_t>(), \
					(const int16_t*)activeBlockIndices.data_ptr<int16_t>(), \
					y.data_ptr<scalar_t>(), N, H, W, C, bOffsH0, bOffsW0, bStrH, bStrW); \
		} else

	#define SIZE_TEMPLATES(addt, transpt, CCC) \
		CALL( 1, CCC, addt, transpt) \
		CALL( 2, CCC, addt, transpt) \
		CALL( 3, CCC, addt, transpt) \
		CALL( 4, CCC, addt, transpt) \
		CALL( 5, CCC, addt, transpt) \
		CALL( 6, CCC, addt, transpt) \
		CALL( 7, CCC, addt, transpt) \
		CALL( 8, CCC, addt, transpt) \
		CALL( 9, CCC, addt, transpt) \
		CALL(10, CCC, addt, transpt) \
		CALL(11, CCC, addt, transpt) \
		CALL(12, CCC, addt, transpt) \
		CALL(13, CCC, addt, transpt) \
		CALL(14, CCC, addt, transpt) \
		CALL(15, CCC, addt, transpt) \
		CALL(16, CCC, addt, transpt) \
		CALL(17, CCC, addt, transpt) \
		CALL(18, CCC, addt, transpt) \
		CALL(19, CCC, addt, transpt) \
		CALL(20, CCC, addt, transpt) \
		CALL(21, CCC, addt, transpt) \
		CALL(22, CCC, addt, transpt) \
		CALL(23, CCC, addt, transpt) \
		CALL(24, CCC, addt, transpt) \
		CALL(25, CCC, addt, transpt) \
		CALL(26, CCC, addt, transpt) \
		CALL(27, CCC, addt, transpt) \
		CALL(28, CCC, addt, transpt) \
		CALL(29, CCC, addt, transpt) \
		CALL(30, CCC, addt, transpt) \
		CALL(31, CCC, addt, transpt) \
		CALL(32, CCC, addt, transpt) \
		CALL(33, CCC, addt, transpt) \
		CALL(34, CCC, addt, transpt) \
		CALL(41, CCC, addt, transpt) \
		CALL(48, CCC, addt, transpt) \
		CALL(63, CCC, addt, transpt) \
		CALL(64, CCC, addt, transpt) \
		CALL(65, CCC, addt, transpt) \
		CALL(81, CCC, addt, transpt) \
			hasInst = false;
	AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "SparseScatterGPU", ([&] {
		LaunchParams<scalar_t> lp(C, bSzH, bSzW, numActive);
		if (transpose && !add) {
			if (lp.fittingC1 >= 32) {
				SIZE_TEMPLATES(false, true, 32)
			} else if (lp.fittingC1 == 16) {
				SIZE_TEMPLATES(false, true, 16)
			} else if (lp.fittingC1 == 24) {
				SIZE_TEMPLATES(false, true, 24)
			}
		} else if (transpose && add) {
			if (lp.fittingC1 >= 32) {
				SIZE_TEMPLATES(true, true, 32)
			} else if (lp.fittingC1 == 16) {
				SIZE_TEMPLATES(true, true, 16)
			} else if (lp.fittingC1 == 24) {
				SIZE_TEMPLATES(true, true, 24)
			}
		} else if (!transpose && !add) {
			if (lp.fittingC1 >= 32) {
				SIZE_TEMPLATES(false, false, 32)
			} else if (lp.fittingC1 == 16) {
				SIZE_TEMPLATES(false, false, 16)
			} else if (lp.fittingC1 == 24) {
				SIZE_TEMPLATES(false, false, 24)
			}
		} else {
			if (lp.fittingC1 >= 32) {
				SIZE_TEMPLATES(true, false, 32)
			} else if (lp.fittingC1 == 16) {
				SIZE_TEMPLATES(true, false, 16)
			} else if (lp.fittingC1 == 24) {
				SIZE_TEMPLATES(true, false, 24)
			}
		}
	#endif
		if (!hasInst) {
			//printf("scatter, C, bSzH, bSzW=%d, %d, %d, fittingC1=%d\n", C, bSzH, bSzW, lp.fittingC1);
			blockScatterTiled1<scalar_t, 512><<<lp.grid, lp.block, lp.shmemSize, stream>>>(
				x.data_ptr<scalar_t>(), (const int16_t*)activeBlockIndices.data_ptr<int16_t>(),
				y.data_ptr<scalar_t>(), N, H, W, C, bOffsH0, bOffsW0, bStrH, bStrW,
				bSzH, lp.bSzH1, bSzW, lp.fittingC1, add, transpose, atomic);
		}
	}));
	#undef SIZE_TEMPLATES
	#undef CALL
	gpuErrorCheck( cudaPeekAtLastError() );

	return y;
}

//template<typename T> struct CopyTensorFunctor<GPUDevice, T> {
//    void operator()(const GPUDevice& gpu, T* dst, const T* src, int count) {
//        cudaMemcpyAsync(dst, src, sizeof(T)*count, cudaMemcpyDeviceToDevice, gpu.stream());
//        gpuErrorCheck( cudaPeekAtLastError() );
//        cudaStreamSynchronize(gpu.stream());
//        gpuErrorCheck( cudaPeekAtLastError() );
//    }
//    const cudaStream_t* getStream(const GPUDevice& gpu) { return &gpu.stream(); }
//};
