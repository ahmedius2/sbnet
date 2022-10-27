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

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <ATen/cuda/CUDAContext.h>
#include "cuda_helpers.h"
#include "op_utils.h"
#include "zero_block_counters.cu.h"
//#include "reduce_mask_cuda.h"

using namespace ST;

//
// Mask is NHW1
// This tensor can be quite small, say highest res is (64,1920,1200,1)
// Could be as small as (64,32,32,1) or even (64,7,7,1) for last ImageNet layer
//
// One possible work partition strategy assuming larger batch:
// wrap HW around tNTHREADS, run N blocks, one block per batch,
// reduce the count inside each block
// use atomicAdd to reduce the total number of blocks
// 
// For small batch inference it's going to be better to have a HW-blocked kernel.
// This works for, say, 1x1920x1200x1 block size 
// Sometimes it's going to be difficult to utilize the GPU.
// For instance how do we partition a 1x7x7 with block size 1?
// 
// Perhaps we can do N*bCntH*bCntW blocks and wrap the threads around block pixels?
// there's going to be some duplication in reads/BW waste but the inputs should be small anyway
// N*BCH*BCW blocks kernel: blockIdx.x=[0, N)
// tNTHREADS is tHb*tWb
//
// blockDim.x = tbH*tbW
// gridDim = (x=bCntW, y=bCntH, z=N)
// So basically run a CUDA block per sparsity block
// threadIdx.x = intra-block w+h*W, rounded up to 32 (warpLanes)
//
template<typename scalar_t, typename tbH, typename tbW>
__device__ void reduceMask_t(
		const int nActualThreads,                 // tbH*tbW rounded up to warpSize
		const scalar_t* mask, int N, int H, int W,   // C is assumed to be 1
		const scalar_t threshold,                    // value to consider non-sparse block
		//int* reducedMask,                       // space for resulting binary max>threshold mask per sparsity block
		unsigned int  numBins,                    // number of bins to partition activeBlockIndices to reduce atomics pressure
		unsigned int  binSize,
		int32_t* binCounts,                           // counts for sub-blocks, initialized to 0
		int16_t* activeBlockIndices,                // result
		const int bOffsH0, const int bOffsW0,     // generally negative - first block element offset for correct padding
		const int bStrH, const int bStrW,         // block strides
		const int bCntH, const int bCntW,         // block counts
		tbH tbHArg, tbW tbWArg, bool avgPooling)  // do maxpool if avgPooling is false
{
	const int bH = tbHArg.get(), bW = tbWArg.get();

	int blockW0 = bOffsW0 + bStrW*blockIdx.x;
	int blockH0 = bOffsH0 + bStrH*blockIdx.y;
	int n = blockIdx.z;
	// one thread per sparsity block pixel
	const int roundedUpThreads = DIVUP(bH*bW, warpLanes)*warpLanes;
	scalar_t mx = avgPooling ? 0.0f : -1e30;
	// allocate and initialize shmem for block reduce
	constexpr int maxBlockDim = 1024;
	assert(blockDim.x <= maxBlockDim);
	__shared__ scalar_t shmemx[maxBlockDim];
	for (int initOffs = 0; initOffs < maxBlockDim; initOffs += blockDim.x)
		if (initOffs + threadIdx.x < maxBlockDim)
			shmemx[initOffs+threadIdx.x] = avgPooling ? 0.0f : -1e30f;
	__syncthreads();

	// for large sparsity blocks we need multiple CUDA block loops
	for (int tOffs = 0; tOffs < roundedUpThreads; tOffs+=blockDim.x)
	{
		int tid = threadIdx.x + tOffs;
		const scalar_t* blockStartN = mask + n*H*W;
		scalar_t readVal = avgPooling ? 0.0f : -1e30f; // this value will be used to pad the warp
		if (tid < bH*bW) { // TODO: not needed?
			int woffs = tid % bW;
			int hoffs = tid / bW;
			unsigned bhh = hoffs+blockH0, bww = woffs + blockW0;
			if (bhh < H && bww < W)
				readVal = blockStartN[bhh*W + bww];
		}

		// actual number of threads is rounded up to 32 but padded with zeroes
		// warp reduce for all threads
		mx = avgPooling ? (mx + readVal) : max(mx, readVal);
#pragma unroll
		for (int offset = warpLanes/2; offset > 0; offset /= 2) {
			//float warped = __shfl_down(mx, offset);
			scalar_t warped = __shfl_down_sync(0xffffffff, mx, offset);
			mx = avgPooling ? (mx + warped) : max(mx, warped);         
		}

		// store (first elems from) warp reduces into shmem
		if (tid % warpLanes == 0) {
			int offs = tid/warpLanes; // tid includes tOffs
			int offsWrap = offs%blockDim.x;
			if (avgPooling)
				// atomics not needed here since we wrap around each blockDim.x
				shmemx[offsWrap] += mx;
			else
				shmemx[offsWrap] = max(shmemx[offsWrap], mx);
		}
		__syncthreads();
	} // tOffs

	// final reduce over all warps
	if (threadIdx.x == 0) {
		scalar_t mx1 = shmemx[0];
		// For sizes >= blockIdx.x we already reduced in the above loop
		const int numWarps = min(DIVUP(bH*bW, warpLanes), blockDim.x);
#pragma unroll
		for (int iWarp = 1; iWarp < numWarps; iWarp++)
			mx1 = avgPooling ? (mx1 + shmemx[iWarp]) : max(mx1, shmemx[iWarp]);

		if (avgPooling)
			mx1 /= scalar_t(bH*bW);

		if (mx1 > threshold) {
			// now we have the maximums computed for each block
			// we need to write out the maximums, total over-threshold count across grid
			// at this point the number of blocks is grid size, so N*bCntH*bCntW
			// bad case scenario is say 4*64*64 (larger batch won't fit into memory)
			// so we can have ~16k blocks
			// need an efficient gmem reduction
			unsigned int blockIndex = n*bH*bW + blockIdx.y*bW + blockIdx.x;
			unsigned int myBin = ((blockIndex*100017+1234567)>>4) % numBins;
			unsigned int inBinOffs;
			// check for bin overflow
			while ((inBinOffs = atomicAdd(&binCounts[myBin], int(1))) >= binSize)
			{
				atomicSub(&binCounts[myBin], int(1));
				myBin++;
			}

			int offs = (myBin*binSize+inBinOffs)*3;
			activeBlockIndices[offs+0] = blockIdx.z;
			activeBlockIndices[offs+1] = blockIdx.y;
			activeBlockIndices[offs+2] = blockIdx.x;
		} // if (mx1 > threshold)
	} // if (tid == 0)
}

//extern "C" {
// kernel entry point
template <typename scalar_t>
__global__ void reduceMask(
		const scalar_t* mask, int N, int H, int W,   // C is assumed to be 1
		const scalar_t threshold,                    // value to consider non-sparse block
		unsigned int  numBins,                    // number of bins to partition activeBlockIndices to reduce atomics pressure
		unsigned int  binSize,
		int32_t* binCounts,                           // counts for sub-blocks, initialized to 0
		int16_t* activeBlockIndices,                // result: block indices split into bins, currently assuming even sized bins.
		// the counter will spill into the next bin on overflow.
		// It is expected that activeBlockIndices is allocated enough memory for worst case
		// number of active indices, ie all blocks are active
		const int bOffsH0, const int bOffsW0,     // generally negative - first block element offset for correct padding
		const int bSzH, const int bSzW,           // block sizes
		const int bStrH, const int bStrW,         // block strides
		const int bCntH, const int bCntW,         // block counts
		bool avgPool)
{
	const int roundUpThreads = DIVUP(bSzH*bSzW, warpLanes)*warpLanes;
	assert((roundUpThreads == blockDim.x || roundUpThreads > 1024) &&
			"Error in reduceMask_t: blockDim.x must be a multiple of warpLanes\n");
	assert(blockDim.y == 1 && blockDim.z == 1 &&
			"Expected block shape=(x=nThreads, y=1, z=1)");

	reduceMask_t<scalar_t, getVar, getVar>(
			roundUpThreads, mask, N, H, W,
			threshold, numBins, binSize, binCounts,
			activeBlockIndices, bOffsH0, bOffsW0,
			bStrH, bStrW, bCntH, bCntW, getVar(bSzH), getVar(bSzW), avgPool);
}

//}

// Define the GPU implementation that launches the CUDA kernel.
//template <typename scalar_t>
//void ReduceMaskGPU(
//	const scalar_t* mask,           // Mask array.
//    int N,                          // Batch dimension of the mask.
//    int H,                          // Height of the mask.
//    int W,                          // Width of the mask.
//    float threshold,                // Threshold for being active.
//    int bOffsH0,                    // Block padding offset height, negative.
//    int bOffsW0,                    // Block padding offset width, negative.
//    int bSzH,                       // Block size height.
//    int bSzW,                       // Block size width.
//    int bStrH,                      // Block stride, height.
//    int bStrW,                      // Block stride, width.
//    int bCntH,                      // Number of blocks, height.
//    int bCntW,                      // Number of blocks, width.
//    unsigned int numBins,           // number of bins in binCounts
//    unsigned int binSize,           // maximum size of each counter bin
//    int16_t* activeBlockIndices,      // triples of [n, ih, iw] indices for active blocks.
//    int32_t* binCounts,                 // Number of indices of active blocks.
//    bool avgPool                    // true for avg pooling, false for max pooling
//    )
void LaunchReduceMaskGPU(
		torch::Tensor mask,               // Mask array.
		int N,                            // Batch dimension of the mask.
		int H,                            // Height of the mask.
		int W,                            // Width of the mask.
		float threshold,                  // Threshold for being active.
		int bOffsH0,                      // Block padding offset height, negative.
		int bOffsW0,                      // Block padding offset width, negative.
		int bSzH,                         // Block size height.
		int bSzW,                         // Block size width.
		int bStrH,                        // Block stride, height.
		int bStrW,                        // Block stride, width.
		int bCntH,                        // Number of blocks, height.
		int bCntW,                        // Number of blocks, width.
		unsigned int numBins,             // number of bins in binCounts
		unsigned int binSize,             // maximum size of each counter bin
		torch::Tensor activeBlockIndices, // triples of [n, ih, iw] indices for active blocks.
		torch::Tensor binCounts,          // Number of indices of active blocks.
		bool avgPool                      // true for avg pooling, false for max pooling
		)
{
	gpuErrorCheck( cudaPeekAtLastError() );

	// TODO
	// We can do better here in terms of grid/block partitioning but this is not currently a perf bottleneck
	//printf("++++++++++++++++++++++++++++++ Launching ZBC, binCounts=%x\n", binCounts);
	//cudaStream_t stream = d.stream();
	cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
	gpuErrorCheck( cudaPeekAtLastError() );

	AT_DISPATCH_ALL_TYPES(mask.scalar_type(), "ReduceMaskGPU", ([&] {
		zeroBlockCounters<<<1, 32, 0, stream>>>(numBins, binCounts.data_ptr<int32_t>());
		gpuErrorCheck( cudaPeekAtLastError() );

		dim3 block(std::min(DIVUP(bSzH*bSzW, 32)*32, 1024), 1, 1);
		dim3 grid(bCntW, bCntH, N);
		reduceMask<<<grid, block, 0, stream>>>(mask.data_ptr<scalar_t>(),
				N, H, W, // C is assumed to be 1
				(scalar_t) threshold, // value to consider non-sparse block
				numBins,   // number of bins to partition activeBlockIndices to reduce atomics pressure
				binSize,
				binCounts.data_ptr<int32_t>(), // counts for sub-blocks, initialized to 0
				activeBlockIndices.data_ptr<int16_t>(),
				bOffsH0,
				bOffsW0,      // generally negative - first block element offset for correct padding
				bSzH, bSzW,   // block sizes
				bStrH, bStrW, // block strides
				bCntH, bCntW, // block counts
				avgPool);
	}));

	gpuErrorCheck( cudaPeekAtLastError() );
}
