#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include "timer.h"
#include <string>


//////////// HELPER FCNS AND KERNELS //////////////////

#define getDigit(val, start, radix) ((val>>start) & (radix - 1))
#define ITEMSPERTHREAD 4

// basically memset, but for unsigned int's only
__device__ void k_assignRange(unsigned int * start, unsigned int value, unsigned int numElems)
{
    for (int i = 0; i < numElems; i++)
      *(start + i) = value;    
}

// basically memcpy, but for unsigned int's only
__device__ void k_copyRange(unsigned int * output, unsigned int * input, unsigned int numElems)
{
    for (int i = 0; i < numElems; i++)
      output[i] = input[i];    
}


// performs a prefix sum, where each thread has the value pred
__device__ unsigned int scan(unsigned int pred, unsigned int * sh_sums)
{
    sh_sums[threadIdx.x] = pred;
    __syncthreads();
    
    for (int skip = 1; skip < blockDim.x; skip *= 2)
    {
        int newValue = (threadIdx.x >= skip) ? sh_sums[threadIdx.x] + sh_sums[threadIdx.x - skip] : sh_sums[threadIdx.x];
        syncthreads();
        sh_sums[threadIdx.x] = newValue;
        syncthreads();
    }    
    if (threadIdx.x > 0)
        return sh_sums[threadIdx.x - 1];
    else
        return 0;
}



// like above, where each thread handles multiple values and outputs several different values in the prefix sum
// itemsThisThread will equal itesmPerThread except if this block has threads that have no items to work with
__device__ void scanMultiple(unsigned int * outputSums, 
                               unsigned int * inputVals,
                               unsigned int localID, // first value in sh_sums[] this thread deals with
                               unsigned int numElems,  
                               unsigned int * sh_sums, // shared memory for computing the sums
                               unsigned int itemsThisThread) // # of items this thread works with
{
    k_copyRange(sh_sums + localID, inputVals, itemsThisThread);
    __syncthreads();
    
    unsigned int newValues[ITEMSPERTHREAD];

    for (int skip = 1; skip < numElems; skip *= 2) 
    {
        for (int i = 0; i < itemsThisThread; i++)
        {
            if (localID + i >= skip)
                newValues[i] = sh_sums[localID + i] + sh_sums[localID + i - skip];
            else
                newValues[i] = sh_sums[localID + i];
        }
        __syncthreads();
        k_copyRange(sh_sums + localID, newValues, itemsThisThread); 
        __syncthreads();  
    }
    
    // write output
    if (threadIdx.x > 0)
        outputSums[0] = sh_sums[localID - 1];
    else
        outputSums[0] = 0;  
    k_copyRange(outputSums + 1, sh_sums + localID, itemsThisThread - 1);  
}


// outputs the "rank" of each item, for partitioning the block by predicate value
__device__ unsigned int split(bool pred, unsigned int blocksize, unsigned int * sh_sums)
{
    unsigned int true_before = scan(pred, sh_sums);
    __shared__ unsigned int false_total;    
    if(threadIdx.x == blocksize - 1)
        false_total = blocksize - (true_before + pred);
    __syncthreads();  
    if(pred) 
        return true_before + false_total;
    else 
        return threadIdx.x - true_before; 
}

// single-block radix sort
__global__ void k_radixSortBlock(unsigned int * d_vals,
                                 unsigned int * d_pos,
                                 unsigned int startBit,
                                 unsigned int radix,
                                 unsigned int numElems)
{
    int inputID = threadIdx.x + blockDim.x * blockIdx.x;
    if (inputID < numElems)
    {
		extern __shared__ unsigned int sh_arr[];
		unsigned int * sh_sums = sh_arr;
		unsigned int * sh_vals = sh_arr + blockDim.x;
		unsigned int * sh_pos = sh_arr + blockDim.x * 2;
		sh_vals[threadIdx.x] = d_vals[inputID];
		sh_pos[threadIdx.x] = d_pos[inputID];
		__syncthreads();
		
		for (int d = 1; d < radix; d <<= 1)
		{
			unsigned int i = split(((sh_vals[threadIdx.x]>>startBit) & d) > 0, min(numElems - blockDim.x * blockIdx.x, blockDim.x), sh_sums);
			unsigned int oldValue = sh_vals[threadIdx.x];
			unsigned int oldPos = sh_pos[threadIdx.x];
			__syncthreads();
			sh_vals[i] = oldValue;
			sh_pos[i] = oldPos;
			__syncthreads();
		}  
		d_vals[blockDim.x * blockIdx.x + threadIdx.x] = sh_vals[threadIdx.x];
		d_pos[blockDim.x * blockIdx.x + threadIdx.x] = sh_pos[threadIdx.x];
	}
}




// 1st step to calculating  globalOffsets (offsets for each block and each radix)
    // this step simply counts the # of each radix value in each block, later we do a scan on that to get globalOffsets
// as well as calculating localOffsets (offsets within each block for each radix)
// d_bucketSize[i][j] is the # of elements of radix i that are in block j
// send only enough threads to look at blockSize - 1 items (since each thread compares to the next item in the block)
__global__ void k_findOffsets(unsigned int * d_globalOffsets,
                              unsigned int * d_localOffsets,
                              unsigned int * d_vals,
                              unsigned int startBit,
                              unsigned int radix,
                              unsigned int numElems)
{

    extern __shared__ unsigned int sh_arr[]; // for storing offsets
    unsigned int inputID = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int blockSize = min(numElems - blockDim.x * blockIdx.x, blockDim.x);
    int thisDigit = getDigit(d_vals[inputID], startBit, radix);
    
    if (inputID < numElems)
	{
		// missing radix values before the first item?  1st thread gives their offset as 0
		if (threadIdx.x == 0)
			k_assignRange(sh_arr, 0, thisDigit + 1);
		
		// missing radix values after the last item?  last thread gives their offset as blockSize
		if (threadIdx.x == blockSize - 1)
			k_assignRange(sh_arr + thisDigit + 1, blockSize, radix - 1 - thisDigit);
		else
		{
			int nextDigit = getDigit(d_vals[inputID + 1], startBit, radix);    		
			// assign offsets for all value(s) between this digit and the next digit, including the next digit but not this one 
			if (nextDigit > thisDigit)
				k_assignRange(sh_arr + thisDigit + 1, threadIdx.x + 1, nextDigit - thisDigit);
		}

		__syncthreads();
		
		// index both output arrays in bucket-major order
		unsigned int outputID = blockIdx.x + gridDim.x  * threadIdx.x;
		
		if (threadIdx.x < radix - 1)
		{
			d_localOffsets[outputID] = sh_arr[threadIdx.x];
			d_globalOffsets[outputID] = sh_arr[threadIdx.x + 1] - sh_arr[threadIdx.x];
		}
		else if (threadIdx.x == radix - 1)
		{
			d_localOffsets[outputID] = sh_arr[threadIdx.x];
			d_globalOffsets[outputID] = blockSize - sh_arr[threadIdx.x];
		}   
	}
}

__global__ void k_scan(unsigned int * d_vals,
                               unsigned int numElems)
{
    unsigned int localID = threadIdx.x * ITEMSPERTHREAD;
    unsigned int inputID = blockDim.x * blockIdx.x + localID;
    unsigned int itemsThisThread = min(numElems - localID, ITEMSPERTHREAD);   
    if (inputID >= numElems)
        return;   
    extern __shared__ unsigned int sh_arr[];
    unsigned int outputSums[ITEMSPERTHREAD];   
    scanMultiple(outputSums, d_vals + inputID, localID, numElems, sh_arr, itemsThisThread);
    __syncthreads();   
    k_copyRange(d_vals + inputID, outputSums, itemsThisThread);   
}



__global__ void k_scatter(unsigned int * d_outputVals, 
                          unsigned int * d_inputVals,
                          unsigned int * d_outputPos,
                          unsigned int * d_inputPos,
                          unsigned int * d_localOffsets, 
                          unsigned int * d_globalOffsets, 
                          int startBit,
                          int radix,
                          unsigned int numElems)
{
    unsigned int inputID = blockDim.x * blockIdx.x + threadIdx.x;
    if (inputID >= numElems)
        return;
    
    int thisDigit = getDigit(d_inputVals[inputID], startBit, radix);
    unsigned int offsetIndex = gridDim.x * thisDigit + blockIdx.x;
    unsigned int outputID = threadIdx.x - d_localOffsets[offsetIndex] + d_globalOffsets[offsetIndex];
    d_outputVals[outputID] = d_inputVals[inputID];
    d_outputPos[outputID] = d_inputPos[inputID];
}

    
// swap pointers
void exch(unsigned int * * a, unsigned int *  * b)
{
    unsigned int * temp = *a;
    *a = *b;
    *b = temp;    
}



// based off of this paper: http://mgarland.org/files/papers/nvr-2008-001.pdf
void your_sort (unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    
    // PREFERENCES
    const unsigned int blockSize = 512;   
    const int numBits = 3;
    const unsigned int radix = pow(2,numBits); 
    unsigned int numBlocks = (numElems - 1) / blockSize + 1;
    assert((radix * numBlocks - 1) / ITEMSPERTHREAD + 1 < 1024);
    
    unsigned int * d_globalOffsets, * d_localOffsets;
    checkCudaErrors(cudaMalloc(&d_globalOffsets, radix * numBlocks * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&d_localOffsets, radix * numBlocks * sizeof(unsigned int)));
    
    unsigned int * d_valuesA = d_inputVals; 
    unsigned int * d_valuesB = d_outputVals;
    unsigned int * d_posB = d_outputPos;
    unsigned int * d_posA = d_inputPos;
    
    for (int d = 0; d < 32; d += numBits)
    {
        k_radixSortBlock<<<numBlocks, blockSize, 3 * blockSize*sizeof(unsigned int)>>>(d_valuesA, d_posA, d, radix, numElems);
        k_findOffsets<<<numBlocks, blockSize, (radix + 1)*sizeof(unsigned int)>>>(d_globalOffsets, d_localOffsets, d_valuesA, d, radix, numElems);
        k_scan<<<1, (radix * numBlocks - 1) / ITEMSPERTHREAD + 1, radix * numBlocks * sizeof(unsigned int)>>>(d_globalOffsets,numElems);
        k_scatter<<<numBlocks, blockSize>>>(d_valuesB, d_valuesA, d_posB, d_posA, d_localOffsets, d_globalOffsets, d, radix, numElems); 
        exch(&d_valuesA, &d_valuesB);
        exch(&d_posA, &d_posB);
    }
    checkCudaErrors(cudaMemcpy(d_outputPos, d_posA, numElems *sizeof(unsigned int), cudaMemcpyDeviceToDevice)); 
}
