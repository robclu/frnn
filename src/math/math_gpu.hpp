/*
 *  Header file for fastRNN math C++ interfaces for executing the GPU 
 *  instances.
 *
 *  Copyright (C) 2015 Rob Clucas robclu1818@gmail.com
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published
 *  by the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation,
 *  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _FRNN_MATH_GPU_
#define _FRNN_MATH_GPU_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <vector>
#include <random>
#include <omp.h>

#include "../tensor/tensor.cuh"
#include "../util/errors.h"
#include "../frnn/frnn.h"
#include "math_kernels_gpu.cuh"
#include "blas/frnn_blas.h"
#include "rand/frnn_rand.h"

/*
 * ==========================================================================================================
 * Function     : axpyGpu
 *
 * Description  : Performs simgle/double precision a*X + Y, using CUBLAS
 *
 * Inputs       : error     : fastRNN error type for result of operations
 *              : a         : Constant for multiplication 
 *              : x         : Vector to multiply with a
 * 
 * Outputs      : y         : Vector used in a*X + Y, and where the result of a*X + Y is stored
 * 
 * Params       : dType     : The type of data used for the computation
 * ==========================================================================================================
 */
template <typename dType>
void axpyGpu( frnn::frnnError& error, const dType a, const std::vector<dType>& x, std::vector<dType>& y ) {

    cublasHandle_t handle;
    cublasStatus_t status;
    dType* da = 0, *dx = 0, *dy = 0;

    status = cublasCreate( &handle );
    cublasSetPointerMode( handle, CUBLAS_POINTER_MODE_DEVICE );

    // Allocate and fill device vectors with host vector data 
    if ( cudaMalloc( (void**)&da, sizeof( dType ) ) != cudaSuccess ) {
        frnn::err::allocError( error, stringify( da ) );
    }
    if ( cudaMalloc( (void**)&dx, x.size() * sizeof( dType ) ) != cudaSuccess ) {
        frnn::err::allocError( error, stringify( dx ) );   
    }
    if ( cudaMalloc( (void**)&dy, y.size() * sizeof( dType ) ) != cudaSuccess ) {
        frnn::err::allocError( error, stringify( dy ) );   
    }

    // Fill device vectors with data
    if ( cudaMemcpy( da, &a, sizeof( dType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
        frnn::err::copyError( error, stringify( da ) );
    }
    if ( cudaMemcpy( dx, &x[0], x.size() * sizeof( dType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
        frnn::err::copyError( error, stringify( da ) );
    }
    if ( cudaMemcpy( dy, &y[0], y.size() * sizeof( dType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
        frnn::err::copyError( error, stringify( da ) );
    }

    // Perform CUBLAS axpy using wrapper blas library
    status = frnn::blas::functions<dType>::axpy( handle, x.size(), da, dx, 1, dy, 1 );

    if ( cudaMemcpy( &y[0], dy, y.size() * sizeof( dType ), cudaMemcpyDeviceToHost ) != cudaSuccess ) {
        frnn::err::copyError( error, stringify( y ) );
    }

    status = cublasDestroy( handle );

    cudaFree( da );
    cudaFree( dx );
    cudaFree( dy );
} 

/*
 * ==========================================================================================================
 * Function     : axpyGpu (for ints)
 *
 * Description  : Performs a*X + Y, using own implementation for ints
 *
 * Inputs       : error     : fastRNN error type for result of operations
 *              : a         : Constant for multiplication 
 *              : x         : Vector to multiply with a
 * 
 * Outputs      : y         : Vector used in a*X + Y, and where the result of a*X + Y is stored
 * ==========================================================================================================
 */
void axpyGpu( frnn::frnnError& error, const int a, const std::vector<int>& x, std::vector<int>& y ) {
    // Still implement - placeholder for now
}

/*
 * ==========================================================================================================
 * Function     : randGpu 
 * 
 * Descrition   : Generates a random number between 2 limits on the GPU
 * 
 * Inputs       : x         : The array that must be filled with random numbers
 *              : N         : The number of elements in the array x to fille with random numbers
 *              : lo        : The lower bound for each random number
 *              : hi        : The upper bound for each random number 
 *              
 * Oututs       : The array of random numbers on the range lo - hi
 * 
 * Params       : dType     : The type of data of the output element
 * ==========================================================================================================
 */ 
template <typename dType>
void randGpu( dType* x, size_t N, dType lo, dType hi ) {
    
    curandGenerator_t   gen;                    // Device random number generator
    frnnError          error;
    dType*              device_randoms = 0;
    
    // Allocate memory on the device
    if ( cudaMalloc( (void**)&device_randoms, N * sizeof( dType) ) != cudaSuccess ) {
        frnn::err::allocError( error, stringify( device_randoms ) );
    }
   
    curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT );                  // Create generator
    curandSetPseudoRandomGeneratorSeed( gen, 1234ULL );                        // Seed RNG
    frnn::rng::generators<dType>::uniform( gen, device_randoms, N );          // Create rand nums on GPU
    
    int threads, blocks;
    threads = ( ( N < 1024 ) ? N : 1024 );
    blocks  = std::min( static_cast<int>( N / threads + 1 ), MAX_BLOCKS );
    

    // Call kernel to chnage the range from 0 - 1 to lo - hi
    scale<<<blocks, threads>>>( device_randoms, N, lo, hi );
    
    // Copy results from deviec mem to x
    if ( cudaMemcpy( x, device_randoms, N * sizeof( dType ), cudaMemcpyDeviceToHost ) != cudaSuccess ) {
        frnn::err::copyError( error, stringify( device_randoms ) );
    }
    
    // Clean up
    cudaFree ( device_randoms );
    curandDestroyGenerator( gen );
}    
    
/*
 * ==========================================================================================================
 * Function     : softmaxGpu
 *
 * Description  : Performs the softmax function of a vector of data x, which is 
 *                  
 *                softmax( x_i ) = exp( x_i ) / sum[ j=1 to J ]( exp( x_j )
 *
 * Inputs       : status    : Cublas status for determining correct completion of operation
 *        
 * Outputs      : x         : Vector to compute the softmax of, and to store the result in
 *
 * Params       : dType     : The type of data (float or int) double not supported due to lack of
 *                            support for doubles in some Nvidia kernel functions
 * ==========================================================================================================
 */ 

template <typename dType>
void softmaxGpu( frnn::frnnError& error, const std::vector<dType>& x, std::vector<dType>& val ) {

    dType* in = 0, *out = 0;
    frnn::functors::exp exp_op;           // Define operation on each element to be exponentiation

    // Check output vector can hold all reasults
    if ( val.size() < x.size() ) val.resize( x.size(), 0 );
    
    // Alllocate memory on the device
    if ( cudaMalloc( (void**)&in, x.size() * sizeof( dType ) ) != cudaSuccess ) {
        frnn::err::allocError( error, stringify( in ) );
    }
    if ( cudaMalloc( (void**)&out, x.size() * sizeof( dType) ) != cudaSuccess ) {
        frnn::err::allocError( error, stringify( out ) );
    }
    
    // Copy data from x to in
    if ( cudaMemcpy( in, &x[0], x.size() * sizeof( dType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
        frnn::err::copyError( error, stringify( in ) );
    }
    if ( cudaMemset( out, 0, x.size() * sizeof( dType ) ) != cudaSuccess ) {
        frnn::err::copyError( error, stringify( out ) );
    }
    
    // Determine the size of the grids for the kernel, we need enough blocks
    // to make sure that each element of the output vector gets a result
    int threads;
    x.size() > 256 * MAX_BLOCKS ? threads = 512 : threads = 256;
    
    int blocks  = std::min( static_cast<int>( x.size() / threads ), MAX_BLOCKS );
    if (  blocks * threads < x.size() ) blocks++;

    // Execute kernel to reduce all blocks, using the exp functor to
    // exponentiate each element before addition
    blockReduceAtomicVectorizedAll<<<blocks, threads>>>( in, out, x.size(), exp_op );
    // Copy result from the first thread inea ch block to the others
    blockScatter<<<blocks, threads>>>( out, x.size() );
    // Do normalization to get get softmax
    softmaxKernel<<<blocks, threads>>>( in, out, x.size() );        

    if ( cudaMemcpy( &val[0], out, x.size() * sizeof( dType ), cudaMemcpyDeviceToHost ) != cudaSuccess ) {
        frnn::err::copyError( error, stringify( val ) );
    }

    cudaFree( in ); cudaFree( out );
}

/*
 * ==========================================================================================================
 * Function     : sumGpu
 *
 * Description  : Performs the sum of the elements in a vector
 *                  
 * Inputs       : error     : fastRNN error type for results of opsofterations
 *              : x         : The vector, araary etc.. (data) to comupte the sum of
 *        
 * Outputs      : val       : The result of the sum of the array 
 *
 * Params       : dType     : The data type of the array elements
 * ==========================================================================================================
 */  
template <typename dType>
dType sumGpu( frnn::frnnError& error, const std::vector<dType>& x ) {

    dType* in = 0, *out = 0, val = 0;

    // Alllocate memory on the device
    if ( cudaMalloc( (void**)&in, x.size() * sizeof( dType ) ) != cudaSuccess ) {
        frnn::err::allocError( error, stringify( in ) );
    }
    if ( cudaMalloc( (void**)&out, sizeof( dType) ) != cudaSuccess ) {
        frnn::err::allocError( error, stringify( out ) );
    }

    // Copy data from x to in
    if ( cudaMemcpy( in, &x[0], x.size() * sizeof( dType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
        frnn::err::copyError( error, stringify( in ) );
    }
    // Set out to 0 on the device
    if ( cudaMemsetAsync( out, 0, sizeof( dType) ) != cudaSuccess ) {
        frnn::err::copyError( error, stringify( out ) );
    }

    // 256 threads per block is optimal, however, if this isn't enough use more
    int threads;
    x.size() > 256 * MAX_BLOCKS ? threads = 512 : threads = 256;
    int blocks  = std::min( static_cast<int>( ( ( x.size() / 2 ) + threads - 1 ) / threads ), MAX_BLOCKS );

    blockReduceAtomicVectorized<<<blocks, threads>>>( in, out, x.size() );

    if ( cudaMemcpy( &val, out, sizeof( dType ), cudaMemcpyDeviceToHost ) != cudaSuccess ) {
        frnn::err::copyError( error, stringify( out ) );
    }

    cudaFree( in ); cudaFree( out );
    return val;
}

/*
 * ==========================================================================================================
 * Function     : sumVectorizedGpu
 *
 * Description  : Performs the sum of the elements in a vector and returns a vector of the same
 *                dimension with each element having the result
 *                  
 * Inputs       : error     : fastRNN error type for results of operations
 *              : x         : The vector, araary etc.. (data) to comupte the sum of
 *        
 * Outputs      : val       : A vector where each element holds the result of the sum
 *
 * Params       : dType     : The data type of the array elements
 * ==========================================================================================================
 */  
template <typename dType>
void sumVectorizedGpu( frnnError& error, const std::vector<dType>& x, std::vector<dType>& val ) {

    dType* in = 0, *out = 0;
    
    // Check output vector can hold results
    if ( val.capacity() < x.size() ) val.reserve( x.size() );

    // Alllocate memory on the device
    if ( cudaMalloc( (void**)&in, x.size() * sizeof( dType ) ) != cudaSuccess ) {
        frnn::err::allocError( error, stringify( in ) );
    }
    if ( cudaMalloc( (void**)&out, x.size() * sizeof( dType) ) != cudaSuccess ) {
        frnn::err::allocError( error, stringify( out ) );
    }
    
    // Copy data from x to in
    if ( cudaMemcpy( in, &x[0], x.size() * sizeof( dType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
        frnn::err::copyError( error, stringify( in ) );
    }
    if ( cudaMemset( out, 0, x.size() * sizeof( dType ) ) != cudaSuccess ) {
        frnn::err::copyError( error, stringify( out ) );
    }
    
    // Determine the size of the grids for the kernel, we need enough blocks
    // to make sure that each element of the output vector gets a result
    int threads;
    x.size() > 256 * MAX_BLOCKS ? threads = 512 : threads = 256;
    int blocks  = std::min( static_cast<int>( x.size() / threads ), MAX_BLOCKS );
    if (  blocks * threads < x.size() ) blocks++;

    blockReduceAtomicVectorizedAll<<<blocks, threads>>>( in, out, x.size() );
    // Copy result from first thread in each block to the others
    blockScatter<<<blocks, threads>>>( out, x.size() );         

    if ( cudaMemcpy( &val[0], out, x.size() * sizeof( dType ), cudaMemcpyDeviceToHost ) != cudaSuccess ) {
        frnn::err::copyError( error, stringify( val ) );
    }

    cudaFree( in ); cudaFree( out );
}

#endif
