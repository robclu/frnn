/*
 *  Header file for cuRNN softmax gpu kernels.
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

#ifndef _CURNN_SOFTMAX_KERNELS_GPU_
#define _CURNN_SOFTMAX_KERNELS_GPU_

#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../../tensor/tensor.cuh"
#include "../../util/errors.h"
#include "../../curnn/curnn.h"
#include "../../math/blas/curnn_blas.h"

namespace curnn{
    
template <typename dType>
void softmax_forward_gpu( std::vector<dType>& ins       , 
                          tensor4<dType>&     wba       ,
                          uint                numInputs ,
                          std::vector<dType>& outs      ) {         
   
    // Thread sizes
    size_t threadsX, threadsY, blocksX, blocksY;

    // Statuses
    curnnError      error;
    cublasHandle_t  handle;
    cublasStatus_t  status;

    status = cublasCreate( &handle );
    cublasSetPointerMode( handle, CUBLAS_POINTER_MODE_HOST );

    // Device pointers : For each page (wba.z) we need a pointer for :
    //      inputs, weights, biases
    std::vector<dType*> dPointers( 3 * wba.z(), 0 );
    dType*              results_h[ wba.z() ];           // Pointers to results of W*x + b on host
    dType**             results_d;                      // Pointers to results of W*x + b on device
    functors::exp       expOp;                          // Exp operation for softmax

    // Outputs vector must have same dimension as number of nodes
    if ( outs.size() < wba.x() ) outs.resize( wba.x(), 0 );
    // Inputs vector must have same dimension as weight metrix num inputs
    if ( ins.size() != numInputs ) {
        curnn::err::dimError( error, stringify( ins ), stringify( numInputs ) );
        return;
    }

    // Each page is done by a separate kernel
    #pragma omp parallel num_threads( wba.z() )
    {
        int threadId = omp_get_thread_num();
        int inOffset = 3 * threadId;
        int wOffset  = 3 * threadId + 1;
        int bOffset  = 3 * threadId + 2;

        // Alllocate memory on the device
        if ( cudaMalloc( (void**)&dPointers[ inOffset ], ins.size() * sizeof( dType ) ) != cudaSuccess ) {
            curnn::err::allocError( error, stringify( ins ) );
        }
        if ( cudaMalloc( (void**)&dPointers[ wOffset ], wba.x() * numInputs * sizeof( dType ) ) != cudaSuccess ) {
            curnn::err::allocError( error, stringify( weights ) );
        }
        if ( cudaMalloc( (void**)&dPointers[ bOffset ], wba.x() * sizeof( dType) ) != cudaSuccess ) {
            curnn::err::allocError( error, stringify( biases ) );
        }
        // Copy data from host to device
        if ( cudaMemcpy( dPointers[ inOffset ], &ins[0], ins.size() * sizeof( dType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
            curnn::err::copyError( error, stringify( ins ) );
        }
        if ( cudaMemcpy( dPointers[ wOffset ], &wba( 0, 0, threadId, 0 ), wba.x() * numInputs * sizeof( dType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
            curnn::err::copyError( error, stringify( weights ) );
        }
        if ( cudaMemcpy( dPointers[ bOffset ], &wba( 0, numInputs, threadId, 0 ), wba.x() * sizeof( dType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
            curnn::err::copyError( error, stringify( biases ) );
        }

        // Multiply inputs and weights (column-wise) and add biases { W^(T)*x + b }
        dType alpha = 1; dType beta = 1;
        status = curnn::blas::functions<dType>::gemv( 
                handle , CUBLAS_OP_N          , wba.x(), numInputs, &alpha              , dPointers[ wOffset ]  , 
                wba.x(), dPointers[ inOffset ], 1      , &beta    , dPointers[ bOffset ], 1                     );
        
        // Assign results to results pointer array
        results_h[ threadId ] = dPointers[ bOffset ];
    }

    // Allocate space and copy the pointers to the resuls to host memory
    if ( cudaMalloc( (void**)&results_d, wba.z() * sizeof( dType* ) ) != cudaSuccess ) {
        curnn::err::allocError( error, stringify( results_h ) );
    }
    if ( cudaMemcpy( results_d, results_h, wba.z() * sizeof( dType* ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
        curnn::err::copyError( error, stringify( results_d ) );
    }

    if ( wba.z() > 1 ) {
        // Determine sizes of blocks and threads for next kernel
        threadsX = wba.x() >= THREADS_PER_BLOCK ? THREADS_PER_BLOCK : wba.x();
        threadsY = wba.z() >= THREADS_PER_BLOCK ? THREADS_PER_BLOCK : wba.z();
        blocksX  = wba.x()  > THREADS_PER_BLOCK ? wba.x() / THREADS_PER_BLOCK + 1 : 1;
        blocksY  = wba.z()  > THREADS_PER_BLOCK ? wba.z() / THREADS_PER_BLOCK + 1 : 1;
        size_t sharedMemAmount = wba.z() * ( wba.x() / 2 ) * sizeof( dType );

        dim3 blocks(  blocksX , blocksY  );
        dim3 threads( threadsX, threadsY );

        // Sum all the results of W*x + b from each page (or layer in network)
        xpny<<<blocks, threads, sharedMemAmount>>>( results_d, wba.x(), wba.z() );
    }

    // Create pointer to the softmax resulta (node activations)
    dType* acts;
    if ( cudaMalloc( (void**)&acts, wba.x() * sizeof( dType ) ) != cudaSuccess ) {
        curnn::err::allocError( error, stringify( acts ) );
    }
    if ( cudaMemset( acts, 0, wba.x() * sizeof( dType ) ) != cudaSuccess ) {
        curnn::err::copyError( error, stringify( acts ) );
    }

    // Define grid size for the softmax operations 
    wba.x() > THREADS_PER_BLOCK * MAX_BLOCKS    ? 
        threadsX = 2 * THREADS_PER_BLOCK        : 
        threadsX = THREADS_PER_BLOCK;
    
    blocksX = std::min( static_cast<int>( wba.x() / threadsX ), MAX_BLOCKS );
    if ( blocksX * threadsX < wba.x() ) blocksX++;

    // Perform softmax on results of Wx + b (see math softmax for what each kernel does)
    blockReduceAtomicVectorizedAll<<<blocksX, threadsX>>>( dPointers[ 2 ], acts, wba.x(), expOp );
    blockScatter<<<blocksX, threadsX>>>( acts, wba.x() );
    softmaxKernel<<<blocksX, threadsX>>>( dPointers[ 2 ], acts, wba.x() );      

    if ( cudaMemcpy( &outs[ 0 ], acts, wba.x() * sizeof( dType ), cudaMemcpyDeviceToHost ) != cudaSuccess ) {
        curnn::err::copyError( error, stringify( outs ) );
    }

    cublasDestroy( handle );
 
    for ( int i = 0; i < dPointers.size(); i++ ) cudaFree( dPointers[i] );
    cudaFree( results_d ); cudaFree( acts );
}

template <typename dType>
void softmax_upadate_wba_gpu( tensor4<dType>& prev_wba, uint act_start,
                              tensor4<dType>& curr_wba, uint err_start ) {
    //
}

}   // Namespace cpu

#endif 