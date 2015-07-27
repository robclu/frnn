/*
 *  Header file for fastRNN softmax gpu kernels.
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

#ifndef _FRNN_SOFTMAX_FUNCTIONS_GPU_
#define _FRNN_SOFTMAX_FUNCITONS_GPU_

#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../../tensor/tensor.cuh"
#include "../../util/errors.h"
#include "../../frnn/frnn.h"
#include "../../math/blas/frnn_blas.h"

namespace frnn {
    
template <typename dType>
void softmaxForwardGpu( std::vector<dType>& ins       , 
                        Tensor4<dType>&     wba       ,
                        uint                num_inputs ,
                        std::vector<dType>& outs      ) {         
   
    // Thread sizes
    size_t threads_x, threads_y, blocks_x, blocks_y;

    // Statuses
    frnnError      error;
    cublasHandle_t  handle;
    cublasStatus_t  status;

    status = cublasCreate( &handle );
    cublasSetPointerMode( handle, CUBLAS_POINTER_MODE_HOST );

    // Device pointers : For each page (wba.z) we need a pointer for :
    //      inputs, weights, biases
    std::vector<dType*> d_pointers( 3 * wba.z(), 0 );
    dType*              results_h[ wba.z() ];           // Pointers to results of W*x + b on host
    dType**             results_d;                      // Pointers to results of W*x + b on device
    functors::exp       exp_op;                         // Exp operation for softmax

    // Outputs vector must have same dimension as number of nodes
    if ( outs.size() < wba.x() ) outs.resize( wba.x(), 0 );
    // Inputs vector must have same dimension as weight metrix num inputs
    if ( ins.size() != num_inputs ) {
        frnn::err::dimError( error, stringify( ins ), stringify( num_inputs ) );
        return;
    }

    // Each page is done by a separate kernel
    #pragma omp parallel num_threads( wba.z() )
    {
        int thread_id      = omp_get_thread_num();
        int in_offset      = 3 * thread_id;
        int weight_offset  = 3 * thread_id + 1;
        int bias_offset    = 3 * thread_id + 2;

        // Alllocate memory on the device
        if ( cudaMalloc( (void**)&d_pointers[ in_offset ], ins.size() * sizeof( dType ) ) != cudaSuccess ) {
            frnn::err::allocError( error, stringify( ins ) );
        }
        if ( cudaMalloc( (void**)&d_pointers[ weight_offset ], wba.x() * num_inputs * sizeof( dType ) ) != cudaSuccess ) {
            frnn::err::allocError( error, stringify( weights ) );
        }
        if ( cudaMalloc( (void**)&d_pointers[ bias_offset ], wba.x() * sizeof( dType) ) != cudaSuccess ) {
            frnn::err::allocError( error, stringify( biases ) );
        }
        // Copy data from host to device
        if ( cudaMemcpy( d_pointers[ in_offset ], &ins[0], ins.size() * sizeof( dType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
            frnn::err::copyError( error, stringify( ins ) );
        }
        if ( cudaMemcpy( d_pointers[ weight_offset ]            , &wba( 0, 0, thread_id, 0 ), 
                         wba.x() * num_inputs * sizeof( dType ) , cudaMemcpyHostToDevice   ) != cudaSuccess ) {
            frnn::err::copyError( error, stringify( weights ) );
        }
        if ( cudaMemcpy( d_pointers[ bias_offset ], &wba( 0, num_inputs, thread_id, 0 ), 
                         wba.x() * sizeof( dType ), cudaMemcpyHostToDevice            ) != cudaSuccess ) {
            frnn::err::copyError( error, stringify( biases ) );
        }

        // Multiply inputs and weights (column-wise) and add biases { W^(T)*x + b }
        dType alpha = 1; dType beta = 1;
        status = frnn::blas::functions<dType>::gemv( 
                handle , CUBLAS_OP_N            , wba.x(), num_inputs, &alpha                   , d_pointers[ weight_offset ]  , 
                wba.x(), d_pointers[ in_offset ], 1      , &beta      , d_pointers[ bias_offset ], 1                            );
        
        // Assign results to results pointer array
        results_h[ thread_id ] = d_pointers[ bias_offset ];
    }

    // Allocate space and copy the pointers to the resuls to host memory
    if ( cudaMalloc( (void**)&results_d, wba.z() * sizeof( dType* ) ) != cudaSuccess ) {
        frnn::err::allocError( error, stringify( results_h ) );
    }
    if ( cudaMemcpy( results_d, results_h, wba.z() * sizeof( dType* ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
        frnn::err::copyError( error, stringify( results_d ) );
    }

    if ( wba.z() > 1 ) {
        // Determine sizes of blocks and threads for next kernel
        threads_x = wba.x() >= THREADS_PER_BLOCK ? THREADS_PER_BLOCK : wba.x();
        threads_y = wba.z() >= THREADS_PER_BLOCK ? THREADS_PER_BLOCK : wba.z();
        blocks_x  = wba.x()  > THREADS_PER_BLOCK ? wba.x() / THREADS_PER_BLOCK + 1 : 1;
        blocks_y  = wba.z()  > THREADS_PER_BLOCK ? wba.z() / THREADS_PER_BLOCK + 1 : 1;
        size_t sharedMemAmount = wba.z() * ( wba.x() / 2 ) * sizeof( dType );

        dim3 blocks(  blocks_x , blocks_y  );
        dim3 threads( threads_x, threads_y );

        // Sum all the results of W*x + b from each page (or layer in network)
        xpny<<<blocks, threads, sharedMemAmount>>>( results_d, wba.x(), wba.z() );
    }

    // Create pointer to the softmax results (node activations)
    dType* acts;
    if ( cudaMalloc( (void**)&acts, wba.x() * sizeof( dType ) ) != cudaSuccess ) {
        frnn::err::allocError( error, stringify( acts ) );
    }
    if ( cudaMemset( acts, 0, wba.x() * sizeof( dType ) ) != cudaSuccess ) {
        frnn::err::copyError( error, stringify( acts ) );
    }

    // Define grid size for the softmax operations 
    wba.x() > THREADS_PER_BLOCK * MAX_BLOCKS    ? 
        threads_x = 2 * THREADS_PER_BLOCK       : 
        threads_x = THREADS_PER_BLOCK;
    
    blocks_x = std::min( static_cast<int>( wba.x() / threads_x ), MAX_BLOCKS );
    if ( blocks_x * threads_x < wba.x() ) blocks_x++;

    // Perform softmax on results of Wx + b (see math softmax for what each kernel does)
    blockReduceAtomicVectorizedAll<<<blocks_x, threads_x>>>( d_pointers[ 2 ], acts, wba.x(), exp_op );
    blockScatter<<<blocks_x, threads_x>>>( acts, wba.x() );
    softmaxKernel<<<blocks_x, threads_x>>>( d_pointers[ 2 ], acts, wba.x() );      

    if ( cudaMemcpy( &outs[ 0 ], acts, wba.x() * sizeof( dType ), cudaMemcpyDeviceToHost ) != cudaSuccess ) {
        frnn::err::copyError( error, stringify( outs ) );
    }

    cublasDestroy( handle );
 
    for ( int i = 0; i < d_pointers.size(); i++ ) cudaFree( d_pointers[i] );
    cudaFree( results_d ); cudaFree( acts );
}
 
template <typename dType>
void softmaxUpadateWbaGpu( Tensor4<dType>& prev_wba, uint act_start, size_t N, 
                           Tensor4<dType>& curr_wba, uint err_start, size_t M ) {
    
    frnnError error;
    
    // Each page of the matrix is done by a separate omp thread
    #pragma omp parallel num_threads( curr_wba.z ) 
    {
        float   *prev_wba_d, *curr_wba_d;
        int      thread_id;
        // Allocate device memory for weights 
        if ( cudaMalloc( (void**)&prev_wba_d, N * sizeof( dType ) ) != cudaSuccess ) {
            frnn::err::allocError( error, stringify( prev_wba_d ) );
        }
        if ( cudaMalloc( (void**)&curr_wba_d, M * sizeof( dType ) ) != cudaSuccess ) {
            frnn::err::allocError( error, stringify( curr_wba_d ) );
        }
        
        // Copy to device memory 
        if ( cudaMemcpy( prev_wba_d,          &prev_wba( act_start, 0, thread_id, 0 ), 
                         N * sizeof( dType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
            frnn::err::copyError( error, stringify( prev_wba ) );
        }
        if ( cudaMemcpy( prev_wba_d,          &prev_wba( err_start, 0, thread_id, 0 ), 
                         N * sizeof( dType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
            frnn::err::copyError( error, stringify( prev_wba ) );
        }
        
        // Invoke the kernel to upadte the weights
    }
    
}

}  // Namespace cpu

#endif 
