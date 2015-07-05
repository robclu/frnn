/*
 *  Header file for cuRNN softmax policy class.
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

#ifndef _CURNN_SOFTMAX_POLICY_
#define _CURNN_SOFTMAX_POLICY_

#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../tensor/tensor.cuh"
#include "../util/errors.h"
#include "../curnn/curnn.h"
#include "../math/math.cuh"

namespace curnn {
namespace ltype {

/*
 * ==========================================================================================================
 * Class        : softmaxPolicy 
 *
 * Desription   : Policy class for a softmax layer, which defines the forward and backward propogations
 *
 * Params       : dType     : The type of data for the network
 *              : nodes     : The number of nodes for the layer
 *              : inputs    : The number of inputs to the layer
 *              : depth     : The number of different inputs in the layer (almost always 1 for softmax)
 * ==========================================================================================================
 */
template <typename dType, uint nodes, uint inputs, uint depth>
class softmaxPolicy {

    public:
        /*
         * ==================================================================================================
         * Function     : sofmaxPolicy
         *
         * Description  : Constructor for the softmaxPolicy. Sets the tensor (wba) which holds the weights,
         *                biases, and activations (using the forst 2 dimensions of the tensor), and the number
         *                of inputs for the layer.
         * ==================================================================================================
         */
        explicit softmaxPolicy() :
            wba( nodes, inputs + 2, depth, 1 ), numInputs( inputs ) {}

        /*
         * ==================================================================================================
         * Function     : forward
         *
         * Description  : Forward propogates the inputs through the layer, to determine the activations
         *                (outputs for the softmax layer) and returns the outputs.
         *
         * Inputs       : ins   : The inputs to the layer, for example the outputs of the hidden layer before
         *                        this layer.
         *
         * Outputs      : outs  : The outputs of the layer after performing softmax( W*x + b ) on the inputs.
         * ==================================================================================================
         */
        void forward( std::vector<dType>& ins, std::vector<dType>& outs );
        
        /*
         * ==================================================================================================
         * Function     : getErrors
         * 
         * Description  : Gets the errors of the layer given a tensor, in this case the tensor will hold the
         *                layer outputs and the targets.
         *                
         * Inputs       : outs      : The outputs of the layer
         *              : targets   : The targets for each output
         * 
         * Outputs      : The results are stored in the errors vector of the class
         * ==================================================================================================
         */
        void getErrors( std::vector<dType> outs, std::vector<dType> targets );
        
    protected:
        tensor4<dType>      wba;            // Tensor for weights, biases, and activations
        std::vector<dType>  errors;         // Errors for the layer
        uint                numInputs;      // Number of inputs for the layer
};

/* ==========================================  Implementations ============================================ */

template <typename dType, uint nds, uint ipts, uint dth>
void softmaxPolicy<dType, nds, ipts, dth>::forward( std::vector<dType>& ins, std::vector<dType>& outs ) {

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
    expFunctor          expOp;                          // Exp operation for softmax

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
        status = cublasSgemv( handle, CUBLAS_OP_N, wba.x(), numInputs, &alpha, dPointers[ wOffset ], 
                              wba.x(), dPointers[ inOffset ], 1, &beta, dPointers[ bOffset ], 1 );
        
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

    cudaDeviceSynchronize();
    
    for ( int i = 0; i < dPointers.size(); i++ ) cudaFree( dPointers[i] );
    cudaFree( results_d ); cudaFree( acts );
}

template <typename dType, uint nds, uint ipts, uint dth>
void softmaxPolicy<dType, nds, ipts, dth>::getErrors( std::vector<dType>& outs, std::vector<dType>& targets ) {
    curnnError& error;
    if ( outs.size() != targets.size() ) {
        curnn::err::dimError( error, stringify( outs ), stringify( targets ) );
        return;
    }
    // Data will never be big enough to use GPU, so use CPU
    for ( uint i = 0; i < outs.size(); i++ ) error[ i ] = outs[ i ] - targets[ i ];
}

}   // Namepsace lloss
}   // Namepsace curnn
#endif 
