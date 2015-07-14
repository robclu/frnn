/*
 *  Header file for fastRNN softmax policy class.
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

#ifndef _FRNN_SOFTMAX_POLICY_
#define _FRNN_SOFTMAX_POLICY_

#include <vector>

#include "../../tensor/tensor.cuh"
#include "../../frnn/frnn.h"
#include "softmax_cpu_functions.hpp"
#include "softmax_gpu_functions.cuh"

namespace frnn {
namespace ltype {

/*
 * ==========================================================================================================
 * Class        : SoftmaxPolicy 
 *
 * Desription   : Policy class for a softmax layer, which defines the forward and backward propogations
 *
 * Params       : dType     : The type of data for the network
 *              : device    : The device type to use (CPU or GPU)
 *              : nodes     : The number of nodes for the layer
 *              : inputs    : The number of inputs to the layer
 *              : depth     : The number of different inputs in the layer (almost always 1 for softmax)
 * ==========================================================================================================
 */
template <typename          dType, 
         frnn::device      dev,
         uint               nodes, 
         uint               inputs, 
         uint               depth>
class SoftmaxPolicy;

/* ============================================== GPU Definitions ========================================  */

template <typename          dType, 
          uint              nodes,
          uint              inputs,
          uint              depth>
class SoftmaxPolicy<dType, frnn::device::GPU, nodes, inputs, depth> {
    
    public:
        /*
         * ==================================================================================================
         * Function     : SofmaxPolicy
         *
         * Description  : Constructor for the softmaxPolicy. Sets the tensor (wba) which holds the weights,
         *                biases, and activations (using the forst 2 dimensions of the tensor), and the number
         *                of inputs for the layer.
         * ==================================================================================================
         */
        explicit SoftmaxPolicy() :
            wba( nodes, inputs + 2, depth, 1 ), num_inputs( inputs ), errors( nodes, 0 ) {}

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
         * Function     : backward 
         * 
         * Description  : Backward propogates the errors through the layer, for the softmax layer (as it is an
         *                output layer) this just determines the difference between the targets and the
         *                outputs.
         * 
         * Inputs       : outs      : The outputs of the layer
         *              : targets   : The targets for each of the outputs of the layer
         *              
         * Outputs      : The results are stored in the errors vector
         * ==================================================================================================
         */
        void backward( std::vector<dType>& outs, std::vector<dType>&targets );

        /* 
         * ==================================================================================================
         * Function     : updateWba 
         * 
         * Description  : Updates the weights, biases and activations
         * 
         * Inputs       : prevLayerActs   : The activations (outputs) of the nodes in the previous layer
         * ==================================================================================================
         */
        void updateWba( const frnn::Tensor4<dType>& prevLayerActs );
        
    protected:
        Tensor4<dType>      wba;             // Tensor for weights, biases, and activations
        Tensor4<dType>      wba_prev;        // Tensor for weights, biases, and activations from the previous timestep
        std::vector<dType>  errors;          // Errors for the layer
        uint                num_inputs;      // Number of inputs for the layer
};

/* =============================================== CPU Definitions ======================================== */

template <typename          dType, 
          uint              nodes,
          uint              inputs,
          uint              depth>
class SoftmaxPolicy<dType, frnn::device::CPU, nodes, inputs, depth> {
    
    public:
        /*
         * ==================================================================================================
         * Function     : SofmaxPolicy
         *
         * Description  : Constructor for the softmaxPolicy. Sets the tensor (wba) which holds the weights,
         *                biases, and activations (using the forst 2 dimensions of the tensor), and the number
         *                of inputs for the layer.
         * ==================================================================================================
         */
        explicit SoftmaxPolicy() :
            wba( nodes, inputs + 2, depth, 1 ), num_inputs( inputs ), errors( nodes, 0 ) {}

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
         * Function     : backward 
         * 
         * Description  : Backward propogates the errors through the layer, for the softmax layer (as it is an
         *                output layer) this just determines the difference between the targets and the
         *                outputs.
         * 
         * Inputs       : outs      : The outputs of the layer
         *              : targets   : The targets for each of the outputs of the layer
         *              
         * Outputs      : The results are stored in the errors vector
         * ==================================================================================================
         */
        void backward( std::vector<dType>& outs, std::vector<dType>&targets );

        /* 
         * ==================================================================================================
         * Function     : updateWba 
         * 
         * Description  : Updates the weights, biases and activations
         * 
         * Inputs       : prevLayerActs   : The activations (outputs) of the nodes in the previous layer
         * ==================================================================================================
         */
        void updateWba( const frnn::Tensor4<dType>& prevLayerActs );
        
    protected:
        Tensor4<dType>      wba;             // Tensor for weights, biases, and activations
        Tensor4<dType>      wba_prev;        // Tensor for weights, biases, and activations from the previous timestep
        std::vector<dType>  errors;          // Errors for the layer
        uint                num_inputs;      // Number of inputs for the layer
};

/* ======================================= GPU IMPLEMENTATIONS ============================================ */

template <typename dType, uint nds, uint ipts, uint dth>
void SoftmaxPolicy<dType, device::GPU, nds, ipts, dth>::forward (
        std::vector<dType>& ins, std::vector<dType>& outs ) {
    // Call softmax forward gpu version 
    softmaxForwardGpu( ins, wba, num_inputs, outs );
}

template <typename dType, uint nds, uint ipts, uint dth>
void SoftmaxPolicy<dType, device::GPU, nds, ipts, dth>::backward(
        std::vector<dType>& outs, std::vector<dType>& targets ) {
    // Even though this is the GPU version, 
    // the CPU version is faster, so use that
    softmaxBackwardCpu( outs, targets, errors );
}

// NOT DONE
template <typename dType, uint nds, uint ipts, uint dth>
void SoftmaxPolicy<dType, device::GPU, nds, ipts, dth>::updateWba( 
        const frnn::Tensor4<dType>& prev_layer_acts ) {
}

/* ======================================= CPU IMPLEMENTATIONS  =========================================== */

// NOT DONE
template <typename dType, uint nds, uint ipts, uint dth>
void SoftmaxPolicy<dType, device::CPU, nds, ipts, dth>::forward( 
        std::vector<dType>& ins, std::vector<dType>& outs ) {
    // CPU implementation not done yet, use gpu
    softmaxForwardGpu( ins, wba, num_inputs, outs );
}

template <typename dType, uint nds, uint ipts, uint dth>
void SoftmaxPolicy<dType, device::CPU, nds, ipts, dth>::backward( 
        std::vector<dType>& outs, std::vector<dType>& targets ) {
    // Call softmax backward cpu kernel
    softmaxBackwardCpu( outs, targets, errors );
}

// NOT DONE
template <typename dType, uint nds, uint ipts, uint dth>
void SoftmaxPolicy<dType, device::CPU, nds, ipts, dth>::updateWba( 
        const frnn::Tensor4<dType>& prev_layer_acts ) {
    
}

}   // Namepsace lloss
}   // Namepsace frnn
#endif 
