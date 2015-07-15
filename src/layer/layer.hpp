/*
 *  Header file for fastRNN layer class.
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

#ifndef _FRNN_LAYER_
#define _FRNN_LAYER_

#include <omp.h>

#include "../tensor/tensor.cuh"
#include "../math/math.hpp"

namespace frnn {

/*
 * ==========================================================================================================
 * Class        : layer 
 *
 * Description  : Layer class for the fastRNN that defines a generic class for a layer
 *
 * Params       : dType         : The type of data for the layer
 *              : dev           : The device to use (CPU or GPU)
 *              : _nodes        : The number of nodes in the layer
 *              : _inputs       : The number of inputs to the layer
 *              : _depth        : The number of timesteps back or forward that have inputs to this layer
 *              : typePolicy    : The type of layer
 * ==========================================================================================================
 */
template <typename                          dType,
          frnn::device                      dev,
          uint                              _nodes,
          uint                              _inputs,
          uint                              _depth,
          template <typename      , 
                    frnn::device  , 
                    uint...       >  class TypePolicy >     
class Layer : public TypePolicy<dType, dev, _nodes, _inputs, _depth> {  

    public:
        uint                num_nodes;
        uint                num_inputs;
        uint                depth;
        std::vector<dType>  outputs;
    public:
        /*
         * ==================================================================================================
         * Function     : layer 
         *
         * Description  : Defines the size of the layer parameters. The wights are stores as pages where each
         *                page is the weights between the inputs or a previous iteration of a hidden layer. 
         *                Each page has the following format :
         *                
         *                | Woo Wo1 ... WoN | N = nodes
         *                | W1o W11 ... W1N |
         *                |  .   .  .    .  |
         *                |  .   .    .  .  | 
         *                | WM0 WM1     WMN | M = max( inputs, nodes )
         *                | boP b1P ... bNP | b = bias, P = page = inputs, hidden_-1, hidden_-2 etc
         *                | aoP a1P ... aNP | a = activation, from Wx + b from its page
         *
         * ==================================================================================================
         */
        explicit Layer() :
            num_nodes( _nodes ), num_inputs( _inputs ), depth( _depth ), outputs( _nodes, 0 ) {}

        /*
         * ==================================================================================================
         * Function     : initializeWeights
         * 
         * Description  : Initialzes the weights between a certain range (by default the weights are
         *                initialized to 0 during construction.
         *
         * Inputs       : min   : The minimum value for the weights
         *              : max   : The maximum value for the weights
         * ==================================================================================================
         */
        inline void initializeWeights( dType min, dType max ) {
            // For each page in the tensor, use a thread to initialize the weights
            #pragma omp parallel num_threads ( depth )
            {
                int thread_id       = omp_get_thread_num();
                dType* weight_start = &this->wba( 0, 0, thread_id, 0 );
                size_t num_elements = num_nodes * std::max( num_nodes, num_inputs );
                
                // CPU version is a lot faster at the moment due to PCU-GPU transfer, so use CPU
                frnn::math<dType, frnn::device::CPU>::rand( weight_start, num_elements, min, max );
            }
        }
        
        /*
         * ==================================================================================================
         * Function     : getWBA
         * 
         * Description  : Returns a constant pointer to the weights (read-only)
         * 
         * Outputs      : A constant pointer to the weights, biases, and activations of the layer
         * ==================================================================================================
         */
        inline const Tensor4<dType>& getWBA() const { 
            // wba tensor in the typePolicy instance
            return this->wba;
        }
        
        /*
         * ==================================================================================================
         * Function     : outputs 
         *
         * Description  : Retuns a pointer to the outputs of the layer
         *
         * Outputs      : A constant pointer to the outputs of the layer
         * ==================================================================================================
         */
        inline const dType* getOutputs() const {
            return &outputs[ 0 ]; 
        }
        
        /*
         * ==================================================================================================
         * Function     : getErrors
         *
         * Description  : Retuns a pointer to the errors of the layer
         *
         * Outputs      : A constant pointer to the errors of the layer
         * ==================================================================================================
         */
        inline const dType* getErrors() const {
            // Errors vector in typepolicy base
            return &(this->errors[ 0 ]); 
        }
};

}   // Namespace frnn

#endif
