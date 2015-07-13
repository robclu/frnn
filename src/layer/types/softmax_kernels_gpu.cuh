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

#ifndef _CURNN_SOFTAMX_KERNELS_GPU_
#define _CURNN_SOFTAMX_KERNELS_GPU_

#include <cuda.h>
#include <cuda_runtime.h>

/*
 * ==========================================================================================================
 * Function         : updateWeights 
 * 
 * Description      : Updates the weights for a layer using the GPU. The update rule (amount to modify the
 *                    weight by) is first determined, and is then subtracted from the current value of the
 *                    weight.
 *                    
 * Inputs           : curr_wba      : A pointer to the current wba tensor 
 *                  : ces           : Current Error Start, which is the endex of the first error element in
 *                  :               : the curr_wba array
 *                  : num_errors    : The dimensionality of the erros (how many errors there are, which is the
 *                  :               : number of elements after ces till the last error element.
 *                  : prev_wba      : The previous wba tensor
 *                  : pas           : Previous Activation Start, which has the same meaning as ces but points
 *                  :               : to where the activations start
 *                  : num_acts      : The number of elements past pas that are activation elements
 *                  : prev_wba_delta: An array of the weight update values from the previous iteration
 *                  : learn_rate    : The learning rate to use for the update
 *                  : momentum      : The amount of momentum to use for the update
 * ==========================================================================================================
 */
template <typename dType>
void updateWeights( dType* curr_wba       , size_t ces       , size_t num_errs,
                    dType* prev_wba       , size_t pas       , size_t num_acts,
                    dType* prev_wba_delta , dType learn_rate , dType momentum ) {
    
    // Get thread index to use as weight index 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Determine the weight update (might need atomics)
    dType prev_weight_update = prev_wba_delta[ idx ];
    prev_weight_delta[ idx ] = momentum * prev_weight_update -              // Momentum contribution
                                learn_rate *                                // Gradient descent
                                curr_wba[ ces + ( idx % num_errs ) ] *      // This layers error      
                                prev_wba[ pas + ( idx / num_acts ) ];       // Prev layer activation
    // Update the weight with the weight update
    curr_wba[ idx ] += prev_weight_delts[ idx ];
}

#endif
