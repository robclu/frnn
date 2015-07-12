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

template <typename dType>
void updateWeights( dType* prev_acts , size_t N         ,
                    dType* curr_errs , size_t M         ,
                    dType  learn_rate, dType momentum   ) { 
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Update weights 
}

#endif
