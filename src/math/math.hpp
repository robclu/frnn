/*
 *  Header file for cuRNN math functions. The math struct is defined which 
 *  uses a template parameter to determine if the CPU or GPU functions 
 *  should be used. static constexpr function pointers are then used to call
 *  the relevant CPU or GPU implementation.
 *
 *  Copyright (C) 2015 Rob Clucas robclu1818@gmail.com
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published
 *  by the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT AN_size.y WARRANTY; without even the implied warranty of
 *  MERCHANTABILIT_size.y or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation,
 *  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _CURNN_MATH_GENERAL_
#define _CURNN_MATH_GENERAL_

#include <vector>

#include "../curnn/types.h"
#include "math_cpu.hpp"
#include "math_gpu.hpp"

namespace curnn {
 
/*
 * ==========================================================================================================
 * Struct       : math
 * 
 * Description  : Defines a struct that holds function pointers to the relevant CPU or GPU implementation
 *                 depending on the CPU or GPU template parameter
 *                 
 * Params       : dType  : The type of data the function must use 
 *              : dev    : The device to use (CPU | GPU)
 * ==========================================================================================================
 */
template <typename dType, curnn::device dev> struct math;

// Specify for CPU
template <typename dType> struct math<dType, curnn::device::CPU> {
    
    // X minus Y function
    typedef void (*x_minus_y_cpu)( std::vector<dType>&, std::vector<dType>&, std::vector<dType>& );
    static constexpr x_minus_y_cpu xmy = &xmyCpu;
    
    // Rand function
    typedef dType (*rand_cpu)( dType, dType );
    static constexpr rand_cpu rand = &randCpu; 

};

// Specify for GPU
template <typename dType> struct math<dType, curnn::device::GPU> {
    
    // a*X plus Y function 
    typedef void (*ax_plus_y_gpu)( curnnError&, const dType a, const std::vector<dType>&, std::vector<dType>& );
    static constexpr ax_plus_y_gpu axpy = &axpyGpu;
   
    // Softmax fucntion 
    typedef void (*softmax_gpu)( curnnError&, const std::vector<dType>&, std::vector<dType>& );
    static constexpr softmax_gpu softmax = &softmaxGpu;
    
    // Sum function
    typedef dType (*sum_gpu)( curnnError&, const std::vector<dType>& );
    static constexpr sum_gpu sum = &sumGpu;
    
    // Sum vectorized function
    typedef void (*sum_vectorized_gpu)( curnnError&, const std::vector<dType>&, std::vector<dType>&);
    static constexpr sum_vectorized_gpu sumVectorized = &sumVectorizedGpu;
    
};

}
#endif 
