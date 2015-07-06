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
#include "math_kernels_cpu.hpp"

namespace curnn {
 
/*
 * ==========================================================================================================
 * Struct       : math
 * 
 * Description  : Defines a struct that holds function pointers to the relevant CPU or GPU implementation
 *                 depending on the CPU or GPU template parameter
 *                 
 * Params       : dType     : The type of data the function must use 
 *              : device    : The device to use ( CURNN_DEVICE__CPU | CURNN_DEVICE_GPU )
 * ==========================================================================================================
 */
template <typename dType, curnn::deviceType device> struct mathTest;

// Specify for CPU
template <typename dType> struct mathTest<dType, curnn::deviceType::CPU> {
    
    // X minus Y function
    typedef void (*x_minus_y)( std::vector<dType>&, std::vector<dType>&, std::vector<dType>& );
    static constexpr x_minus_y xmy = &xmyCpu;
    
};

// Specify for GPU
template <typename dType> struct mathTest<dType, curnn::deviceType::GPU> {
};

}
#endif 
