/*
 *  Header file for cuRNN softmax layer cpu kernels.
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

#ifndef _CURNN_SOFTMAX_KERNELS_CPU_
#define _CURNN_SOFTMAX_KERNELS_CPU_

#include "../../curnn/types.h"
#include "../../util/errors.h"
#include "../../math/math.hpp"

namespace curnn {
    
template <typename dType>
void softmaxBackwardCpu( std::vector<dType>& outs, std::vector<dType>& targets, std::vector<dType>& errors ) {
  
    curnnError error; 
    // Check dimensions
    if ( outs.size() != targets.size() ) {
        curnn::err::dimError( error, stringify( outs ), stringify( targets ) );
    } else if ( outs.size() != errors.size() ) { 
        curnn::err::dimError( error, stringify( outs ), stringify( errors ) );
    }
    
    // Call CPU X minus Y kernel because these vectors will never be big 
    // enough to warrant the data transfer between the CPU and the GPU
    curnn::math<dType, device::CPU>::xmy( outs, targets, errors );
}

}   // Namespace curnn

#endif
