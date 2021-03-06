/*
 *  Header file for fastRNN types.
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

#ifndef _FRNN_TYPES_
#define _FRNN_TYPES_

// Other frnn types
#include "vectorized_types_cpu.h"
#include "vectorized_types_gpu.h"

// Change if necessary
#define MAX_BLOCKS          65536
#define THREADS_PER_BLOCK   256

namespace frnn {

/*
 * ==========================================================================================================
 * Enum         : device
 * 
 * Decsiption   : Enumerator for the devices available for use
 * ==========================================================================================================
 */
enum device : bool {
    CPU,
    GPU
};

/*
 * ==========================================================================================================
 * Enum         : frnnError
 *
 * Description  : Enumerator for possible erorrs in frnn.
 * ==========================================================================================================
 */
enum frnnError {
    FRNN_ALLOC_ERROR       = 1,
    FRNN_COPY_ERROR        = 2,
    FRNN_DIMENSION_ERROR   = 3
 };

}   // Namepace frnn

#endif
