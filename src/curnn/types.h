/*
 *  Header file for cuRNN types.
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

#ifndef _CURNN_TYPES_
#define _CURNN_TYPES_

// Other curnn types
#include "vectorizedTypesCpu.h"
#include "vectorizedTypesGpu.h"

// Change if necessary
#define MAX_BLOCKS          65536
#define THREADS_PER_BLOCK   256

namespace curnn {

/*
 * ==========================================================================================================
 * Enum         : curnnError
 *
 * Description  : Enumerator for possible erorrs in curnn.
 * ==========================================================================================================
 */
enum curnnError {
    CURNN_ALLOC_ERROR       = 1,
    CURNN_COPY_ERROR        = 2,
    CURNN_DIMENSION_ERROR   = 3
    };

}   // Namepace curnn

#endif
