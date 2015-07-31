/*
 *  Header file for fastRNN tensor util functions.
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

#ifndef _FRNN_TENSOR_UTILS_
#define _FRNN_TENSOR_UTILS_

#include <vector>
#include <numeric>

namespace frnn {
namespace tensor {

/*
 * ==========================================================================================================
 * Struct       : DimensionMapper
 * 
 * Description  : Takes an index of an element in a new tensor and determines the index in each of the
 *                dimension of the old tensor which return the corresponding element in the new tensor
 *                
 * Params       : idx       : The index in the new tensor for which the mapping to the old tensor must be
 *                            determined
 *              : first     : If this is the first iteration, in which case the functor operation is different
 * ==========================================================================================================
 */
template <size_t idx, bool first> struct DimensionMapper;

template <size_t idx> struct DimensionMapper<idx, false>
{
    public:
        size_t operator()(const size_t dimSize, std::vector<size_t>& prevDimensionSizes) const 
        {
            size_t prevDimensionSizesProduct = std::accumulate(prevDimensionSizes.begin()   ,
                                                               prevDimensionSizes.end()     ,
                                                               1                            ,
                                                               std::multiplies<size_t>()    );
            return ((idx % (prevDimensionSizesProduct * dimSize)) / prevDimensionSizesProduct);
        }
};

// Specialization for all other cases
template<size_t idx> struct DimensionMapper<idx, true> 
{
    public:
        // Note : arguments are for consistancy
        size_t operator()(const size_t dimSize, const std::vector<size_t>& prevDimensionSizes) const 
        {
            return idx % dimSize;
        }
};

}       // End namespace tensor
}       // End namespace frnn

#endif
