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
 * Params       : isFirst     : If this is the first iteration, in which case the functor operation is 
 *                              different
 * ==========================================================================================================
 */
template <size_t iter> struct DimensionMapper
{
    public:
        /*
         * ==================================================================================================
         * Function         : operator()
         * 
         * Description      : Determines the index of a dimension in a tensor being sliced, for the index in
         *                    the new tensor (which is being created from the slice)
         * 
         * Inputs           : idx                   : The index of the element in the new tensor
         *                  : dimSize               : The size of the dimension in the original tensor for
         *                                            which the index must be determined
         *                  : prevDimensionSizes    : The sizes of the previous dimensions for which the 
         *                                            indices have been determined
         * ==================================================================================================
         */             
        size_t operator()(const size_t idx, const size_t dimSize, 
                          std::vector<size_t>& prevDimensionSizes) const 
        {
            size_t prevDimensionSizesProduct = std::accumulate(prevDimensionSizes.begin()   ,
                                                               prevDimensionSizes.end()     ,
                                                               1                            ,
                                                               std::multiplies<size_t>()    );
            return ((idx % (prevDimensionSizesProduct * dimSize)) / prevDimensionSizesProduct);
        }
};

template<> struct DimensionMapper<0> 
{
    public:
        /*
         * ==================================================================================================
         * Function         : operator()
         * 
         * Description      : Determines the index of a dimension in a tensor being sliced, for the index in
         *                    the new tensor (which is being created from the slice)
         * 
         * Inputs           : idx                   : The index of the element in the new tensor
         *                  : dimSize               : The size of the dimension in the original tensor for
         *                                            which the index must be determined
         *                  : prevDimensionSizes    : The sizes of the previous dimensions for which the 
         *                                            indices have been determined
         * ==================================================================================================
         */      
        size_t operator()(const size_t idx, const size_t dimSize, 
                          const std::vector<size_t>& prevDimensionSizes) const 
        {
            return idx % dimSize;
        }
};

}       // End namespace tensor
}       // End namespace frnn

#endif
