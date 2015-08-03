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
 * Description  : Takes an index, idx say 1, of an element in a new tensor, say B, which is a slice of an old 
 *                tensor, say A, and determines the index in a dimension of the old tensor A which can be used
 *                to fetch the corresponding element in A for idx in B.
 *                
 *                Say A is a 2D tensor with dimensions x = 2, y = 3, which is :
 *                
 *                A = [ x00 x01 ; 
 *                      x10 x11 ; 
 *                      x20 x21 ]     
 *                
 *                Then say that B = A[j, i] (in this case the transpose of A), then B will look like
 *                
 *                B = [ x00 x10 x20 ;  = [ idx0 idx1 idx2 idx3 idx4 idx5 ] (in memory)
 *                      x01 x11 x21 ]
 *                      
 *                then the DimensionMapper functor, for a given idx, say idx2, determines the offset in a
 *                dimension (i, or j in this case) of the element in A's memory which corresponds to idx2. In
 *                this case it would be x20 from A, so the functor would get 0 for dimension i, and 2 for
 *                dimension j.
 * Params       : iter     : Which iteration of the index mapping the algorithm is on
 * ==========================================================================================================
 */
template <size_t iter> struct DimensionMapper{
public:
    /*
     * ======================================================================================================
     * Function         : operator()
     * Description      : Determines the index of a dimension in a tensor which can be used to fetch the
     *                    element in that tensor corresponding to idx in this tensor
     * Inputs           : idx                   : The index of the element in the new tensor
     *                  : dim_size              : The size of the dimension in the original tensor for
     *                                            which the index must be determined
     *                  : prev_dim_sizes        : The sizes of the previous dimensions for which the 
     *                                            indices have been determined
     * ======================================================================================================
     */             
    size_t operator()(const size_t idx, const size_t dim_size, std::vector<size_t>& prev_dim_sizes) const 
    {
        // Computes the product of the previous dimensions sizes which were used by 
        // previous iterations, so if the previous dimension sizes were [2,3,1] then 
        // the product returns 6, which can be used to map idx to the tensor being sliced
        size_t prev_dim_sizes_product = std::accumulate(prev_dim_sizes.begin()       ,
                                                        prev_dim_sizes.end()         ,
                                                        1                            ,
                                                        std::multiplies<size_t>()    );
        
        return ((idx % (prev_dim_sizes_product * dim_size)) / prev_dim_sizes_product);
    }
};

template<> struct DimensionMapper<0> {
public:
    /*
     * ======================================================================================================
     * Function         : operator()
     * Description      : Determines the index of a dimension in a tensor which can be used to fetch the
     *                    element in that tensor corresponding to idx in this tensor
     * Inputs           : idx                   : The index of the element in the new tensor
     *                  : dim_size              : The size of the dimension in the original tensor for
     *                                            which the index must be determined
     *                  : prev_dim_sizes        : The sizes of the previous dimensions for which the 
     *                                            indices have been determined
     * ======================================================================================================
     */      
    size_t operator()(const size_t idx, const size_t dim_size) const 
    {
        return idx % dim_size;
    }
};

}       // End namespace tensor
}       // End namespace frnn

#endif
