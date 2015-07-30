/*
 *  Test file for fastRNN matrx class.
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
 *  _size.you should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation,
 *	Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "tensor.h"

TEST( frnnTensor, CanCreateTensorWithDefaultConstructor ) 
{
    frnn::Tensor<float, 3> testTensor;
    EXPECT_EQ( testTensor.size(), 0 );
}

TEST( frnnTensor, CanSpecifyTensorDimensionsWithList ) 
{
    frnn::Tensor<float, 2> testTensor = {4, 3};
    EXPECT_EQ( testTensor.size(), 12 );
}

TEST( frnnTensor, CanCreateTensorFromDimensionSizesAdnData )
{
    std::vector<size_t> dimensionSizes = {2, 3};
    std::vector<float> data = {1.f, 2.f, 
                               3.f, 4.f,
                               5.f, 6.f};
    frnn::Tensor<float, 2> tensor(dimensionSizes, data);
    
    const std::vector<float>& tensorData = tensor.data();
    
    EXPECT_EQ( tensorData[1], 2.f );
}

TEST( frnnTensor, CanGetRankOfTensor ) 
{
    frnn::Tensor<float, 3> tensor = {1, 4, 4};
    
    int rank = tensor.rank();
    
    EXPECT_EQ( rank, 3 );
}

TEST( frnnTensor, CanGetTensorDimensions )
{
    frnn::Tensor<int, 3> tensor = {2, 1, 3};
    
    std::vector<size_t> dims = tensor.dimSizes();
    
    EXPECT_EQ( dims[0], 2 );
    EXPECT_EQ( dims[1], 1 );
    EXPECT_EQ( dims[2], 3 );
}

TEST( frnnTensor, CanGetSizeOfTensor ) 
{
    frnn::Tensor<double, 4> tensor = {2, 3, 2, 4};
    
    int size = tensor.size();
    
    EXPECT_EQ( size, 48 );
}

TEST( frnnTensor, CanGetSizeOfASpecificDimensionOfTensor ) 
{
    frnn::Tensor<float, 3> tensor = {1, 2, 3};
    
    int dim0Size = tensor.size(0);
    int dim2Size = tensor.size(2);
    
    EXPECT_EQ( dim0Size, 1 );
    EXPECT_EQ( dim2Size, 3 );
}

TEST( frnnTensor, CanHandleOutOfRangeIndexForSizeFunction ) 
{
    frnn::Tensor<int, 8> tensor = {1, 2, 4, 5, 3, 1, 1, 8};
    
    // Wrong due to 0 indexing
    int dimSize8  = tensor.size(8);
    int dimSize10 = tensor.size(10);
    
    EXPECT_EQ( dimSize8 , 0 );
    EXPECT_EQ( dimSize10, 0 );
}

TEST( frnnTensor, CanGetReferenceToTensorData ) 
{
    frnn::Tensor<float, 3> tensor1 = {1, 2, 3};
    
    const std::vector<float>& tensorData = tensor1.data();
    
    EXPECT_EQ( tensorData.size(), 6 );
    EXPECT_EQ( tensorData[0], 0.f );
}
    
TEST( frnnTensor, CanSubtractTwoTensors ) 
{
    frnn::Tensor<float, 3> tensor1 = {1, 2, 3};
    frnn::Tensor<float, 3> tensor2 = {1, 2, 3};
    
    frnn::Tensor<float, 3> newTensor = tensor1 - tensor2;
    
    EXPECT_EQ( newTensor.size(), tensor1.size() );
}

TEST( frnnTensor, CanSubtractThreeTensors ) 
{
    frnn::Tensor<float, 3> tensor1 = {1, 2, 3};
    frnn::Tensor<float, 3> tensor2 = {1, 2, 3};
    frnn::Tensor<float, 3> tensor3 = {1, 2, 3};
    
    frnn::Tensor<float, 3> newTensor = tensor1 - tensor2 - tensor3;
    
    EXPECT_EQ( newTensor.size(), tensor1.size() );
}

TEST( frnnTensor, CanGetTensorRankAfterOperation )
{
    frnn::Tensor<int, 4> tensor1 = {1, 2, 1, 1};
    frnn::Tensor<int, 4> tensor2 = {1, 2, 1, 1};
    
    frnn::Tensor<int, 4> tensor3 = tensor1 - tensor2;
    
    int rank = tensor3.rank();
    EXPECT_EQ( rank, 4 );
}

TEST( frnnTensor, CanGetDimensionSizesAfterOperation ) 
{
    frnn::Tensor<int, 4> tensor1 = {1, 2, 1, 1};
    frnn::Tensor<int, 4> tensor2 = {1, 2, 1, 1};
    
    frnn::Tensor<int, 4> tensor3 = tensor1 - tensor2;
    
    std::vector<size_t> dims = tensor3.dimSizes();
    EXPECT_EQ( dims[ 0 ], 1 );
    EXPECT_EQ( dims[ 1 ], 2 );
    EXPECT_EQ( dims[ 2 ], 1 );
    EXPECT_EQ( dims[ 3 ], 1 );
}

TEST( frnnTensor, CanGetElementOfTensor ) 
{
    frnn::Tensor<float, 3> tensor = {2, 5, 4};
    
    float element = tensor(1, 3, 3);
    
    EXPECT_EQ( element, 0.f );
}

TEST( frnnTensor, CanSetElementOfTensor ) 
{
    frnn::Tensor<int, 3> tensor = {3, 3, 3};
    
    // Set 2nd element 
    tensor(1, 0, 0) = 4;

    int x = tensor(1, 0, 0);
    x = 12;
    
    // Get data 
    const std::vector<int>& tensorData = tensor.data();
    
    EXPECT_EQ( tensorData[1], 4 );
    EXPECT_EQ( tensor(1, 0, 0), 4 );
}

TEST( frnnTensor, ThrowsErrorForInvalidAccessOperatorArguments) 
{
    frnn::Tensor<double, 4> tensor = {4, 5, 3, 3};
    
    // Provide invalid number of arguments, 
    // should throw error and return 0
    double element = tensor(1, 1, 1, 1, 1);
    
    EXPECT_EQ( element, 0.0 );
} 

TEST( frnnTensor, ThrowsErrorForOutOfRangeElementAccess) 
{
    frnn::Tensor<int, 3> tensor = {3, 3, 3};
    
    // Access element 3 of dimension with size 3
    // (Tensor indexing is 0 based)
    int element1 = tensor(1, 3, 2);
    int element2 = tensor(4, 1, 1);
    int element3 = tensor(1, 1, 5);
    
    EXPECT_EQ( element1, 0 );
    EXPECT_EQ( element2, 0 );
    EXPECT_EQ( element3, 0 );
}
