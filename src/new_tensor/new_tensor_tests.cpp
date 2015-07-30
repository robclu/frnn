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
