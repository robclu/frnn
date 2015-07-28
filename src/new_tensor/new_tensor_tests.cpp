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

#include "new_tensor.h"

using namespace frnn;

TEST(frnnNewTensor, CanCreateTensor) {
    frnn::Tensor<float, 3> testTensor;
    EXPECT_EQ( testTensor.size(), 0 );
    
    //Test counter function
    std::vector<4> temp = {2, 4, 6, 8};
    
    int off = frnn::offset( temp, 0, 1, 2, 3, 4);
    std::cout << off << std::endl;
}

TEST(frnnNewTensor, CanSpecifyTensorDimensionsInConstructor) {
    frnn::Tensor<float, 2> testTensor = {4, 3};
    EXPECT_EQ( testTensor.size(), 12 );
}

TEST(frnnNewTensor, CanSubtractTwoTensors) {
    Tensor<float, 3> tensor1 = {1, 2, 3};
    Tensor<float, 3> tensor2 = {1, 2, 3};
    
    Tensor<float, 3> newTensor = tensor1 - tensor2;
    
    EXPECT_EQ( newTensor.size(), tensor1.size() );
}

TEST(frnnNewTensor, CanSubtractThreeTensors) {
    Tensor<float, 3> tensor1 = {1, 2, 3};
    Tensor<float, 3> tensor2 = {1, 2, 3};
    Tensor<float, 3> tensor3 = {1, 2, 3};
    
    Tensor<float, 3> newTensor = tensor1 - tensor2 - tensor3;
    
    EXPECT_EQ( newTensor.size(), tensor1.size() );
}
