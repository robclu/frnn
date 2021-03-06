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

#include "tensor.cuh"

const size_t W = 3;
const size_t X = 4;
const size_t Y = 4;
const size_t Z = 4;

TEST(frnnTensor, CanCreateTensorCorrectly) {
	frnn::Tensor4<float> testTensor(X, Y, Z, W);

	EXPECT_EQ( testTensor.w(), W );
	EXPECT_EQ( testTensor.x(), X );
	EXPECT_EQ( testTensor.y(), Y );
	EXPECT_EQ( testTensor.z(), Z );
	EXPECT_EQ( testTensor.size(), W * X * Y * Z );
}

TEST(frnnTensor, TensorValuesDefaultToZero) {
	frnn::Tensor4<float> testTensor(X, Y, Z, W);

	for (size_t l = 0; l < testTensor.w(); l++) {
		for (size_t k = 0; k < testTensor.z(); k++) {
			for (size_t j = 0; j < testTensor.y(); j++) {
				for (size_t i = 0; i < testTensor.x(); i++) {
					EXPECT_EQ( testTensor(i, j, k, l), 0.f );
				}
			}
		}
	}
}

TEST(frnnTensor, CanReshapeTensor) {
	frnn::Tensor4<float> testTensor(X, Y, Z, W);

	testTensor.reshape(10, 10, 7, 2);

	EXPECT_EQ( testTensor.w(), 2  );
	EXPECT_EQ( testTensor.x(), 10 );
	EXPECT_EQ( testTensor.y(), 10 );
	EXPECT_EQ( testTensor.z(), 7  );
	EXPECT_EQ( testTensor.size(), 10 * 10 * 7 * 2 );
}

TEST(frnnTensor, TensorValuesAreZeroAfterReshape) {
	frnn::Tensor4<float> testTensor(X, Y, Z, W);
	testTensor.reshape(10, 2, 3, 9);

	for (size_t l = 0; l < testTensor.w(); l++) {
		for (size_t k = 0; k < testTensor.z(); k++) {
			for (size_t j = 0; j < testTensor.y(); j++) {
				for (size_t i = 0; i < testTensor.x(); i++) {
					EXPECT_EQ( testTensor(i, j, k, l), 0.f );
				}
			}
		}
	}
}

TEST(frnnTensor, ReshapeCanUsexistingDimensionality) {
	frnn::Tensor4<float> testTensor(X, Y, Z, W);
	testTensor.reshape(10, -1, 3, -1);

	EXPECT_EQ( testTensor.x(), 10 );
	EXPECT_EQ( testTensor.y(), Y  );
	EXPECT_EQ( testTensor.z(), 3  );
	EXPECT_EQ( testTensor.w(), W  );
	EXPECT_EQ( testTensor.size(), 10 * Y * 3 * W );
}

TEST(frnnTensor, CanGetAndSetElementWithAccessOperator) {
	frnn::Tensor4<float> testTensor(X, Y, Z, W);
	testTensor(1, 0, 0, 0) = 5.f;

	// Use operator
	float testVal = testTensor(1, 0, 0, 0);

	EXPECT_EQ( testVal, 5.f );
}

TEST(frnnTensor, OutputsErrorForOutOfRangeIndxAndReturnsFirstElementValue) {
	frnn::Tensor4<float> testTensor( X, Y, Z, W );

	float testVal = testTensor(100, 12, 34, 2);

	EXPECT_EQ( testVal, testTensor(0, 0, 0, 0) );
}

TEST(frnnTensor, CanGetPointerToElement) {
	frnn::Tensor4<float> testTensor(X, Y, Z, W);

	testTensor(0, 0, 1, 0) = 10.f;	
	float* testVal = &testTensor(0, 0, 1, 0);

	EXPECT_EQ( *testVal, 10.f );
}

TEST(frnnTensor, CanModifyElementThroughPointer) {
   	frnn::Tensor4<float> testTensor(X, Y, Z, W);

	testTensor(0, 0, 1, 0) = 10.f;	
	float* testVal = &testTensor(0, 0, 1, 0);

    *testVal = 4.f;
    
	EXPECT_EQ( *testVal, 4.f );
}

TEST(frnnTensor, CanCreateArrayFromTensor) {
	frnn::Tensor4<float> testTensor(X, Y, Z, W);

	// Set elements
	for (size_t i = 0; i < testTensor.x(); i++) {
		testTensor(i, 0, 0, 0) = float(i);
	}

	float* testPointer;
	testPointer = &testTensor(0, 0, 0, 0);

	for (int i = 0; i < X; i++) {
		EXPECT_EQ( float(i), testPointer[i] );
	}
}

TEST(frnnTensor, CanMoveTensorDataCorrectly) {
    frnn::Tensor4<float> t1(2, 2, 1, 1);
    frnn::Tensor4<float> t2(2, 2, 1, 1);
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            t1(i, j, 0, 0) = float(j);
        }
    }
    
    t1.moveData(t2);
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            EXPECT_EQ( t2(i, j, 0, 0), float(j) );
        }
    }
}

