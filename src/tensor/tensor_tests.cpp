/*
 *  Test file for cuRNN matrix class.
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

TEST( curnnTensor, CanCreateTensorCorrectly ) {
	curnn::tensor4<float> testTensor( X, Y, Z, W );

	EXPECT_EQ( testTensor.w, W );
	EXPECT_EQ( testTensor.x, X );
	EXPECT_EQ( testTensor.y, Y );
	EXPECT_EQ( testTensor.z, Z );
	EXPECT_EQ( testTensor.size(), W * X * Y * Z );
}

TEST( curnnTensor, TensorValuesDefaultToZero ) {
	curnn::tensor4<float> testTensor( X, Y, Z, W );

	for ( size_t i = 0; i < testTensor.size(); i++ ) {
		EXPECT_EQ( testTensor.data[ i ], (float)0 );
	}
}

TEST( curnnTensor, CanReshapeTensor ) {
	curnn::tensor4<float> testTensor( X, Y, Z, W );

	testTensor.reshape( 10, 10, 7, 2 );

	EXPECT_EQ( testTensor.w, 2 );
	EXPECT_EQ( testTensor.x, 10 );
	EXPECT_EQ( testTensor.y, 10 );
	EXPECT_EQ( testTensor.z, 7 );
	EXPECT_EQ( testTensor.size(), 10 * 10 * 7 * 2 );
}

TEST( curnnTensor, TensorValuesAreZeroAfterReshape ) {
	curnn::tensor4<float> testTensor( X, Y, Z, W );
	testTensor.reshape( 10, 2, 3, 9 );

	for ( size_t i = 0; i < testTensor.size(); i++ ) {
		EXPECT_EQ( testTensor.data[ i ], (float)0 );
	}
}

TEST( curnnTensor, ReshapeCanUseExistingDimensionality ) {
	curnn::tensor4<float> testTensor( X, Y, Z, W );
	testTensor.reshape( 10, -1, 3, -1 );

	EXPECT_EQ( testTensor.x, 10 );
	EXPECT_EQ( testTensor.y, Y );
	EXPECT_EQ( testTensor.z, 3 );
	EXPECT_EQ( testTensor.w, W );
	EXPECT_EQ( testTensor.size(), 10 * Y * 3 * W );
}

TEST( curnnTensor, CanGetElementWithAccessOperator ) {
	curnn::tensor4<float> testTensor( X, Y, Z, W );
	testTensor.data[ 1 ] = 5.f;

	// Use operator
	float testVal = testTensor( 1, 0, 0, 0 );

	EXPECT_EQ( testVal, 5.f );
}

TEST( curnnTensor, CanSetElementWithAccessOperator ) {
	curnn::tensor4<float> testTensor( X, Y, Z, W );
	testTensor.data[ 1 ] = 5.f;

	// Use operator
	testTensor( 1, 0, 0, 0 ) = 10.f;
	float testVal = testTensor( 1, 0, 0, 0 );

	EXPECT_EQ( testVal, 10.f );
}
TEST( curnnTensor, OutputsErrorForOutOfRangeIndexAndReturnsFirstElementValue ) {
	curnn::tensor4<float> testTensor( X, Y, Z, W );

	float testVal = testTensor( 100, 12, 34, 2 );

	EXPECT_EQ( testVal, testTensor.data[ 0 ] );
}

TEST( curnnTensor, CanGetPointerToElement ) {
	curnn::tensor4<float> testTensor( X, Y, Z, W );

	testTensor( 0, 0, 1, 0 ) = 10.f;	
	float* testVal = &testTensor( 0, 0, 1, 0 );

	EXPECT_EQ( *testVal, 10.f );
}

TEST( curnnTensor, CanCreateArrayFromTensor ) {
	curnn::tensor4<float> testTensor( X, Y, Z, W );

	// Set elements
	for ( size_t i = 0; i < testTensor.x; i++ ) {
		testTensor( i, 0, 0, 0 ) = float ( i );
	}

	float* testPointer;
	testPointer = &testTensor( 0, 0, 0, 0 );

	for ( int i = 0; i < X; i++ ) {
		EXPECT_EQ( float( i ), testPointer[ i ] );
	}
}
