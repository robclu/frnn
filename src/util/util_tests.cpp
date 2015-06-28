/*
 *  Test file for cuRNN util functions.
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

#include "../curnn/curnn.h"

TEST( curnnErrors, DeterminesErrorForBadAlloc ) {
	
	// Declare curnnError variable 
	curnn::curnnError error;
	
	// Declare device pointer
	float* devPointer;

	// Try an malloc without casting to void pointer, to get error
	if ( cudaMalloc( (void**)devPointer, sizeof( float ) ) != cudaSuccess ) {
		curnn::err::allocError( error, stringify( devPointer ) );
	}

	EXPECT_EQ( error, curnn::curnnError::CURNN_ALLOC_ERROR );
}

TEST( curnnErrors, DeterminesErrorForBadCopy ) {
	
	// Declare curnnError variable 
	curnn::curnnError error;

	std::vector<float> hostData = { 3.f, 4.f };
	float* devData;

	// Allocate memory on device (for 1 float)
	if ( cudaMalloc( (void**)&devData, sizeof( float ) ) != cudaSuccess ) {
		curnn::err::allocError( error, stringify( devPointer ) );
	}

	// Try and copy too much data
	if ( cudaMemcpy( devData, &hostData[0], hostData.size() * sizeof( float ), cudaMemcpyHostToDevice ) !=
		   cudaSuccess ) {
		curnn::err::copyError( error, stringify( devData ) );
	}

	EXPECT_EQ( error, curnn::curnnError::CURNN_COPY_ERROR );
}

int main(int argc, char** argv) 
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

