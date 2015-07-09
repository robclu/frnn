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
#include <typeinfo>

#include "types.h"

/* 
 * =========================================== NOTES ========================================================
 *
 * 1. When using the vectorized instances in device code with templated functions, 
 *    you will need to first use a typedef, so for example if the template prameter 
 *    is dType then
 *
 *       template <typename dType> 
 *       void example_function( ... ) {
 *			
 *			typedef typename curnn::VectorizedTypeGpu<dType, 2> vec2;
 *
 *			vec2 vectorized_dtype;
 *		}			
 *
 * ==========================================================================================================
 */

TEST( curnnTypesGpu, CanDetermineVectorizedDoublesFromDouble ) {
	double1 cudaDouble1;
	double2 cudaDouble2;

	// Declare a curnn vectorized doubles using type traits
	curnn::VectorizedTypeGpu<double, 1>::vect_type curnnDouble1;
	curnn::VectorizedTypeGpu<double, 2>::vect_type curnnDouble2;
	
	EXPECT_EQ( typeid( cudaDouble1 ).name(), typeid( curnnDouble1 ).name() );
	EXPECT_EQ( typeid( cudaDouble2 ).name(), typeid( curnnDouble2 ).name() );
}

TEST( curnnTypesGpu, CanDetermineVectorizedFloatFromFloat ) {
	float1 cudaFloat1;
	float2 cudaFloat2;
	float4 cudaFloat4;

	// Declare curnn vectorized floats using type traits
	curnn::VectorizedTypeGpu<float, 1>::vect_type curnnFloat1;
	curnn::VectorizedTypeGpu<float, 2>::vect_type curnnFloat2;
	curnn::VectorizedTypeGpu<float, 4>::vect_type curnnFloat4;
	
	EXPECT_EQ( typeid( cudaFloat1 ).name(), typeid( curnnFloat1 ).name() );
	EXPECT_EQ( typeid( cudaFloat2 ).name(), typeid( curnnFloat2 ).name() );
	EXPECT_EQ( typeid( cudaFloat4 ).name(), typeid( curnnFloat4 ).name() );
}

TEST( curnnTypesGpu, CanDetermineVectorizedIntFromInt ) {
	int1 cudaInt1;
	int2 cudaInt2;
	int4 cudaInt4;

	// Declare curnn vectorized ints using type traits
	curnn::VectorizedTypeGpu<int, 1>::vect_type curnnInt1;
	curnn::VectorizedTypeGpu<int, 2>::vect_type curnnInt2;
	curnn::VectorizedTypeGpu<int, 4>::vect_type curnnInt4;
	
	EXPECT_EQ( typeid( cudaInt1 ).name(), typeid( curnnInt1 ).name() );
	EXPECT_EQ( typeid( cudaInt2 ).name(), typeid( curnnInt2 ).name() );
	EXPECT_EQ( typeid( cudaInt4 ).name(), typeid( curnnInt4 ).name() );
}

TEST( curnnTypesGpu, CanDetermineVectorizedUIntFromUInt ) {
	uint1 cudaUint1;
	uint2 cudaUint2;
	uint4 cudaUint4;

	// Declare curnn vectorized unsigned ints using type traits
	curnn::VectorizedTypeGpu<uint, 1>::vect_type curnnUint1;
	curnn::VectorizedTypeGpu<uint, 2>::vect_type curnnUint2;
	curnn::VectorizedTypeGpu<uint, 4>::vect_type curnnUint4;
	
	EXPECT_EQ( typeid( cudaUint1 ).name(), typeid( curnnUint1 ).name() );
	EXPECT_EQ( typeid( cudaUint2 ).name(), typeid( curnnUint2 ).name() );
	EXPECT_EQ( typeid( cudaUint4 ).name(), typeid( curnnUint4 ).name() );
}

TEST( curnnTypesGpu, CanDetermineVectorizedCharFromChar ) {
	char1 cudaChar1;
	char2 cudaChar2;
	char4 cudaChar4;

	// Declare curnn vectorized unsigned ints using type traits
	curnn::VectorizedTypeGpu<char, 1>::vect_type curnnChar1;
	curnn::VectorizedTypeGpu<char, 2>::vect_type curnnChar2;
	curnn::VectorizedTypeGpu<char, 4>::vect_type curnnChar4;
	
	EXPECT_EQ( typeid( cudaChar1 ).name(), typeid( curnnChar1 ).name() );
	EXPECT_EQ( typeid( cudaChar2 ).name(), typeid( curnnChar2 ).name() );
	EXPECT_EQ( typeid( cudaChar4 ).name(), typeid( curnnChar4 ).name() );
}

TEST( curnnTypesCpu, CanDetermineVectorizedFloatFromFloat ) {
    curnn::VectorizedTypeCpu<float>::vect_type curnnVectorizedCpu;
    __m128 sseVectorizedCpu;
    
    EXPECT_EQ( typeid( curnnVectorizedCpu ).name(), typeid( sseVectorizedCpu ).name() );
}

TEST( curnnTypesCpu, CanDetermineVectorizedDoubleFromDouble ) {
    curnn::VectorizedTypeCpu<double>::vect_type curnnVectorizedCpu;
    __m128d sseVectorizedCpu;
    
    EXPECT_EQ( typeid( curnnVectorizedCpu ).name(), typeid( sseVectorizedCpu ).name() );
}

TEST( curnnTypesCpu, CanDetermineVectorizedIntFromInt ) {
    curnn::VectorizedTypeCpu<int>::vect_type curnnVectorizedCpu;
    __m128i sseVectorizedCpu;
    
    EXPECT_EQ( typeid( curnnVectorizedCpu ).name(), typeid( sseVectorizedCpu ).name() );
}

TEST( curnnTypesCpu, CanDetermineVectorizedCharFromChar ) {
    curnn::VectorizedTypeCpu<char>::vect_type curnnVectorizedCpu;
    __m128i sseVectorizedCpu;
    
    EXPECT_EQ( typeid( curnnVectorizedCpu ).name(), typeid( sseVectorizedCpu ).name() );
}
