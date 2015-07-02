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

#include "matrix.hpp"

const size_t ROWS = 4;
const size_t COLS = 4;

TEST( curnnMatrix, CanCreateMatrixWithCorrectDataSize ) {
	curnn::matrix<float, ROWS, COLS> testMatrix;

	EXPECT_EQ( testMatrix.rows, ROWS );
	EXPECT_EQ( testMatrix.cols, COLS );
	EXPECT_EQ( testMatrix.data.size(), ROWS * COLS );
}

TEST (curnnMatrix, CanCreateMatrixWithInitializerList ) {
	curnn::matrix<float, 2, 2> testMatrix = { 1, 2, 3, 4 };

	EXPECT_EQ( testMatrix.data[0], 1 );
	EXPECT_EQ( testMatrix.data[1], 2 );
	EXPECT_EQ( testMatrix.data[2], 3 );
	EXPECT_EQ( testMatrix.data[3], 4 );
}

TEST( curnnMatrix, CanResizeMatrix ) {
	curnn::matrix<float, ROWS, COLS> testMatrix;

	testMatrix.resize( 10, 10 );

	EXPECT_EQ( testMatrix.rows, 10 );
	EXPECT_EQ( testMatrix.cols, 10 );
	EXPECT_EQ( testMatrix.data.size(), 10 * 10 );
}
