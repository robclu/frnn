/*
 *  Namespace declarations for cuRNN. I honestly don't know why these 
 *  functions can be forward declared in this file but not found in their
 *  header files, but defining them in this header file and including it 
 *  works, and is also a lot cleaner!
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

#ifndef _CURNN_CURNN_
#define _CURNN_CURNN_

#include "types.h"
#include "../util/errors.h"

namespace curnn {
namespace err {

void allocError( curnn::curnnError&, const char* );
void copyError(  curnn::curnnError&, const char* );
	
}	// Namespace err
}	// Namespace curnn

#endif 