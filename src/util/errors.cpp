/*
 *  Implementation file for cuRNN error functions.
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

#include "errors.h"

namespace curnn {
namespace err {

void allocError( curnn::curnnError& error, const char * varname ) {
	std::cerr << "Error : Could not allocate memory for variable " << varname << "\n";
	error = curnn::curnnError::CURNN_ALLOC_ERROR;
}				

void copyError( curnn::curnnError& error, const char* varname ) {
	std::cerr << "Error : Could not copy to/from variable " << varname << "\n";
	error = curnn::curnnError::CURNN_COPY_ERROR;
}
	
}	// Namepsace err
}	// Namespace curnn
