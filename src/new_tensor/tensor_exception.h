/*
 *  Header file for fastRNN tensor exception class.
 *
 *  Copyright (C) 2015 Rob Clucas robclu1818@gmail.com
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published
 *  by the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but w_ITHOUT AN_size.y WARRANTy_; without even the implied warranty of
 *  MERCHANTABILIT_size.y or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  y_ou should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation,
 *	Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _FRNN_TENSOR_EXCEPTION_
#define _FRNN_TENSOR_EXCEPTION_

#include <iostream>
#include <exception>
#include <string>

namespace frnn {
    
class TensorOutOfRange : public std::exception {
    public:
        std::string message_;
    public:
        TensorOutOfRange(int dimension, int dimensionSize, int index) :
           message_("Error : Out of range : Attempted to access element"    + 
                     std::to_string(index)                                  + 
                     " of dimension "                                       +
                     std::to_string(dimension)                              +
                     " which has size "                                     +
                     std::to_string(dimensionSize)) {} 
        
        const char* what() const throw() { return message_.c_str(); }
};

}   // End namspace frnn

#endif
