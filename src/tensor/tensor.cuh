/*
 *  Header file for fastRNN tensor class.
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

#ifndef _FRNN_TENSOR_
#define _FRNN_TENSOR_

#include <vector>
#include <iostream>
#include <limits>

namespace frnn  {

/*
 * ==========================================================================================================
 * Class		: Tensor4
 *
 * Description	: Provides a 4D tensor to store 4 dimensionaly data, or to join data to 4 dimensions to that
 *				  less passes need to be made to the GPU
 *
 * Params		: dType		: The data type for the matrix
 * ==========================================================================================================
 */
template <typename dType>
class Tensor4 {
	private:
		uint				w_;
		uint				x_;
		uint				y_; 
		uint				z_;
		std::vector<dType>	data;
	public:
		/*
		 * ==================================================================================================
		 * Function			: Tensor4 
		 *
		 * Description		: Default constructor which sets the dimensions of the tensor to be 0
		 * ==================================================================================================
		 */
		explicit Tensor4() :
			x_(0), y_(0), z_(0), w_(0) {}

		/*
		 * ==================================================================================================
		 * Function			: Tensor4 (constructor)
		 *
		 * Description		: Sets the number of elements in each dimension of the tensor and allocates and 
		 *					  sets the tensor data to be zero
		 *
		 * Inputs			: _x	: Number of elements in the 1st dimension
		 *					: _y	: Number of elements in the 2nd dimension
		 *					: _z	: Number of elements in the 3rd dimension
		 *					: _w	: Number of elements in the 4th dimension
		 * ==================================================================================================
		 */
		Tensor4(uint _x, uint _y, uint _z, uint _w) :
			x_(_x), y_(_y), z_(_z), w_(_w), data(_x * _y * _z * _w, 0) {}

        /*
         * ==================================================================================================
         * Function         : getData 
         * 
         * Description      : Gets the data vector from the Tesnor
         * 
         * Outputs          : The data vector for the Tensor (to support moving)
         * ==================================================================================================
         */
        inline std::vector<dType>& getData() { return data; }
        
        /*
         * ==================================================================================================
         * Function         : moveData
         * 
         * Description      : Moves the data from another tensor into this Tensor4
         *  
         * Inputs           : otherTensor   : The tensor from which the data must be moved from
         * ==================================================================================================
         */
        inline void moveData(Tensor4<dType>& otherTensor) {
            otherTensor.getData() = std::move(data);
        }
         
		/* ==================================================================================================
		 * Function		: size
		 *
		 * Description	: Retuns the size of the tensor (total number of elements)
		 * ==================================================================================================
		 */
		__inline__ __device__ __host__ size_t size() const {
			return data.size();
		}

		__inline__ __device__ __host__ uint x() const { return x_; }
		__inline__ __device__ __host__ uint y() const { return y_; }
		__inline__ __device__ __host__ uint z() const { return z_; }
		__inline__ __device__ __host__ uint w() const { return w_; }

		/*
		 * ==================================================================================================
		 * Function		: reshape 
		 *
		 * Description	: Reshapes the tensor along each dimension, -1 keeps the dimensionality 
		 *
		 * Inputs		: x_new		: New number of elements for 1st dimension
		 *				: y_new		: New number of elements for 2nd dimension
		 *				: z_new		: New number of elements for 3rd dimension
		 *				: w_new		: New number of elements for 4th dimension
		 * ==================================================================================================
		 */
		__inline__ __device__ __host__ void reshape(int x_new, int y_new, int z_new, int w_new) {
			x_ = (x_new != -1) ? static_cast<uint>(x_new) : x_;		
			y_ = (y_new != -1) ? static_cast<uint>(y_new) : y_;		
			z_ = (z_new != -1) ? static_cast<uint>(z_new) : z_;		
			w_ = (w_new != -1) ? static_cast<uint>(w_new) : w_;		
			data.resize(w_ * x_ * y_ * z_, 0);
		}

		/*
		 * ==================================================================================================
		 * Function		: operator() 
		 *
		 * Description	: Overload () operator to get a specific element
		 * Params		: dType		: The data type for the matrix
		 *				: x_new		: Element position in 1st (x) dimension
		 *				: y_new		: Element position in 2nd (y) dimension
		 *				: z_new		: Element position in 3rd (z) dimension
		 *				: w_new		: Element position in 4th (w) dimension
		 * ==================================================================================================
		 */
		dType& operator() (uint x_elem, uint y_elem, uint z_elem, uint w_elem) {
			int error = 0;
			if (x_elem < 0 || x_elem >= x_) error = -1;
			if (y_elem < 0 || y_elem >= y_) error = -2;
			if (z_elem < 0 || z_elem >= z_) error = -3;
			if (w_elem < 0 || w_elem >= w_) error = -4;

			switch (error) {
				case -1:
					std::cerr << "Out of Range Error : Element " << x_elem << 
						         " out of range of dimension 1 (x) for tensor : Returning first element\n";
					return data[0];
				case -2:
					std::cerr << "Out of Range Error : Element " << y_elem << 
						         " out of range of dimension 2 (y) for tensor : Returning first element\n";
					return data[0];
				case -3:
					std::cerr << "Out of Range Error : Element " << z_elem << 
						         " out of range of dimension 3 (z) for tensor : Returning first element\n";
					return data[0];
				case -4:
					std::cerr << "Out of Range Error : Element " << w_elem << 
						         " out of range of dimension 4 (w) for tensor : Returning first element\n";
					return data[0];
			}
			int offset = x_ * y_ * z_ * w_elem	+			// 4th dimension offset
				         x_ * y_ * z_elem		+			// 3rd dimension offset
						 x_ * y_elem			+			// 2nd dimension offset
						 x_elem;						    // 1st dimension offset
			return 	data[offset];
		}

		/*
		 * ==================================================================================================
		 * Function		: operator() 
		 *
		 * Description	: Overload () operator to set a specific element
		 * Params		: dType		: The data type for the matrix
		 *				: x_new		: Element position in 1st (x) dimension
		 *				: y_new		: Element position in 2nd (y) dimension
		 *				: z_new		: Element position in 3rd (z) dimension
		 *				: w_new		: Element position in 4th (w) dimension
		 * ==================================================================================================
		 */
		dType const& operator()(uint x_elem, uint y_elem, uint z_elem, uint w_elem) const {
			int error = 0;
			if (x_elem < 0 || x_elem >= x_) error = -1;
			if (y_elem < 0 || y_elem >= y_) error = -2;
			if (z_elem < 0 || z_elem >= z_) error = -3;
			if (w_elem < 0 || w_elem >= w_) error = -4;

			switch (error) {
				case -1:
					std::cerr << "Out of Range Error : Element " << x_elem << 
						         " out of range of dimension 1 (x) for tensor : Returning first element\n";
					return data[0];
				case -2:
					std::cerr << "Out of Range Error : Element " << y_elem << 
						         " out of range of dimension 2 (y) for tensor : Returning first element\n";
					return data[0];
				case -3:
					std::cerr << "Out of Range Error : Element " << z_elem << 
						         " out of range of dimension 3 (z) for tensor : Returning first element\n";
					return data[0];
				case -4:
					std::cerr << "Out of Range Error : Element " << w_elem << 
						         " out of range of dimension 4 (w) for tensor : Returning first element\n";
					return data[0];
			}
			int offset = x_ * y_ * z_ * w_elem	+			// 4th dimension offset
				         x_ * y_ * z_elem		+			// 3rd dimension offset
						 x_ * y_elem			+			// 2nd dimension offset
						 x_elem;						// 1st dimension offset
			return data[offset];
		}
};

}
#endif
