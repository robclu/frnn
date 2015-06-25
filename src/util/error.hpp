#ifndef CURNN_UTIL_ERROR_INCLUDED
#define CURNN_UTIL_ERROR_INCLUDED

#include <iostream>

using namespace std;

namespace currn {
	namespace util {
		namespace err {
			/*
			 * ==============================================================================================
			 * Function		: allocError
			 *
			 * Description	: Prints an error message of allocation failed
			 *
			 * Inputs		: varname	: The name of the variable that could not be allocated
			 * ==============================================================================================
			 */
			inline void allocError( char * varname ) {
				cerr << "Error allocating memory for " << varname << "\n";
			}

			/*
			 * ==============================================================================================
			 * Function		: copyError
			 *
			 * Description	: Prints an error message if host to device, or device to host copy failed
			 *
			 * Inputs		: varname	: The name of the variable that could not be copied to/from
			 * ==============================================================================================
			 */
			inline void copyError( char* varname ) {
				cerr << "Error copying to/from variable " << varname << "\n";
			}
		}
	}
}

#endif

