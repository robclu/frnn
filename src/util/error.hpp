#ifndef CURNN_ERROR_INCLUDED
#define CURNN_ERROR_INCLUDED

#include <iostream>

using namespace std;

namespace currn {
	namespace util {
		namespace err {
			/*
			 * =====================================================================
			 * Function		: allocError
			 *
			 * Description	: Prints an error message of allocation failed
			 * =====================================================================
			 */
			inline void allocError( char * varname ) {
				cerr << "Error allocating memory for " << varname << endl;
			}
		}
	}
}
#endif

