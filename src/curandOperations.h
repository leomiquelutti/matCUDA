#ifndef CURANDOPERATIONS_H
#define CURANDOPERATIONS_H

#include <curand.h>

#include "Array.h"

namespace matCUDA
{
	template <typename TElement>
	class curandOperations
	{	
		template <typename TElement> friend class Array;

	public:

		curandStatus_t rand( Array<TElement> *out );
	};
}

#endif