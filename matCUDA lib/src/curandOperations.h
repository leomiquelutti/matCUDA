#ifndef CURANDOPERATIONS_H
#define CURANDOPERATIONS_H

#include <curand.h>

#include "matCUDA.h"

namespace matCUDA
{
	template <typename TElement>
	class curandOperations
	{	
		template <typename TElement> friend class Array;

	public:

		curandStatus_t rand( Array<TElement> *out );
		curandStatus_t rand_zerocopy( Array<TElement> *out );
		curandStatus_t rand_stream( Array<TElement> *out );
	};
}

#endif