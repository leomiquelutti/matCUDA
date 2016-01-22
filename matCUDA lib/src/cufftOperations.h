#ifndef CUFFTOPERATIONS_H
#define CUFFTOPERATIONS_H

#include <cufft.h>

#include "matCUDA.h"

namespace matCUDA
{
	template <typename TElement>
	class cufftOperations
	{	

	public:

		cufftResult_t fft( Array<TElement> *in, Array<TElement> *out );
		cufftResult_t fft_stream( Array<TElement> *in, Array<TElement> *out );
		cufftResult_t fft_zerocopy( Array<TElement> *in, Array<TElement> *out );

		cufftResult_t fft( Array<float> *in, Array<ComplexFloat> *out );
		cufftResult_t fft_stream( Array<float> *in, Array<ComplexFloat> *out );
		cufftResult_t fft_zerocopy( Array<float> *in, Array<ComplexFloat> *out );

		cufftResult_t fft( Array<double> *in, Array<ComplexDouble> *out );
		cufftResult_t fft_stream( Array<double> *in, Array<ComplexDouble> *out );
		cufftResult_t fft_zerocopy( Array<double> *in, Array<ComplexDouble> *out );

	};
}

#endif