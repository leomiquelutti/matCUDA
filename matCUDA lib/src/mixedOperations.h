#ifndef MIXEDOPERATIONS_H
#define MIXEDOPERATIONS_H

#include <cublas.h>
#include <cublas_v2.h>
#include <cublasXt.h>

#include <cusolverSp.h>
#include <cusolverDn.h>

#include "matCUDA.h"
#include "cusolverOperations.h"
#include "cublasOperations.h"

#ifndef EIG_MAX_ITER
#define EIG_MAX_ITER 500
#endif

namespace matCUDA
{
	template <typename TElement>
	class mixedOperations
	{	

	public:
		
		cublasStatus_t eig( Array<TElement> *A, Array<TElement> *eigvec );

	private:
	};
}

#endif