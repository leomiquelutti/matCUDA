#ifndef CUSOLVEROPERATIONS_H
#define CUSOLVEROPERATIONS_H

#include <cusolverSp.h>

#include "matCUDA.h"

namespace matCUDA
{
	template <typename TElement>
	class cusolverOperations
	{	
		template <typename TElement> friend class Array;

	public:

		cusolverStatus_t dpss( Array<TElement> *eigenvector, index_t N, double NW, index_t degree );

	private:

		cusolverStatus_t dpss_type( cusolverSpHandle_t *handle, int m, int nnz, cusparseMatDescr_t *descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, float mu0, const float *d_eigenvector0, int maxite, float tol, float *d_mu, float *d_eigenvector );
		cusolverStatus_t dpss_type( cusolverSpHandle_t *handle, int m, int nnz, cusparseMatDescr_t *descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, double mu0, const double *d_eigenvector0, int maxite, double tol, double *d_mu, double *d_eigenvector );
	};
}

#endif