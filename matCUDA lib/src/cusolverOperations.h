#ifndef CUSOLVEROPERATIONS_H
#define CUSOLVEROPERATIONS_H

#include <cusolverSp.h>
#include <cusolverDn.h>

#include "matCUDA.h"

namespace matCUDA
{
	template <typename TElement>
	class cusolverOperations
	{	
		template <typename TElement> friend class Array;

	public:

		cusolverStatus_t dpss( Array<TElement> *eigenvector, index_t N, double NW, index_t degree );

		cusolverStatus_t QR( Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R );
		cusolverStatus_t QR_zerocopy( Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R );


	private:

		cusolverStatus_t cusolverSpTcsreigvsi( cusolverSpHandle_t *handle, int m, int nnz, cusparseMatDescr_t *descrA, const TElement *csrValA, const int *csrRowPtrA, const int *csrColIndA, TElement mu0, const TElement *x0, int maxite, TElement tol, TElement *mu, TElement *x );
		
		cusolverStatus_t cusolverDnTgeqrf_bufferSize( cusolverDnHandle_t *handle, int m, int n, TElement *A, int lda, int *Lwork );
		cusolverStatus_t cusolverDnTgeqrf( cusolverDnHandle_t *handle, int m, int n, TElement *A, int lda, TElement *TAU, TElement *Workspace, int Lwork, int *devInfo );
		cusolverStatus_t cusolverDnTormqr( cusolverDnHandle_t *handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const TElement *A, int lda, const TElement *tau, TElement *C, int ldc, TElement *work, int lwork, int *devInfo );
		
	};
}

#endif