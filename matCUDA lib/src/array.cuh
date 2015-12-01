#ifndef ARRAY_CUDA_H
#define ARRAY_CUDA_H

#include <cuComplex.h>

#include "common.h"

// Utilities and system includes
//#include <assert.h>

#ifndef DEVICE_ID
#define DEVICE_ID 0
#endif

//template <typename TElement> class Array;

//typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
//{
//    unsigned int Acols, Arows, Bcols, Brows, Ccols, Crows;
//} sMatrixSize;

template<typename T>
cudaError_t add(	T *c,
					const T *a,
					const T *b,
					__int32 size,
					Type2Type<T>);

template<typename T>
cudaError_t minus(	T *c,
					const T *a,
					const T *b,
					__int32 size,
					Type2Type<T>);

template<typename T>
cudaError_t equal(	T *c,
					const T *a,
					__int32 size,
					Type2Type<T>);

template<typename T>
cudaError_t multiply(	T *c,
						const T *a,
						const T *b,
						__int32 width_a,
						__int32 height_a,
						__int32 width_b,
						Type2Type<T>);

cudaError_t conjugate(	ComplexFloat *c,
						const ComplexFloat *a,
						__int32 size,
						Type2Type<ComplexFloat>);

cudaError_t conjugate(	ComplexDouble *c,
						const ComplexDouble *a,
						__int32 size,
						Type2Type<ComplexDouble>);

template<typename T>
cudaError_t transp(	T *c,
					const T *a,
					__int32 dim,
					Type2Type<T>);

template<typename T>
cudaError_t determinant(T *c,
						const T *dev_a,
						__int32 rows);

template <typename T>
__host__ void zeros_under_diag(T *a, __int32 size);

template <typename T>
__global__ void zeros_under_diag_kernel(T *a, __int32 size);

//__global__ void zeros_under_diag_kernel(float *a, __int32 size);
//__global__ void zeros_under_diag_kernel(double *a, __int32 size);
//__global__ void zeros_under_diag_kernel(cuComplex *a, __int32 size);
//__global__ void zeros_under_diag_kernel(cuDoubleComplex *a, __int32 size);

template <typename T>
__host__ void zeros_above_diag(T *a, __int32 size);

template <typename T>
__global__ void zeros_above_diag_kernel(T *a, __int32 size);

//__global__ void zeros_above_diag_kernel(float *a, __int32 size);
//__global__ void zeros_above_diag_kernel(double *a, __int32 size);
//__global__ void zeros_above_diag_kernel(cuComplex *a, __int32 size);
//__global__ void zeros_above_diag_kernel(cuDoubleComplex *a, __int32 size);

template <typename T>
__host__ void cuda_eye(T *a, __int32 size);

template <typename T>
__global__ void cuda_eye_kernel(T *a, __int32 size);

template <typename T>
__host__ T cuda_min( T *a, int *idx, __int32 size );

template <typename T>
__host__ T cuda_max( T *a, int *idx, __int32 size );

template <typename T>
__host__ void cuda_elementwise_multiplication( T *a, T *b, T *c, size_t N );

template <typename T>
__global__ void cuda_elementwise_multiplication_kernel( T *a, T *b, T *c, size_t N );

template <typename T>
__host__ void cuda_elementwise_division( T *a, T *b, T *c, size_t N );

template <typename T>
__global__ void cuda_elementwise_division_kernel( T *a, T *b, T *c, size_t N );

#endif