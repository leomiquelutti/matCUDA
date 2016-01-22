#include <thrust\complex.h>
#include <thrust\device_ptr.h>
#include <thrust\extrema.h>

#include "array.cuh"

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

typedef thrust::complex<float> DeviceComplexFloat;
typedef thrust::complex<double> DeviceComplexDouble;

typedef DeviceComplexFloat DeviceComplex;

#define BLOCK_ROWS 4

template <typename T, typename U = T>
struct OperationData
{
	const T *a;
	const T *b;
	T *c;

	U **dev_a;
	U **dev_b;
	U **dev_c;

	__int32 size;
	__int32 size_b;
	__int32 size_c;
};

template <typename T, typename U = T>
struct Matrix
{
	__int32 width;
	__int32 height;
	T *elements;
	__int32 stride;
};

template<typename T>
__global__ void add(T *c,
					const T *a,
					const T *b,
					__int32 size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size)
	{
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

template<typename T>
__global__ void minus(	T *c,
						const T *a,
						const T *b,
						__int32 size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size)
	{
        c[tid] = a[tid] - b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

template<typename T>
__global__ void equal(	T *c,
						const T *a,
						__int32 size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size)
	{
        c[tid] = a[tid];
        tid += blockDim.x * gridDim.x;
    }
}

template<typename T>
__global__ void conj(	T *c,
						const T *a,
						__int32 size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size)
	{
		c[tid].real(a[tid].real());
		c[tid].imag(a[tid].imag() * -1);
        tid += blockDim.x * gridDim.x;
    }
}

template<typename T>
__global__ void transpose(	T *odata,
							const T *idata,
							const int blockRows )
{
	__shared__ T tile[TILE_DIM][TILE_DIM+1];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int C = blockIdx.y * TILE_DIM + threadIdx.y;
  int w = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(C+j)*w + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  C = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(C+j)*w + x] = tile[threadIdx.x][threadIdx.y + j];

}

template<typename T>
__global__ void multiply(Matrix<T> A, Matrix<T> B, Matrix<T> C)
{
	T Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > A.height || col > B.width) return;
	for (int e = 0; e < A.width; ++e)
		Cvalue += (A.elements[row * A.width + e]) * (B.elements[e * B.width + col]);
	C.elements[row * C.width + col] = Cvalue;
}

template<typename T>
__global__ void det(T *c,
					const T *a,
					__int32 rows)
{
	T result = (T)1;
	int size = rows*rows;
	
	int tid = 0;
	int row = 1;
	while(tid < size)
	{
		result *= a[tid];
		tid = tid + (row++) * (rows + 1);
	}

	c[0] = result;
}

template<typename T, typename U>
__host__ cudaError_t allocMem(struct OperationData<T, U> *pdata)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)(pdata->dev_a), pdata->size*sizeof(T));
	if(cudaStatus != cudaSuccess)
		return cudaStatus;

	if(pdata->b != NULL)
	{
		cudaStatus = cudaMalloc((void**)(pdata->dev_b), ((pdata->size_b)?pdata->size_b:pdata->size)*sizeof(T));
		if(cudaStatus != cudaSuccess)
			return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)(pdata->dev_c), ((pdata->size_c)?pdata->size_c:pdata->size)*sizeof(T));
	if(cudaStatus != cudaSuccess)
		return cudaStatus;

	return cudaStatus;
}

template<typename T, typename U>
__host__ cudaError_t copyHostDevice(struct OperationData<T, U> *pdata)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMemcpyAsync(*(pdata->dev_a), pdata->a, pdata->size*sizeof(T), cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess)
		return cudaStatus;

	if(pdata->dev_b != NULL)
	{
		cudaStatus = cudaMemcpyAsync(*(pdata->dev_b), pdata->b, ((pdata->size_b)?pdata->size_b:pdata->size)*sizeof(T), cudaMemcpyHostToDevice);
		if(cudaStatus != cudaSuccess)
			return cudaStatus;
	}
	
	return cudaStatus;
}

template<typename T, typename U>
__host__ cudaError_t copyDeviceHost(struct OperationData<T, U> *pdata)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMemcpyAsync(pdata->c, *(pdata->dev_c), ((pdata->size_c)?pdata->size_c:pdata->size)*sizeof(T), cudaMemcpyDeviceToHost);
	if(cudaStatus != cudaSuccess)
		return cudaStatus;

	return cudaStatus;
}

template<typename T>
__host__ void deallocate(struct OperationData<T> *pdata)
{
	cudaFree(*(pdata->dev_a));

	if(pdata->dev_b != NULL)
		cudaFree(*(pdata->dev_b));

	cudaFree(*(pdata->dev_c));
}

template<typename T, typename U>
__host__ void deallocate(struct OperationData<T, U> *pdata)
{
	cudaFree(*(pdata->dev_a));
	
	if(pdata->dev_b != NULL)
		cudaFree(*(pdata->dev_b));
	
	cudaFree(*(pdata->dev_c));
}

template<typename T>
__host__ cudaError_t add(T *c,
						 const T *a,
						 const T *b,
						 __int32 size,
						 Type2Type<T>)
{
	T *dev_a = 0;
    T *dev_b = 0;
    T *dev_c = 0;
    cudaError_t cudaStatus;

	// TODO Use a ctor
	struct OperationData<T> data = {0};
	data.a = a;
	data.b = b;
	data.c = c;
	data.dev_a = &dev_a;
	data.dev_b = &dev_b;
	data.dev_c = &dev_c;
	data.size = size;

	const int threadsPerBlock = 128;
	const int blocksPerGrid = imin((size+threadsPerBlock-1)/threadsPerBlock,32);

	try
	{
		cudaStatus = allocMem(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		cudaStatus = copyHostDevice(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		add<T><<<blocksPerGrid , threadsPerBlock>>>(dev_c, dev_a, dev_b, size);
	
		cudaStatus = copyDeviceHost(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		deallocate(&data);
	}
	catch(...)
	{
		deallocate(&data);
    	throw;
	}

	return cudaStatus;
}

template __host__ cudaError_t add<float>(float *c, const float *a, const float *b, __int32 size, Type2Type<float>);
template __host__ cudaError_t add<double>(double *c, const double *a, const double *b, __int32 size, Type2Type<double>);
template __host__ cudaError_t add<int>(int *c, const int *a, const int *b, __int32 size, Type2Type<int>);

__host__ cudaError_t add(ComplexFloat *c, const ComplexFloat *a, const ComplexFloat *b, __int32 size, Type2Type<ComplexFloat>)
{
	DeviceComplexFloat *dev_a = 0;
    DeviceComplexFloat *dev_b = 0;
    DeviceComplexFloat *dev_c = 0;
    cudaError_t cudaStatus;

	// TODO Use a ctor
	struct OperationData<ComplexFloat, DeviceComplexFloat> data = {0};
	data.a = a;
	data.b = b;
	data.c = c;
	data.dev_a = &dev_a;
	data.dev_b = &dev_b;
	data.dev_c = &dev_c;
	data.size = size;

	const int threadsPerBlock = 128;
	const int blocksPerGrid = imin((size+threadsPerBlock-1)/threadsPerBlock,32);

	try
	{
		cudaStatus = allocMem(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		cudaStatus = copyHostDevice(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		add<DeviceComplexFloat><<<blocksPerGrid , threadsPerBlock>>>(dev_c, dev_a, dev_b, size);
	
		cudaStatus = copyDeviceHost(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		deallocate(&data);
	}
	catch(...)
	{
		deallocate(&data);
    	throw;
	}

	return cudaStatus;
}

__host__ cudaError_t add(ComplexDouble *c, const ComplexDouble *a, const ComplexDouble *b, __int32 size, Type2Type<ComplexDouble>)
{
	DeviceComplexDouble *dev_a = 0;
    DeviceComplexDouble *dev_b = 0;
    DeviceComplexDouble *dev_c = 0;
    cudaError_t cudaStatus;

	// TODO Use a ctor
	struct OperationData<ComplexDouble, DeviceComplexDouble> data = {0};
	data.a = a;
	data.b = b;
	data.c = c;
	data.dev_a = &dev_a;
	data.dev_b = &dev_b;
	data.dev_c = &dev_c;
	data.size = size;

	const int threadsPerBlock = 128;
	const int blocksPerGrid = imin((size+threadsPerBlock-1)/threadsPerBlock,32);

	try
	{
		cudaStatus = allocMem(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		cudaStatus = copyHostDevice(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		add<DeviceComplexDouble><<<blocksPerGrid , threadsPerBlock>>>(dev_c, dev_a, dev_b, size);
	
		cudaStatus = copyDeviceHost(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		deallocate(&data);
	}
	catch(...)
	{
		deallocate(&data);
    	throw;
	}

	return cudaStatus;
}

template<typename T>
__host__ cudaError_t minus(	T *c,
							const T *a,
							const T *b,
							__int32 size,
							Type2Type<T>)
{
	T *dev_a = 0;
    T *dev_b = 0;
    T *dev_c = 0;
    cudaError_t cudaStatus;

	// TODO Use a ctor
	struct OperationData<T> data = {0};
	data.a = a;
	data.b = b;
	data.c = c;
	data.dev_a = &dev_a;
	data.dev_b = &dev_b;
	data.dev_c = &dev_c;
	data.size = size;

	const int threadsPerBlock = 128;
	const int blocksPerGrid = imin((size+threadsPerBlock-1)/threadsPerBlock,32);

	try
	{
		cudaStatus = allocMem(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		cudaStatus = copyHostDevice(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		minus<T><<<blocksPerGrid , threadsPerBlock>>>(dev_c, dev_a, dev_b, size);
	
		cudaStatus = copyDeviceHost(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		deallocate(&data);
	}
	catch(...)
	{
		deallocate(&data);
    	throw;
	}

	return cudaStatus;
}

template __host__ cudaError_t minus<float>(float *c, const float *a, const float *b, __int32 size, Type2Type<float>);
template __host__ cudaError_t minus<double>(double *c, const double *a, const double *b, __int32 size, Type2Type<double>);
template __host__ cudaError_t minus<int>(int *c, const int *a, const int *b, __int32 size, Type2Type<int>);

__host__ cudaError_t minus(ComplexFloat *c, const ComplexFloat *a, const ComplexFloat *b, __int32 size, Type2Type<ComplexFloat>)
{
	DeviceComplexFloat *dev_a = 0;
    DeviceComplexFloat *dev_b = 0;
    DeviceComplexFloat *dev_c = 0;
    cudaError_t cudaStatus;

	// TODO Use a ctor
	struct OperationData<ComplexFloat, DeviceComplexFloat> data = {0};
	data.a = a;
	data.b = b;
	data.c = c;
	data.dev_a = &dev_a;
	data.dev_b = &dev_b;
	data.dev_c = &dev_c;
	data.size = size;

	const int threadsPerBlock = 128;
	const int blocksPerGrid = imin((size+threadsPerBlock-1)/threadsPerBlock,32);

	try
	{
		cudaStatus = allocMem(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		cudaStatus = copyHostDevice(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		minus<DeviceComplexFloat><<<blocksPerGrid , threadsPerBlock>>>(dev_c, dev_a, dev_b, size);
	
		cudaStatus = copyDeviceHost(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		deallocate(&data);
	}
	catch(...)
	{
		deallocate(&data);
    	throw;
	}

	return cudaStatus;
}

__host__ cudaError_t minus(ComplexDouble *c, const ComplexDouble *a, const ComplexDouble *b, __int32 size, Type2Type<ComplexDouble>)
{
	DeviceComplexDouble *dev_a = 0;
    DeviceComplexDouble *dev_b = 0;
    DeviceComplexDouble *dev_c = 0;
    cudaError_t cudaStatus;

	// TODO Use a ctor
	struct OperationData<ComplexDouble, DeviceComplexDouble> data = {0};
	data.a = a;
	data.b = b;
	data.c = c;
	data.dev_a = &dev_a;
	data.dev_b = &dev_b;
	data.dev_c = &dev_c;
	data.size = size;

	const int threadsPerBlock = 128;
	const int blocksPerGrid = imin((size+threadsPerBlock-1)/threadsPerBlock,32);

	try
	{
		cudaStatus = allocMem(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		cudaStatus = copyHostDevice(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		minus<DeviceComplexDouble><<<blocksPerGrid , threadsPerBlock>>>(dev_c, dev_a, dev_b, size);
	
		cudaStatus = copyDeviceHost(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		deallocate(&data);
	}
	catch(...)
	{
		deallocate(&data);
    	throw;
	}

	return cudaStatus;
}

template<typename T>
__host__ cudaError_t equal(	T *c,
							const T *a,
							__int32 size,
							Type2Type<T>)
{
	T *dev_a = 0;
    T *dev_c = 0;
    cudaError_t cudaStatus;

	// TODO Use a ctor
	struct OperationData<T> data = {0};
	data.a = a;
	data.b = NULL;
	data.c = c;
	data.dev_a = &dev_a;
	data.dev_b = NULL;
	data.dev_c = &dev_c;
	data.size = size;

	const int threadsPerBlock = 128;
	const int blocksPerGrid = imin((size+threadsPerBlock-1)/threadsPerBlock,32);

	try
	{
		cudaStatus = allocMem(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		cudaStatus = copyHostDevice(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		equal<T><<<blocksPerGrid , threadsPerBlock>>>(dev_c, dev_a, size);
	
		cudaStatus = copyDeviceHost(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		deallocate(&data);
	}
	catch(...)
	{
		deallocate(&data);
    	throw;
	}

	return cudaStatus;
}

template __host__ cudaError_t equal<float>(float *c, const float *a, __int32 size, Type2Type<float>);
template __host__ cudaError_t equal<double>(double *c, const double *a, __int32 size, Type2Type<double>);
template __host__ cudaError_t equal<int>(int *c, const int *a, __int32 size, Type2Type<int>);

__host__ cudaError_t equal(ComplexFloat *c, const ComplexFloat *a, __int32 size, Type2Type<ComplexFloat>)
{
	DeviceComplexFloat *dev_a = 0;
    DeviceComplexFloat *dev_c = 0;
    cudaError_t cudaStatus;

	// TODO Use a ctor
	struct OperationData<ComplexFloat, DeviceComplexFloat> data = {0};
	data.a = a;
	data.b = NULL;
	data.c = c;
	data.dev_a = &dev_a;
	data.dev_b = NULL;
	data.dev_c = &dev_c;
	data.size = size;

	const int threadsPerBlock = 128;
	const int blocksPerGrid = imin((size+threadsPerBlock-1)/threadsPerBlock,32);

	try
	{
		cudaStatus = allocMem(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		cudaStatus = copyHostDevice(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		equal<DeviceComplexFloat><<<blocksPerGrid , threadsPerBlock>>>(dev_c, dev_a, size);
	
		cudaStatus = copyDeviceHost(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		deallocate(&data);
	}
	catch(...)
	{
		deallocate(&data);
    	throw;
	}

	return cudaStatus;
}

__host__ cudaError_t equal(ComplexDouble *c, const ComplexDouble *a, __int32 size, Type2Type<ComplexDouble>)
{
	DeviceComplexDouble *dev_a = 0;
    DeviceComplexDouble *dev_c = 0;
    cudaError_t cudaStatus;

	// TODO Use a ctor
	struct OperationData<ComplexDouble, DeviceComplexDouble> data = {0};
	data.a = a;
	data.b = NULL;
	data.c = c;
	data.dev_a = &dev_a;
	data.dev_b = NULL;
	data.dev_c = &dev_c;
	data.size = size;

	const int threadsPerBlock = 128;
	const int blocksPerGrid = imin((size+threadsPerBlock-1)/threadsPerBlock,32);

	try
	{
		cudaStatus = allocMem(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		cudaStatus = copyHostDevice(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		equal<DeviceComplexDouble><<<blocksPerGrid , threadsPerBlock>>>(dev_c, dev_a, size);
	
		cudaStatus = copyDeviceHost(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		deallocate(&data);
	}
	catch(...)
	{
		deallocate(&data);
    	throw;
	}

	return cudaStatus;
}

__host__ cudaError_t conjugate(ComplexFloat *c, const ComplexFloat *a, __int32 size, Type2Type<ComplexFloat>)
{
	DeviceComplexFloat *dev_a = 0;
    DeviceComplexFloat *dev_c = 0;
    cudaError_t cudaStatus;

	// TODO Use a ctor
	struct OperationData<ComplexFloat, DeviceComplexFloat> data = {0};
	data.a = a;
	data.b = NULL;
	data.c = c;
	data.dev_a = &dev_a;
	data.dev_b = NULL;
	data.dev_c = &dev_c;
	data.size = size;

	const int threadsPerBlock = 128;
	const int blocksPerGrid = imin((size+threadsPerBlock-1)/threadsPerBlock,32);

	try
	{
		cudaStatus = allocMem(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		cudaStatus = copyHostDevice(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		conj<DeviceComplexFloat><<<blocksPerGrid , threadsPerBlock>>>(dev_c, dev_a, size);
	
		cudaStatus = copyDeviceHost(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		deallocate(&data);
	}
	catch(...)
	{
		deallocate(&data);
    	throw;
	}

	return cudaStatus;
}

__host__ cudaError_t conjugate(ComplexDouble *c, const ComplexDouble *a, __int32 size, Type2Type<ComplexDouble>)
{
	DeviceComplexDouble *dev_a = 0;
    DeviceComplexDouble *dev_c = 0;
    cudaError_t cudaStatus;

	// TODO Use a ctor
	struct OperationData<ComplexDouble, DeviceComplexDouble> data = {0};
	data.a = a;
	data.b = NULL;
	data.c = c;
	data.dev_a = &dev_a;
	data.dev_b = NULL;
	data.dev_c = &dev_c;
	data.size = size;

	const int threadsPerBlock = 128;
	const int blocksPerGrid = imin((size+threadsPerBlock-1)/threadsPerBlock,32);

	try
	{
		cudaStatus = allocMem(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		cudaStatus = copyHostDevice(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		conj<DeviceComplexDouble><<<blocksPerGrid , threadsPerBlock>>>(dev_c, dev_a, size);
	
		cudaStatus = copyDeviceHost(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		deallocate(&data);
	}
	catch(...)
	{
		deallocate(&data);
    	throw;
	}

	return cudaStatus;
}

template<typename T>
__host__ cudaError_t transp(T *c,
							const T *a,
							__int32 dim,
							Type2Type<T>)
{
	T *dev_a = 0;
    T *dev_c = 0;
    cudaError_t cudaStatus;

	// TODO Use a ctor
	struct OperationData<T> data = {0};
	data.a = a;
	data.b = NULL;
	data.c = c;
	data.dev_a = &dev_a;
	data.dev_b = NULL;
	data.dev_c = &dev_c;
	data.size = dim*dim;

	dim3 grid((dim+(TILE_DIM-1))/TILE_DIM, (dim+(TILE_DIM-1))/TILE_DIM, 1);
	dim3 threads(TILE_DIM, BLOCK_ROWS, 1);

	try
	{
		cudaStatus = allocMem(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		cudaStatus = copyHostDevice(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		transpose<T><<<grid, threads>>>(dev_c, dev_a, BLOCK_ROWS);

		cudaStatus = copyDeviceHost(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		deallocate(&data);
	}
	catch(...)
	{
		deallocate(&data);
    	throw;
	}

	return cudaStatus;
}

template __host__ cudaError_t transp<float>(float *c, const float *a, __int32 dim, Type2Type<float>);
template __host__ cudaError_t transp<double>(double *c, const double *a, __int32 dim, Type2Type<double>);
template __host__ cudaError_t transp<int>(int *c, const int *a, __int32 dim, Type2Type<int>);

__host__ cudaError_t transp(ComplexFloat *c, const ComplexFloat *a, __int32 dim, Type2Type<ComplexFloat>)
{
	DeviceComplexFloat *dev_a = 0;
    DeviceComplexFloat *dev_c = 0;
    cudaError_t cudaStatus;

	// TODO Use a ctor
	struct OperationData<ComplexFloat, DeviceComplexFloat> data = {0};
	data.a = a;
	data.b = NULL;
	data.c = c;
	data.dev_a = &dev_a;
	data.dev_b = NULL;
	data.dev_c = &dev_c;
	data.size = dim*dim;

	dim3 grid((dim+(TILE_DIM-1))/TILE_DIM, (dim+(TILE_DIM-1))/TILE_DIM, 1);
	dim3 threads(TILE_DIM, BLOCK_ROWS, 1);

	try
	{
		cudaStatus = allocMem(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		cudaStatus = copyHostDevice(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		transpose<DeviceComplexFloat><<<grid, threads>>>(dev_c, dev_a, BLOCK_ROWS);
	
		cudaStatus = copyDeviceHost(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		deallocate(&data);
	}
	catch(...)
	{
		deallocate(&data);
    	throw;
	}

	return cudaStatus;
}

__host__ cudaError_t transp(ComplexDouble *c, const ComplexDouble *a, __int32 dim, Type2Type<ComplexDouble>)
{
	DeviceComplexDouble *dev_a = 0;
    DeviceComplexDouble *dev_c = 0;
    cudaError_t cudaStatus;

	// TODO Use a ctor
	struct OperationData<ComplexDouble, DeviceComplexDouble> data = {0};
	data.a = a;
	data.b = NULL;
	data.c = c;
	data.dev_a = &dev_a;
	data.dev_b = NULL;
	data.dev_c = &dev_c;
	data.size = dim*dim;

	dim3 grid((dim+(TILE_DIM-1))/TILE_DIM, (dim+(TILE_DIM-1))/TILE_DIM, 1);
	dim3 threads(TILE_DIM, BLOCK_ROWS, 1);

	try
	{
		cudaStatus = allocMem(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		cudaStatus = copyHostDevice(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		transpose<DeviceComplexDouble><<<grid, threads>>>(dev_c, dev_a, BLOCK_ROWS);
	
		cudaStatus = copyDeviceHost(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		deallocate(&data);
	}
	catch(...)
	{
		deallocate(&data);
    	throw;
	}

	return cudaStatus;
}

template<typename T>
__host__ cudaError_t multiply(	T *c,
								const T *a,
								const T *b,
								__int32 width_a,
								__int32 height_a,
								__int32 width_b,
								Type2Type<T>)
{
	cudaError_t cudaStatus;

	struct Matrix<T> A, B, C;
	A.width = width_a;
	A.height = height_a;
	B.width = width_b;
	B.height = A.width;
	C.height = A.height;
	C.width = B.width;
	
	struct OperationData<T> data = {0};
	data.a = a;
	data.b = b;
	data.c = c;
	data.dev_a = &(A.elements);
	data.dev_b = &(B.elements);
	data.dev_c = &(C.elements);
	data.size = A.width*A.height;
	data.size_b = B.width*B.height;
	data.size_c = C.width*C.height;

	dim3 threads(16, 16);
	dim3 grid(	(B.width + threads.x - 1) / threads.x,
				(A.height + threads.y - 1) / threads.y);
	try
	{
		cudaStatus = allocMem(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		cudaStatus = copyHostDevice(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		multiply<T><<<grid, threads>>>(A, B, C);
	
		cudaStatus = copyDeviceHost(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		deallocate(&data);
	}
	catch(...)
	{
		deallocate(&data);
    	throw;
	}

	return cudaStatus;
}

template __host__ cudaError_t multiply<float>(float *c, const float *a, const float *b, __int32 width_a, __int32 height_a, __int32 width_b, Type2Type<float>);
template __host__ cudaError_t multiply<double>(double *c, const double *a, const double *b, __int32 width_a, __int32 height_a, __int32 width_b, Type2Type<double>);
template __host__ cudaError_t multiply<int>(int *c, const int *a, const int *b, __int32 width_a, __int32 height_a, __int32 width_b, Type2Type<int>);

__host__ cudaError_t multiply(	ComplexFloat *c,
								const ComplexFloat *a,
								const ComplexFloat *b,
								__int32 width_a,
								__int32 height_a,
								__int32 width_b,
								Type2Type<ComplexFloat>)
{
	cudaError_t cudaStatus;

	struct Matrix<DeviceComplexFloat> A, B, C;
	A.width = width_a;
	A.height = height_a;
	B.width = width_b;
	B.height = A.width;
	C.height = A.height;
	C.width = B.width;
	
	struct OperationData<ComplexFloat, DeviceComplexFloat> data = {0};
	data.a = a;
	data.b = b;
	data.c = c;
	data.dev_a = &(A.elements);
	data.dev_b = &(B.elements);
	data.dev_c = &(C.elements);
	data.size = A.width*A.height;
	data.size_b = B.width*B.height;
	data.size_c = C.width*C.height;

	dim3 threads(16, 16);
	dim3 grid(	(B.width + threads.x - 1) / threads.x,
				(A.height + threads.y - 1) / threads.y);
	try
	{
		cudaStatus = allocMem(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		cudaStatus = copyHostDevice(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		multiply<DeviceComplexFloat><<<grid, threads>>>(A, B, C);
	
		cudaStatus = copyDeviceHost(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		deallocate(&data);
	}
	catch(...)
	{
		deallocate(&data);
    	throw;
	}

	return cudaStatus;
}

__host__ cudaError_t multiply(	ComplexDouble *c,
								const ComplexDouble *a,
								const ComplexDouble *b,
								__int32 width_a,
								__int32 height_a,
								__int32 width_b,
								Type2Type<ComplexDouble>)
{
	cudaError_t cudaStatus;

	struct Matrix<DeviceComplexDouble> A, B, C;
	A.width = width_a;
	A.height = height_a;
	B.width = width_b;
	B.height = A.width;
	C.height = A.height;
	C.width = B.width;
	
	struct OperationData<ComplexDouble, DeviceComplexDouble> data = {0};
	data.a = a;
	data.b = b;
	data.c = c;
	data.dev_a = &(A.elements);
	data.dev_b = &(B.elements);
	data.dev_c = &(C.elements);
	data.size = A.width*A.height;
	data.size_b = B.width*B.height;
	data.size_c = C.width*C.height;

	dim3 threads(16, 16);
	dim3 grid(	(B.width + threads.x - 1) / threads.x,
				(A.height + threads.y - 1) / threads.y);
	try
	{
		cudaStatus = allocMem(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		cudaStatus = copyHostDevice(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		multiply<DeviceComplexDouble><<<grid, threads>>>(A, B, C);
	
		cudaStatus = copyDeviceHost(&data);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		deallocate(&data);
	}
	catch(...)
	{
		deallocate(&data);
    	throw;
	}

	return cudaStatus;
}

template<typename T>
__host__ cudaError_t determinant(T *c, const T *dev_a, __int32 rows)
{
	// TODO Optmize this!

	T *dev_c = 0;
    cudaError_t cudaStatus;

	try
	{
		cudaStatus = cudaMalloc((void**)(&dev_c), sizeof(T));
		if(cudaStatus != cudaSuccess)
			throw std::exception();
	
		det<T><<<1,1>>>(dev_c, dev_a, rows);
	
		cudaStatus = cudaMemcpyAsync(c, dev_c, sizeof(T), cudaMemcpyDeviceToHost);
		if(cudaStatus != cudaSuccess)
			throw std::exception();

		cudaFree(dev_c);
	}
	catch(...)
	{
		cudaFree(dev_c);
    	throw;
	}

	return cudaStatus;
}

template __host__ cudaError_t determinant(float *c, const float *dev_a, __int32 rows);
template __host__ cudaError_t determinant(double *c, const double *dev_a, __int32 rows);

// put zeros under the main diagonal of matrix

template<> __global__ void zeros_under_diag_kernel(ComplexFloat *a, __int32 size)
{    
	// index
	cuFloatComplex *aux;
	aux = (cuFloatComplex *)a;
	int column = threadIdx.x + blockIdx.x*blockDim.x;
	int line = threadIdx.y + blockIdx.y*blockDim.y;

	while( line > 0 && line < size && column < line )
	{
		aux[line + column*size] = make_cuComplex( 0, 0 );
		column += blockDim.x * gridDim.x;
		line += blockDim.y * gridDim.y;
	}

	__syncthreads();
}

template<> __global__ void zeros_under_diag_kernel(ComplexDouble *a, __int32 size)
{    
	// index
	cuDoubleComplex *aux;
	aux = (cuDoubleComplex *)a;
	int column = threadIdx.x + blockIdx.x*blockDim.x;
	int line = threadIdx.y + blockIdx.y*blockDim.y;

	while( line > 0 && line < size && column < line )
	{
		aux[line + column*size] = make_cuDoubleComplex( 0, 0 );
		column += blockDim.x * gridDim.x;
		line += blockDim.y * gridDim.y;
	}

	__syncthreads();
}
template <typename T>
__global__ void zeros_under_diag_kernel(T *a, __int32 size)
{    
	// index
	int column = threadIdx.x + blockIdx.x*blockDim.x;
	int line = threadIdx.y + blockIdx.y*blockDim.y;

	while( line > 0 && line < size && column < line )
	{
		a[line + column*size] = 0;
		column += blockDim.x * gridDim.x;
		line += blockDim.y * gridDim.y;
	}

	__syncthreads();
}

template __global__ void zeros_under_diag_kernel( int *a, __int32 size );
template __global__ void zeros_under_diag_kernel( float *a, __int32 size );
template __global__ void zeros_under_diag_kernel( double *a, __int32 size );

template <typename T>
__host__ void zeros_under_diag(T *a, __int32 size)
{    
	dim3 threadsPerBlock2( 32, 32 );
	dim3 blocksPerGrid2( min( (size + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (size + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );

	//zeros_under_diag_kernel<T> <<< blocksPerGrid2, threadsPerBlock2 >>>( a, size );
	zeros_under_diag_kernel <<< blocksPerGrid2, threadsPerBlock2 >>>( a, size );
}

template __host__ void zeros_under_diag( int *a, __int32 size );
template __host__ void zeros_under_diag( float *a, __int32 size );
template __host__ void zeros_under_diag( double *a, __int32 size );
template __host__ void zeros_under_diag( ComplexFloat *a, __int32 size );
template __host__ void zeros_under_diag( ComplexDouble *a, __int32 size );

//__global__ void zeros_under_diag_kernel(int *a, __int32 size)
//{    
//	// index
//	int column = threadIdx.x + blockIdx.x*blockDim.x;
//	int line = threadIdx.y + blockIdx.y*blockDim.y;
//
//	while( line > 0 && line < size && column < line )
//	{
//		a[line + column*size] = 0;
//		column += blockDim.x * gridDim.x;
//		line += blockDim.y * gridDim.y;
//	}
//
//	__syncthreads();
//}
//
//__global__ void zeros_under_diag_kernel(float *a, __int32 size)
//{    
//	// index
//	int column = threadIdx.x + blockIdx.x*blockDim.x;
//	int line = threadIdx.y + blockIdx.y*blockDim.y;
//
//	while( line > 0 && line < size && column < line )
//	{
//		a[line + column*size] = 0;
//		column += blockDim.x * gridDim.x;
//		line += blockDim.y * gridDim.y;
//	}
//
//	__syncthreads();
//}
//
//__global__ void zeros_under_diag_kernel(double *a, __int32 size)
//{    
//	// index
//	int column = threadIdx.x + blockIdx.x*blockDim.x;
//	int line = threadIdx.y + blockIdx.y*blockDim.y;
//
//	while( line > 0 && line < size && column < line )
//	{
//		a[line + column*size] = 0;
//		column += blockDim.x * gridDim.x;
//		line += blockDim.y * gridDim.y;
//	}
//
//	__syncthreads();
//}
//
//__global__ void zeros_under_diag_kernel(ComplexFloat *a, __int32 size)
//{    
//	// index
//	cuFloatComplex *aux;
//	aux = (cuFloatComplex *)a;
//	int column = threadIdx.x + blockIdx.x*blockDim.x;
//	int line = threadIdx.y + blockIdx.y*blockDim.y;
//
//	while( line > 0 && line < size && column < line )
//	{
//		aux[line + column*size] = make_cuComplex( 0, 0 );
//		column += blockDim.x * gridDim.x;
//		line += blockDim.y * gridDim.y;
//	}
//
//	__syncthreads();
//}
//
//__global__ void zeros_under_diag_kernel(ComplexDouble *a, __int32 size)
//{    
//	// index
//	cuDoubleComplex *aux;
//	aux = (cuDoubleComplex *)a;
//	int column = threadIdx.x + blockIdx.x*blockDim.x;
//	int line = threadIdx.y + blockIdx.y*blockDim.y;
//
//	while( line > 0 && line < size && column < line )
//	{
//		aux[line + column*size] = make_cuDoubleComplex( 0, 0 );
//		column += blockDim.x * gridDim.x;
//		line += blockDim.y * gridDim.y;
//	}
//
//	__syncthreads();
//}

// put zeros above the main diagonal of matrix

template<> __global__ void zeros_above_diag_kernel(ComplexFloat *a, __int32 size)
{    
	// index
	cuFloatComplex *aux;
	aux = (cuFloatComplex *)a;
	int column = threadIdx.x + blockIdx.x*blockDim.x;
	int line = threadIdx.y + blockIdx.y*blockDim.y;

	while( line > 0 && line < size && column > line )
	{
		aux[line + column*size] = make_cuComplex( 0, 0 );
		column += blockDim.x * gridDim.x;
		line += blockDim.y * gridDim.y;
	}

	__syncthreads();
}

template<> __global__ void zeros_above_diag_kernel(ComplexDouble *a, __int32 size)
{    
	// index
	cuDoubleComplex *aux;
	aux = (cuDoubleComplex *)a;
	int column = threadIdx.x + blockIdx.x*blockDim.x;
	int line = threadIdx.y + blockIdx.y*blockDim.y;

	while( line > 0 && line < size && column > line )
	{
		aux[line + column*size] = make_cuDoubleComplex( 0, 0 );
		column += blockDim.x * gridDim.x;
		line += blockDim.y * gridDim.y;
	}

	__syncthreads();
}

template <typename T>
__global__ void zeros_above_diag_kernel(T *a, __int32 size)
{    
	// index
	int column = threadIdx.x + blockIdx.x*blockDim.x;
	int line = threadIdx.y + blockIdx.y*blockDim.y;

	while( line > 0 && line < size && column > line )
	{
		a[line + column*size] = 0;
		column += blockDim.x * gridDim.x;
		line += blockDim.y * gridDim.y;
	}

	__syncthreads();
}

template __global__ void zeros_above_diag_kernel(int *a, __int32 size);
template __global__ void zeros_above_diag_kernel(float *a, __int32 size);
template __global__ void zeros_above_diag_kernel(double *a, __int32 size);

template <typename T>
__host__ void zeros_above_diag(T *a, __int32 size)
{    
	dim3 threadsPerBlock2( 32, 32 );
	dim3 blocksPerGrid2( min( (size + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (size + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );

	//zeros_above_diag_kernel<T> <<< blocksPerGrid2, threadsPerBlock2 >>>( a, size );
	zeros_above_diag_kernel <<< blocksPerGrid2, threadsPerBlock2 >>>( a, size );
}

template __host__ void zeros_above_diag( int *a, __int32 size );
template __host__ void zeros_above_diag( float *a, __int32 size );
template __host__ void zeros_above_diag( double *a, __int32 size );
template __host__ void zeros_above_diag( ComplexFloat *a, __int32 size );
template __host__ void zeros_above_diag( ComplexDouble *a, __int32 size );

// identity matrix 

template<> __global__ void cuda_eye_kernel(cuComplex *a, __int32 size)
{
	// index
	int column = threadIdx.x + blockIdx.x*blockDim.x;
	int line = threadIdx.y + blockIdx.y*blockDim.y;

	while( line >= 0 && line < size && column >= 0 && column < size ) {
		a[line + column*size] = make_cuComplex( line == column, 0 );
		column += blockDim.x * gridDim.x;
		line += blockDim.y * gridDim.y;
	}

	__syncthreads();
}

template<> __global__ void cuda_eye_kernel(cuDoubleComplex *a, __int32 size)
{
	// index
	int column = threadIdx.x + blockIdx.x*blockDim.x;
	int line = threadIdx.y + blockIdx.y*blockDim.y;

	while( line >= 0 && line < size && column >= 0 && column < size ) {
		a[line + column*size] = make_cuDoubleComplex( line == column, 0 );
		column += blockDim.x * gridDim.x;
		line += blockDim.y * gridDim.y;
	}

	__syncthreads();
}

template <typename T>
__global__ void cuda_eye_kernel(T *a, __int32 size) 
{
	// index
	int column = threadIdx.x + blockIdx.x*blockDim.x;
	int line = threadIdx.y + blockIdx.y*blockDim.y;

	while( line >= 0 && line < size && column >= 0 && column < size ) {
		a[line + column*size] = line == column;
		column += blockDim.x * gridDim.x;
		line += blockDim.y * gridDim.y;
	}

	__syncthreads();
}

template __global__ void cuda_eye_kernel(int *a, __int32 size);
template __global__ void cuda_eye_kernel(float *a, __int32 size); 
template __global__ void cuda_eye_kernel(double *a, __int32 size);

template<> __host__ void cuda_eye( ComplexFloat *a, __int32 size )
{    
	dim3 threadsPerBlock2( 32, 32 );
	dim3 blocksPerGrid2( min( (size + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (size + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );

	//cuda_eye_kernel<T> <<< blocksPerGrid2, threadsPerBlock2 >>>( a, size );
	cuda_eye_kernel <<< blocksPerGrid2, threadsPerBlock2 >>>( (cuFloatComplex *)a, size );
	CUDA_CALL( cudaDeviceSynchronize() );
}

template<> __host__ void cuda_eye( ComplexDouble *a, __int32 size )
{    
	dim3 threadsPerBlock2( 32, 32 );
	dim3 blocksPerGrid2( min( (size + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (size + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );

	//cuda_eye_kernel<T> <<< blocksPerGrid2, threadsPerBlock2 >>>( a, size );
	cuda_eye_kernel <<< blocksPerGrid2, threadsPerBlock2 >>>( (cuDoubleComplex *)a, size );
	CUDA_CALL( cudaDeviceSynchronize() );
}

template <typename T>
__host__ void cuda_eye(T *a, __int32 size)
{    
	dim3 threadsPerBlock2( 32, 32 );
	dim3 blocksPerGrid2( min( (size + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (size + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );

	//cuda_eye_kernel<T> <<< blocksPerGrid2, threadsPerBlock2 >>>( a, size );
	cuda_eye_kernel <<< blocksPerGrid2, threadsPerBlock2 >>>( a, size );
	CUDA_CALL( cudaDeviceSynchronize() );
}

template __host__ void cuda_eye( int *a, __int32 size );
template __host__ void cuda_eye( float *a, __int32 size );
template __host__ void cuda_eye( double *a, __int32 size );

// min of Array

template <typename T>
__host__ T cuda_min( T *a, int *idx, __int32 size )
{
	thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(a);

	thrust::device_ptr<T> min_ptr = thrust::min_element(dev_ptr, dev_ptr + size);
	
	*idx = (&min_ptr[0] - &dev_ptr[0]);
	return min_ptr[0];
}

template __host__ float cuda_min( float *a, index_t *idx, __int32 size );
template __host__ double cuda_min( double *a, index_t *idx, __int32 size );

// max of Array

template <typename T>
__host__ T cuda_max( T *a, int *idx, __int32 size )
{
	thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(a);

	thrust::device_ptr<T> max_ptr = thrust::max_element(dev_ptr, dev_ptr + size);
	
	*idx = (&max_ptr[0] - &dev_ptr[0]);
	return max_ptr[0];
}

template __host__ float cuda_max( float *a, index_t *idx, __int32 size );
template __host__ double cuda_max( double *a, index_t *idx, __int32 size );

// elementwise multiplication

template<> __global__ void cuda_elementwise_multiplication_kernel( cuFloatComplex *a, cuFloatComplex *b, cuFloatComplex *c, size_t N )
{
	// index
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	float r, i;
	
	while( idx >= 0 && idx < N ) {
		r = a[idx].x*b[idx].x - a[idx].y*b[idx].y;
		i = a[idx].x*b[idx].y + a[idx].y*b[idx].x;
		c[idx] = make_cuFloatComplex(r,i);

		idx += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

template<> __global__ void cuda_elementwise_multiplication_kernel( cuDoubleComplex *a, cuDoubleComplex *b, cuDoubleComplex *c, size_t N )
{
	// index
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	double r, i;
	
	while( idx >= 0 && idx < N ) {
		r = a[idx].x*b[idx].x - a[idx].y*b[idx].y;
		i = a[idx].x*b[idx].y + a[idx].y*b[idx].x;
		c[idx] = make_cuDoubleComplex(r,i);

		idx += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

template <typename T>
__global__ void cuda_elementwise_multiplication_kernel( T *a, T *b, T *c, size_t N )
{
	// index
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	while( idx >= 0 && idx < N ) {
		c[idx] = a[idx]*b[idx];

		idx += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

template __global__ void cuda_elementwise_multiplication_kernel( float *a, float *b, float *c, size_t N );
template __global__ void cuda_elementwise_multiplication_kernel( double *a, double *b, double *c, size_t N );

template<> __host__ void cuda_elementwise_multiplication( ComplexFloat *a, ComplexFloat *b, ComplexFloat *c, size_t N )
{
	dim3 threadsPerBlock2( 32, 32 );
	dim3 blocksPerGrid2( min( (N + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (N + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );

	cuda_elementwise_multiplication_kernel<<< blocksPerGrid2, threadsPerBlock2 >>>( (cuFloatComplex *)a, (cuFloatComplex *)b, (cuFloatComplex *)c, N );
	CUDA_CALL( cudaDeviceSynchronize() );
}

template<> __host__ void cuda_elementwise_multiplication( ComplexDouble *a, ComplexDouble *b, ComplexDouble *c, size_t N )
{
	dim3 threadsPerBlock2( 32, 32 );
	dim3 blocksPerGrid2( min( (N + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (N + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );

	cuda_elementwise_multiplication_kernel<<< blocksPerGrid2, threadsPerBlock2 >>>( (cuDoubleComplex *)a, (cuDoubleComplex *)b, (cuDoubleComplex *)c, N );
	CUDA_CALL( cudaDeviceSynchronize() );
}

template <typename T>
__host__ void cuda_elementwise_multiplication( T *a, T *b, T *c, size_t N )
{
	dim3 threadsPerBlock2( 32, 32 );
	dim3 blocksPerGrid2( min( (N + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (N + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );

	cuda_elementwise_multiplication_kernel<<< blocksPerGrid2, threadsPerBlock2 >>>( a, b, c, N );
	CUDA_CALL( cudaDeviceSynchronize() );
}

template __host__ void cuda_elementwise_multiplication( float *a, float *b, float *c, size_t N );
template __host__ void cuda_elementwise_multiplication( double *a, double *b, double *c, size_t N );

// elementwise division

template<> __global__ void cuda_elementwise_division_kernel( cuFloatComplex *a, cuFloatComplex *b, cuFloatComplex *c, size_t N )
{
	// index
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	float r, i;
	
	while( idx >= 0 && idx < N ) {
		r = (b[idx].x*a[idx].x + b[idx].y*a[idx].y)/(pow(a[idx].x,2) + pow(a[idx].y,2));
		i = (b[idx].y*a[idx].x - b[idx].x*a[idx].y)/(pow(a[idx].x,2) + pow(a[idx].y,2));
		c[idx] = make_cuFloatComplex(r,i);

		idx += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

template<> __global__ void cuda_elementwise_division_kernel( cuDoubleComplex *a, cuDoubleComplex *b, cuDoubleComplex *c, size_t N )
{
	// index
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	double r, i;
	
	while( idx >= 0 && idx < N ) {
		r = (b[idx].x*a[idx].x + b[idx].y*a[idx].y)/(pow(a[idx].x,2) + pow(a[idx].y,2));
		i = (b[idx].y*a[idx].x - b[idx].x*a[idx].y)/(pow(a[idx].x,2) + pow(a[idx].y,2));
		c[idx] = make_cuDoubleComplex(r,i);

		idx += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

template <typename T>
__global__ void cuda_elementwise_division_kernel( T *a, T *b, T *c, size_t N )
{
	// index
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	while( idx >= 0 && idx < N ) {
		c[idx] = b[idx]/a[idx];

		idx += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

template __global__ void cuda_elementwise_division_kernel( float *a, float *b, float *c, size_t N );
template __global__ void cuda_elementwise_division_kernel( double *a, double *b, double *c, size_t N );

template<> __host__ void cuda_elementwise_division( ComplexFloat *a, ComplexFloat *b, ComplexFloat *c, size_t N )
{
	dim3 threadsPerBlock2( 32, 32 );
	dim3 blocksPerGrid2( min( (N + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (N + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );

	cuda_elementwise_division_kernel<<< blocksPerGrid2, threadsPerBlock2 >>>( (cuFloatComplex *)a, (cuFloatComplex *)b, (cuFloatComplex *)c, N );
	CUDA_CALL( cudaDeviceSynchronize() );
}

template<> __host__ void cuda_elementwise_division( ComplexDouble *a, ComplexDouble *b, ComplexDouble *c, size_t N )
{
	dim3 threadsPerBlock2( 32, 32 );
	dim3 blocksPerGrid2( min( (N + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (N + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );

	cuda_elementwise_division_kernel<<< blocksPerGrid2, threadsPerBlock2 >>>( (cuDoubleComplex *)a, (cuDoubleComplex *)b, (cuDoubleComplex *)c, N );
	CUDA_CALL( cudaDeviceSynchronize() );
}

template <typename T>
__host__ void cuda_elementwise_division( T *a, T *b, T *c, size_t N )
{
	dim3 threadsPerBlock2( 32, 32 );
	dim3 blocksPerGrid2( min( (N + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (N + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );

	cuda_elementwise_division_kernel<<< blocksPerGrid2, threadsPerBlock2 >>>( a, b, c, N );
	CUDA_CALL( cudaDeviceSynchronize() );
}

template __host__ void cuda_elementwise_division( float *a, float *b, float *c, size_t N );
template __host__ void cuda_elementwise_division( double *a, double *b, double *c, size_t N );

//__global__ void cuda_elementwise_division_kernel( cuFloatComplex *a, cuFloatComplex *b, cuFloatComplex *c, size_t N )
//{
//	// index
//	int idx = threadIdx.x + blockIdx.x*blockDim.x;
//	float r, i;
//	
//	while( idx >= 0 && idx < N ) {
//		r = (b[idx].x*a[idx].x + b[idx].y*a[idx].y)/(pow(a[idx].x,2) + pow(a[idx].y,2));
//		i = (b[idx].y*a[idx].x - b[idx].x*a[idx].y)/(pow(a[idx].x,2) + pow(a[idx].y,2));
//		c[idx] = make_cuFloatComplex(r,i);
//
//		idx += blockDim.x * gridDim.x;
//	}
//
//	__syncthreads();
//}
//
//__global__ void cuda_elementwise_division_kernel( cuDoubleComplex *a, cuDoubleComplex *b, cuDoubleComplex *c, size_t N )
//{
//	// index
//	int idx = threadIdx.x + blockIdx.x*blockDim.x;
//	double r, i;
//	
//	while( idx >= 0 && idx < N ) {
//		r = (b[idx].x*a[idx].x + b[idx].y*a[idx].y)/(pow(a[idx].x,2) + pow(a[idx].y,2));
//		i = (b[idx].y*a[idx].x - b[idx].x*a[idx].y)/(pow(a[idx].x,2) + pow(a[idx].y,2));
//		c[idx] = make_cuDoubleComplex(r,i);
//
//		idx += blockDim.x * gridDim.x;
//	}
//
//	__syncthreads();
//}
//
//__global__ void cuda_elementwise_division_kernel( float *a, float *b, float *c, size_t N )
//{
//	// index
//	int idx = threadIdx.x + blockIdx.x*blockDim.x;
//
//	while( idx >= 0 && idx < N ) {
//		c[idx] = b[idx]/a[idx];
//
//		idx += blockDim.x * gridDim.x;
//	}
//
//	__syncthreads();
//}
//
//__global__ void cuda_elementwise_division_kernel( double *a, double *b, double *c, size_t N )
//{
//	// index
//	int idx = threadIdx.x + blockIdx.x*blockDim.x;
//
//	while( idx >= 0 && idx < N ) {
//		c[idx] = (double)(b[idx]/a[idx]);
//
//		idx += blockDim.x * gridDim.x;
//	}
//
//	__syncthreads();
//}
//
//__host__ void cuda_elementwise_division( ComplexFloat *a, ComplexFloat *b, ComplexFloat *c, size_t N )
//{
//	dim3 threadsPerBlock2( 32, 32 );
//	dim3 blocksPerGrid2( min( (N + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (N + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );
//
//	cuda_elementwise_division_kernel<<< blocksPerGrid2, threadsPerBlock2 >>>( (cuFloatComplex *)a, (cuFloatComplex *)b, (cuFloatComplex *)c, N );
//	CUDA_CALL( cudaDeviceSynchronize() );
//}
//
//__host__ void cuda_elementwise_division( ComplexDouble *a, ComplexDouble *b, ComplexDouble *c, size_t N )
//{
//	dim3 threadsPerBlock2( 32, 32 );
//	dim3 blocksPerGrid2( min( (N + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (N + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );
//
//	cuda_elementwise_division_kernel<<< blocksPerGrid2, threadsPerBlock2 >>>( (cuDoubleComplex *)a, (cuDoubleComplex *)b, (cuDoubleComplex *)c, N );
//	CUDA_CALL( cudaDeviceSynchronize() );
//}
//
//__host__ void cuda_elementwise_division( float *a, float *b, float *c, size_t N )
//{
//	dim3 threadsPerBlock2( 32, 32 );
//	dim3 blocksPerGrid2( min( (N + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (N + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );
//
//	cuda_elementwise_division_kernel<<< blocksPerGrid2, threadsPerBlock2 >>>( a, b, c, N );
//	CUDA_CALL( cudaDeviceSynchronize() );
//}
//
//__host__ void cuda_elementwise_division( double *a, double *b, double *c, size_t N )
//{
//	dim3 threadsPerBlock2( 32, 32 );
//	dim3 blocksPerGrid2( min( (N + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (N + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );
//
//	cuda_elementwise_division_kernel<<< blocksPerGrid2, threadsPerBlock2 >>>( a, b, c, N );
//	CUDA_CALL( cudaDeviceSynchronize() );
//}

// abs - calculates elementWise magnitude values of Array

template<> __global__ void cuda_abs_kernel( cuFloatComplex *a, cuFloatComplex *result, size_t N ) 
{
	// index
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	while( idx >= 0 && idx < N ) {
		result[idx] = make_cuFloatComplex( sqrt(a[idx].x*a[idx].x + a[idx].y*a[idx].y), 0 );

		idx += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

template<> __global__ void cuda_abs_kernel( cuDoubleComplex *a, cuDoubleComplex *result, size_t N )
{
	// index
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	while( idx >= 0 && idx < N ) {
		result[idx] = make_cuDoubleComplex( sqrt(a[idx].x*a[idx].x + a[idx].y*a[idx].y), 0 );

		idx += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

template<> __host__ void cuda_abs( ComplexFloat *a, ComplexFloat *result, size_t N )
{
	dim3 threadsPerBlock2( 32, 32 );
	dim3 blocksPerGrid2( min( (N + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (N + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );

	cuda_abs_kernel<cuFloatComplex><<< blocksPerGrid2, threadsPerBlock2 >>>( (cuFloatComplex*)a, (cuFloatComplex*)result, N );
	CUDA_CALL( cudaDeviceSynchronize() );
}

template<> __host__ void cuda_abs( ComplexDouble *a, ComplexDouble *result, size_t N )
{
	dim3 threadsPerBlock2( 32, 32 );
	dim3 blocksPerGrid2( min( (N + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (N + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );

	cuda_abs_kernel<cuDoubleComplex><<< blocksPerGrid2, threadsPerBlock2 >>>( (cuDoubleComplex*)a, (cuDoubleComplex*)result, N );
	CUDA_CALL( cudaDeviceSynchronize() );
}

// abs2 - calculates elementWise magnitude squared values of Array

template<> __global__ void cuda_abs2_kernel( cuFloatComplex *a, cuFloatComplex *result, size_t N ) 
{
	// index
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	while( idx >= 0 && idx < N ) {
		result[idx] = make_cuFloatComplex( (a[idx].x*a[idx].x + a[idx].y*a[idx].y), 0 );

		idx += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

template<> __global__ void cuda_abs2_kernel( cuDoubleComplex *a, cuDoubleComplex *result, size_t N )
{
	// index
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	while( idx >= 0 && idx < N ) {
		result[idx] = make_cuDoubleComplex( (a[idx].x*a[idx].x + a[idx].y*a[idx].y), 0 );

		idx += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

template<> __host__ void cuda_abs2( ComplexFloat *a, ComplexFloat *result, size_t N )
{
	dim3 threadsPerBlock2( 32, 32 );
	dim3 blocksPerGrid2( min( (N + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (N + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );

	cuda_abs2_kernel<cuFloatComplex><<< blocksPerGrid2, threadsPerBlock2 >>>( (cuFloatComplex*)a, (cuFloatComplex*)result, N );
	CUDA_CALL( cudaDeviceSynchronize() );
}

template<> __host__ void cuda_abs2( ComplexDouble *a, ComplexDouble *result, size_t N )
{
	dim3 threadsPerBlock2( 32, 32 );
	dim3 blocksPerGrid2( min( (N + threadsPerBlock2.x - 1)/threadsPerBlock2.x , 32 ), min( (N + threadsPerBlock2.y - 1)/threadsPerBlock2.y , 32 ) );

	cuda_abs2_kernel<cuDoubleComplex><<< blocksPerGrid2, threadsPerBlock2 >>>( (cuDoubleComplex*)a, (cuDoubleComplex*)result, N );
	CUDA_CALL( cudaDeviceSynchronize() );
}