#ifndef COMMON_H
#define COMMON_H

#define imin(a,b)	(a<b?a:b)
#define imax(a,b)	(a<b?b:a)

#define TILE_DIM	4

typedef __int32 index_t;

#include <complex>
typedef std::complex<float> ComplexFloat;
typedef std::complex<double> ComplexDouble;
//typedef ComplexFloat Complex;
//typedef ComplexDouble Complex;

#define CONFIDENCE_INTERVAL					10;
#define CONFIDENCE_INTERVAL_FLOAT			1E-6;
#define CONFIDENCE_INTERVAL_DOUBLE			1E-16;


/* Some handy stuff */
#define CUDA_CALL(value) do {		           								      \
	cudaError_t _m_cudaStat = value;										          \
	if (_m_cudaStat != cudaSuccess) {									          	\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(-1);        													                  \
	} } while(0)
 
#define CUBLAS_CALL(value) do {                                                                 \
	cublasStatus_t _m_status = value;                                                             \
	if (_m_status != CUBLAS_STATUS_SUCCESS){                                                      \
		fprintf(stderr, "Error %d at line %d in file %s\n", (int)_m_status, __LINE__, __FILE__);    \
	exit(-2);                                                                                     \
	}                                                                                             \
	} while(0)

#define CUFFT_CALL(value) do {                                                                 \
	cufftResult_t _m_status = value;                                                             \
	if (_m_status != CUFFT_SUCCESS){                                                      \
		fprintf(stderr, "Error %d at line %d in file %s\n", (int)_m_status, __LINE__, __FILE__);    \
	exit(-3);                                                                                     \
	}                                                                                             \
	} while(0)
 
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) {  \
    printf("Error at %s:%d\n",__FILE__,__LINE__);             \
    exit(-4);}} while(0)
 
#define CUSPARSE_CALL(value) do {                                                                 \
	cusparseStatus_t _m_status = value;                                                             \
	if (_m_status != CUSPARSE_STATUS_SUCCESS){                                                      \
		fprintf(stderr, "Error %d at line %d in file %s\n", (int)_m_status, __LINE__, __FILE__);    \
	exit(-5);                                                                                     \
	}                                                                                             \
	} while(0)
 
#define CUSOLVER_CALL(value) do {                                                                 \
	cusolverStatus_t _m_status = value;                                                             \
	if (_m_status != CUSOLVER_STATUS_SUCCESS){                                                      \
		fprintf(stderr, "Error %d at line %d in file %s\n", (int)_m_status, __LINE__, __FILE__);    \
	exit(-5);                                                                                     \
	}                                                                                             \
	} while(0)


template <typename T>
struct Type2Type
{
	typedef T OriginalType;
};

class CudaDevice
{
    public:
        static CudaDevice& getInstance()
        {
            static CudaDevice instance;
		    return instance;
        }
    private:
        CudaDevice();
        
		CudaDevice(CudaDevice const&);
        void operator=(CudaDevice const&);
};

#endif