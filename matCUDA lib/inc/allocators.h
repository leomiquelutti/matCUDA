#ifndef ALLOCATORS_H
#define ALLOCATORS_H

#include <cuda_runtime.h>

template< class T >
struct DefaultAllocator
{
	typedef T Elem;
	static void deallocate(T * p) { delete[] p; }
	static T * allocate(size_t s) { return new T[s]; }

	static void copy( T* begin, T* end, T* dst )
	{
		std::copy ( begin, end, dst );
	}

	static T* clone (T* begin, T* end)
	{
		T* p = allocate( end - begin );
		copy( begin, end, p );
		return p;
	}
};

template< class T >
struct DeviceAllocator
{
	typedef T Elem;
	static void deallocate(T * p) { cudaFree(p); }
	static T * allocate(size_t s)
	{
		void* p;
		cudaMalloc(&p, s * sizeof(T) );
		return (T*)p;
	}

	static void copy( T* begin, T* end, T* dst )
	{
		cudaMemcpy( dst, begin, (end-begin)*sizeof(T), cudaMemcpyDeviceToDevice );
	}

	static T* clone (T* begin, T* end) {
		T* p = allocate( end - begin );
		copy( begin, end, p );
		return p;
	}
};

template< class T >
struct HostAllocator
{
	typedef T Elem;
	static void deallocate(T * p) { cudaFreeHost(p); }
	static void deallocate(T * p, int size) { cudaFreeHost(p); }
	static T * allocate(size_t s)
	{
		T* p;
		cudaMallocHost(&p, s * sizeof(T) );
		return p;
	}

	static void copy( T* begin, T* end, T* dst )
	{
		cudaMemcpy( dst, begin, (end-begin)*sizeof(T), cudaMemcpyHostToHost );
	}

	static T* clone (T* begin, T* end)
	{
		T* p = allocate( end - begin );
		copy( begin, end, p );
		return p;
	}
};

template< class T >
struct MappedHostAllocator : public HostAllocator<T>
{
	static T * allocate(size_t s)
	{
		T* p;
		cudaHostAlloc(&p, s * sizeof(T), cudaHostAllocMapped );
		return p;
	}

	static T* getDevicePointer(T* hp)
	{
		T* dp;
		cudaGetDevicePointer(&dp,hp,0);
		return dp;
	}
};

template< class A, class T>
void alloc_copy(A,A, T* begin, T* end, T* dst)
{
	A::copy(begin,end,dst);
}

template< class T>
void alloc_copy(DefaultAllocator<T>,DeviceAllocator<T>, T* begin, T* end, T* dst)
{
	cudaMemcpy(dst, begin, (end-begin)*sizeof(T), cudaMemcpyHostToDevice);
}

template< class T>
void alloc_copy(DeviceAllocator<T>,DefaultAllocator<T>, T* begin, T* end, T* dst)
{
	cudaMemcpy(dst, begin, (end-begin)*sizeof(T), cudaMemcpyDeviceToHost);
}

#endif