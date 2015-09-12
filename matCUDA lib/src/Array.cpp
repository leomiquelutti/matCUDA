#include <stdarg.h>
#include <fstream>
#include <boost/exception/all.hpp>
#include <boost/math/special_functions.hpp>

#include "common.h"

#include "Array.h"

// ArrayDescriptor implementation

#define VAR_NAME_PRINTER(name) var_name_printer(#name)

std::string var_name_printer(char *name) { return std::string(name); }

namespace matCUDA
{
	ArrayDescriptor::ArrayDescriptor(int count)
		: m_count(count)
	{
			m_dim = new index_t[count];
	}

	ArrayDescriptor::ArrayDescriptor(ArrayDescriptor &source)
		: m_count(source.m_count)
	{
		m_dim = new index_t[m_count];
		for(index_t i = 0; i < m_count; i++)
			m_dim[i] = source.m_dim[i];
	}

	ArrayDescriptor::ArrayDescriptor(std::vector<index_t> &source)
		: m_count((int)source.size())
	{
		m_dim = new index_t[m_count];
		index_t i = 0;
		for(std::vector<int>::iterator it = source.begin(); it != source.end(); ++it)
			m_dim[i++] = *it;
	}

	ArrayDescriptor::ArrayDescriptor(int source[])
		: m_count(sizeof(source)/sizeof(int))
	{
		m_dim = new index_t[m_count];
		for(index_t i = 0; i < m_count; i++)
			m_dim[i] = source[i];
	}

	ArrayDescriptor::~ArrayDescriptor() { delete[] m_dim; }

	int ArrayDescriptor::GetDim(int dim) { return m_dim[dim]; }

	int ArrayDescriptor::GetNDim() { return m_count; }

	int ArrayDescriptor::GetNumberOfElements() 
	{
		int NDim = ArrayDescriptor::GetNDim();
		int TotalDim = 1;
		for (int i = 0; i < NDim; i++ )
			TotalDim *= ArrayDescriptor::GetDim( i );
		return TotalDim;
	}

	void ArrayDescriptor::Swap()
	{
		if(m_count != 2)
			return;

		index_t temp = m_dim[0];
		m_dim[0] = m_dim[1];
		m_dim[1] = temp;
	}

	bool ArrayDescriptor::operator == (const ArrayDescriptor &other) const
	{
		if(m_count != other.m_count)
			return false;

		for(index_t i = 0; i < m_count; i++)
			if(m_dim[i] != other.m_dim[i])
				return false;

		return true;
	}

	bool ArrayDescriptor::operator != (const ArrayDescriptor &other) const
	{
		return !(*this == other);
	}

	// LinearIndexerND implementation

	LinearIndexer::LinearIndexer(ArrayDescriptor &descriptor)
		: m_descriptor(descriptor)
	{
		m_pos = new index_t[m_descriptor.m_count];
	}

	LinearIndexer::LinearIndexer(std::vector<index_t> &descriptor)
		: m_descriptor(descriptor)
	{
		m_pos = new index_t[m_descriptor.m_count];
	}

	LinearIndexer::~LinearIndexer()
	{
		delete[] m_pos;
	}

	ArrayDescriptor& LinearIndexer::GetDescriptor()
	{
		return m_descriptor;
	}

	index_t LinearIndexer::GetIndex() const
	{
		index_t size = 1;
		index_t idx = 0;
		for(int i = 0; i < m_descriptor.m_count; i++)
		{
			idx = idx + (m_pos[i] * size);
			size = size * m_descriptor.m_dim[i];
		}

		return idx;
	}

	index_t LinearIndexer::GetSize() const
	{
		index_t size = 1;
		for(int i = 0; i < m_descriptor.m_count; i++)
		{
			size = size * m_descriptor.m_dim[i];
		}

		return size;
	}

	index_t LinearIndexer::GetNeededBufferCapacity(ArrayDescriptor &descriptor)
	{
		index_t size = 1;
		for(int i = 0; i < descriptor.m_count; i++) 
			size *= descriptor.m_dim[i];
		
		return size;
	}

	index_t LinearIndexer::GetNeededBufferCapacity(std::vector<index_t> &descriptor)
	{
		index_t size = 1;
		for(std::vector<int>::iterator it = descriptor.begin(); it != descriptor.end(); ++it)
			size *= *it;
	
		return size;
	}

	// ArrayData implementation

	template <typename TElement, class TAllocator>
	ArrayData<TElement, TAllocator>::ArrayData(	ArrayDescriptor &descriptor,
												const TElement &defaulTElement,
												const TAllocator &allocator)
												: m_data(0),
												m_numElements(LinearIndexer::GetNeededBufferCapacity(descriptor)),
												m_allocator(allocator),
												m_size(-1)
	{
		// TODO Find a better solution
		CudaDevice::getInstance();

		// MxN matrices will be rounded up to square
		// So transpose operation can use CUDA optimizations
		/*index_t elements = 0;
		if(descriptor.GetNDim() != 2)
		{
			elements = m_numElements;
		} else
		{
			m_size = imax(((descriptor.GetDim(0)+(TILE_DIM-1))/TILE_DIM)*TILE_DIM,((descriptor.GetDim(1)+(TILE_DIM-1))/TILE_DIM)*TILE_DIM);
			elements = m_size*m_size;
		}

		m_data = m_allocator.allocate(elements);
		for(index_t i = 0; i < elements; i++)
			new (&m_data[i]) TElement(defaulTElement);*/

		m_data = m_allocator.allocate(m_numElements);
		for(index_t i = 0; i < m_numElements; i++)
			new (&m_data[i]) TElement(defaulTElement);
	}

	template ArrayData<int, std::allocator<int>>::ArrayData(ArrayDescriptor &descriptor,
															const int &defaulTElement,
															const std::allocator<int> &allocator);
	template ArrayData<float, std::allocator<float>>::ArrayData(ArrayDescriptor &descriptor,
																const float &defaulTElement,
																const std::allocator<float> &allocator);
	template ArrayData<double, std::allocator<double>>::ArrayData(	ArrayDescriptor &descriptor,
																	const double &defaulTElement,
																	const std::allocator<double> &allocator);
	template ArrayData<ComplexFloat, std::allocator<ComplexFloat>>::ArrayData(	ArrayDescriptor &descriptor,
																				const ComplexFloat &defaulTElement,
																				const std::allocator<ComplexFloat> &allocator);
	template ArrayData<ComplexDouble, std::allocator<ComplexDouble>>::ArrayData(ArrayDescriptor &descriptor,
																				const ComplexDouble &defaulTElement,
																				const std::allocator<ComplexDouble> &allocator);

	template <typename TElement, class TAllocator>
	ArrayData<TElement, TAllocator>::ArrayData(	index_t numElements,
												const TAllocator &allocator)
												: m_data(0),
												m_numElements(numElements),
												m_allocator(allocator),
												m_size(-1)
	{
		// TODO Find a better solution
		CudaDevice::getInstance();

		m_data = m_allocator.allocate(m_numElements);
	}

	template ArrayData<int, std::allocator<int>>::ArrayData(index_t numElements,
															const std::allocator<int> &allocator);
	template ArrayData<float, std::allocator<float>>::ArrayData(index_t numElements,
																const std::allocator<float> &allocator);
	template ArrayData<double, std::allocator<double>>::ArrayData(	index_t numElements,
																	const std::allocator<double> &allocator);
	template ArrayData<ComplexFloat, std::allocator<ComplexFloat>>::ArrayData(index_t numElements,
																	const std::allocator<ComplexFloat> &allocator);
	template ArrayData<ComplexDouble, std::allocator<ComplexDouble>>::ArrayData(index_t numElements,
																	const std::allocator<ComplexDouble> &allocator);

	template ArrayData<int, struct MappedHostAllocator<int>>::ArrayData(index_t numElements,
																		const struct MappedHostAllocator<int> &allocator);
	template ArrayData<float, struct MappedHostAllocator<float>>::ArrayData(index_t numElements,
																		const struct MappedHostAllocator<float> &allocator);
	template ArrayData<double, struct MappedHostAllocator<double>>::ArrayData(index_t numElements,
																		const struct MappedHostAllocator<double> &allocator);
	template ArrayData<ComplexFloat, struct MappedHostAllocator<ComplexFloat>>::ArrayData(index_t numElements,
																		const struct MappedHostAllocator<ComplexFloat> &allocator);
	template ArrayData<ComplexDouble, struct MappedHostAllocator<ComplexDouble>>::ArrayData(index_t numElements,
																		const struct MappedHostAllocator<ComplexDouble> &allocator);

	template <typename TElement, class TAllocator>
	ArrayData<TElement, TAllocator>::ArrayData(ArrayData &source)
			: m_data(0),
			m_numElements(source.m_numElements),
			m_allocator(source.m_allocator),
			m_size(source.m_size)
	{
		// TODO Find a better solution
		CudaDevice::getInstance();

		m_data = m_allocator.allocate(m_numElements);
		for(index_t i = 0; i < m_numElements; i++)
			new (&m_data[i]) TElement(source.m_data[i]);

		// MxN matrices will be rounded up to square
		// So transpose operation can use CUDA optimizations
		/*index_t elements = (m_size == -1) ? source.m_numElements : (source.m_size*source.m_size);
		m_data = m_allocator.allocate(elements);
		for(index_t i = 0; i < elements; i++)
			new (&m_data[i]) TElement(source.m_data[i]);*/
	}

	template ArrayData<int>::ArrayData(ArrayData &source);
	template ArrayData<float>::ArrayData(ArrayData &source);
	template ArrayData<double>::ArrayData(ArrayData &source);
	template ArrayData<ComplexFloat>::ArrayData(ArrayData &source);
	template ArrayData<ComplexDouble>::ArrayData(ArrayData &source);
 
	template <typename TElement, class TAllocator>
	ArrayData<TElement, TAllocator>::~ArrayData()
	{
		index_t elments = (m_size == -1) ? m_numElements : (m_size*m_size);
		for(index_t i = 0; i < elments; i++)
			(&m_data[i])->TElement::~TElement();
		m_allocator.deallocate(m_data, elments);
	}

	template ArrayData<int>::~ArrayData();
	template ArrayData<float>::~ArrayData();
	template ArrayData<double>::~ArrayData();
	template ArrayData<ComplexFloat>::~ArrayData();
	template ArrayData<ComplexDouble>::~ArrayData();

	template <typename TElement, class TAllocator>
	const TElement& ArrayData<TElement, TAllocator>::operator [] (index_t index) const
	{
		return m_data[index];
	}

	template const int& ArrayData<int>::operator [] (index_t index) const;
	template const float& ArrayData<float>::operator [] (index_t index) const;
	template const double& ArrayData<double>::operator [] (index_t index) const;
	template const ComplexFloat& ArrayData<ComplexFloat>::operator [] (index_t index) const;
	template const ComplexDouble& ArrayData<ComplexDouble>::operator [] (index_t index) const;

	template <typename TElement, class TAllocator>
	TElement& ArrayData<TElement, TAllocator>::operator [] (index_t index)
	{
		return m_data[index];
	}

	template int& ArrayData<int>::operator [] (index_t index);
	template float& ArrayData<float>::operator [] (index_t index);
	template double& ArrayData<double>::operator [] (index_t index);
	template ComplexFloat& ArrayData<ComplexFloat>::operator [] (index_t index);
	template ComplexDouble& ArrayData<ComplexDouble>::operator [] (index_t index);

	template <typename TElement, class TAllocator>
	bool ArrayData<TElement, TAllocator>::operator == (ArrayData<TElement, TAllocator> &a)
	{
		return !(*this!=a);
	}

	template bool ArrayData<int>::operator == (ArrayData<int> &a);
	template bool ArrayData<float>::operator == (ArrayData<float> &a);
	template bool ArrayData<double>::operator == (ArrayData<double> &a);
	template bool ArrayData<ComplexFloat>::operator == (ArrayData<ComplexFloat> &a);
	template bool ArrayData<ComplexDouble>::operator == (ArrayData<ComplexDouble> &a);

	template<> bool ArrayData<ComplexFloat>::operator != (ArrayData<ComplexFloat> &a)
	{
		bool auxReal, auxImag;
		for(index_t i = 0; i < m_numElements; i++) {
			auxReal = AlmostEqual2sComplement(m_data[i].real(), a.m_data[i].real());
			auxImag = AlmostEqual2sComplement(m_data[i].imag(), a.m_data[i].imag());
			if( auxReal == false || auxImag == false )
				return true;
		}

		return false;
	}

	template<> bool ArrayData<ComplexDouble>::operator != (ArrayData<ComplexDouble> &a)
	{
		bool auxReal, auxImag;
		for(index_t i = 0; i < m_numElements; i++) {
			auxReal = AlmostEqual2sComplement(m_data[i].real(), a.m_data[i].real());
			auxImag = AlmostEqual2sComplement(m_data[i].imag(), a.m_data[i].imag());
			if( auxReal == false || auxImag == false )
				return true;
		}

		return false;
	}

	template <typename TElement, class TAllocator>
	bool ArrayData<TElement, TAllocator>::operator != (ArrayData<TElement, TAllocator> &a)
	{
		for(index_t i = 0; i < m_numElements; i++) {
			if( AlmostEqual2sComplement(m_data[i], a.m_data[i]) == false )
				return true;
		}
		return false;
	}

	//template<> bool ArrayData<ComplexFloat>::operator != (ArrayData<ComplexFloat> &a)
	//{
	//	bool auxReal, auxImag;
	//	for(index_t i = 0; i < m_numElements; i++) {
	//		auxReal = AlmostEqual2sComplement(m_data[i].real(), a.m_data[i].real());
	//		auxImag = AlmostEqual2sComplement(m_data[i].imag(), a.m_data[i].imag());
	//		if( auxReal == false || auxImag == false )
	//			return true;
	//	}
	//
	//	return false;
	//}
	//
	//template<> bool ArrayData<ComplexDouble>::operator != (ArrayData<ComplexDouble> &a)
	//{
	//	bool auxReal, auxImag;
	//	for(index_t i = 0; i < m_numElements; i++) {
	//		auxReal = AlmostEqual2sComplement(m_data[i].real(), a.m_data[i].real());
	//		auxImag = AlmostEqual2sComplement(m_data[i].imag(), a.m_data[i].imag());
	//		if( auxReal == false || auxImag == false )
	//			return true;
	//	}
	//
	//	return false;
	//}
	//
	//template <typename TElement, class TAllocator>
	//bool ArrayData<TElement, TAllocator>::operator != (ArrayData<TElement, TAllocator> &a)
	//{
	//	for(index_t i = 0; i < m_numElements; i++) {
	//		if( AlmostEqual2sComplement(m_data[i], a.m_data[i]) == false )
	//			return true;
	//	}
	//	return false;
	//}
	////{
	////	for(index_t i = 0; i < m_numElements; i++) {
	////		BOOST_CHECK_CLOSE(m_data[i], a.m_data[i], 0.0001f);
	////			return true;
	////	}
	////	return false;
	////}
	////{
	////	for(index_t i = 0; i < m_numElements; i++) {
	////		if (m_data[i] != a.m_data[i])
	////			return true;
	////	}
	////	return false;
	////}

	//template bool ArrayData<int>::operator != (ArrayData<int> &a);
	template bool ArrayData<float>::operator != (ArrayData<float> &a);
	template bool ArrayData<double>::operator != (ArrayData<double> &a);
	//template bool ArrayData<ComplexFloat>::operator != (ArrayData<ComplexFloat> &a);
	//template bool ArrayData<ComplexDouble>::operator != (ArrayData<ComplexDouble> &a);

	template <typename TElement, class TAllocator>
	ArrayData<TElement, TAllocator>& ArrayData<TElement, TAllocator>::operator = (ArrayData<TElement, TAllocator> &a)
	{
		try
		{
			equal<TElement>(m_data, a.m_data, m_numElements, Type2Type<TElement>());
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		return *this;
	}

	template ArrayData<int>& ArrayData<int>::operator = (ArrayData<int> &a);
	template ArrayData<float>& ArrayData<float>::operator = (ArrayData<float> &a);
	template ArrayData<double>& ArrayData<double>::operator = (ArrayData<double> &a);
	template ArrayData<ComplexFloat>& ArrayData<ComplexFloat>::operator = (ArrayData<ComplexFloat> &a);
	template ArrayData<ComplexDouble>& ArrayData<ComplexDouble>::operator = (ArrayData<ComplexDouble> &a);

	template <typename TElement, class TAllocator>
	ArrayData<TElement, TAllocator> ArrayData<TElement, TAllocator>::operator - (ArrayData<TElement, TAllocator> &a)
	{
		ArrayData<TElement> result(*this);
	
		try
		{
			minus<TElement>(result.m_data, m_data, a.m_data, m_numElements, Type2Type<TElement>());
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		return result;
	}

	template ArrayData<int> ArrayData<int>::operator - (ArrayData<int> &a);
	template ArrayData<float> ArrayData<float>::operator - (ArrayData<float> &a);
	template ArrayData<double> ArrayData<double>::operator - (ArrayData<double> &a);
	template ArrayData<ComplexFloat> ArrayData<ComplexFloat>::operator - (ArrayData<ComplexFloat> &a);
	template ArrayData<ComplexDouble> ArrayData<ComplexDouble>::operator - (ArrayData<ComplexDouble> &a);

	template <typename TElement, class TAllocator>
	ArrayData<TElement, TAllocator>& ArrayData<TElement, TAllocator>::operator += (ArrayData<TElement, TAllocator> &a)
	{
		try
		{
			add<TElement>(m_data, m_data, a.m_data, m_numElements, Type2Type<TElement>());
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		return *this;
	}

	template ArrayData<int>& ArrayData<int>::operator += (ArrayData<int> &a);
	template ArrayData<float>& ArrayData<float>::operator += (ArrayData<float> &a);
	template ArrayData<double>& ArrayData<double>::operator += (ArrayData<double> &a);
	template ArrayData<ComplexFloat>& ArrayData<ComplexFloat>::operator += (ArrayData<ComplexFloat> &a);
	template ArrayData<ComplexDouble>& ArrayData<ComplexDouble>::operator += (ArrayData<ComplexDouble> &a);

	template <typename TElement, class TAllocator>
	ArrayData<TElement, TAllocator>& ArrayData<TElement, TAllocator>::operator -= (ArrayData<TElement, TAllocator> &a)
	{
		try
		{
			minus<TElement>(m_data, m_data, a.m_data, m_numElements, Type2Type<TElement>());
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		return *this;
	}

	template ArrayData<int>& ArrayData<int>::operator -= (ArrayData<int> &a);
	template ArrayData<float>& ArrayData<float>::operator -= (ArrayData<float> &a);
	template ArrayData<double>& ArrayData<double>::operator -= (ArrayData<double> &a);
	template ArrayData<ComplexFloat>& ArrayData<ComplexFloat>::operator -= (ArrayData<ComplexFloat> &a);
	template ArrayData<ComplexDouble>& ArrayData<ComplexDouble>::operator -= (ArrayData<ComplexDouble> &a);

	template <typename TElement, class TAllocator>
	ArrayData<TElement, TAllocator> ArrayData<TElement, TAllocator>::operator + (ArrayData<TElement, TAllocator> &a)
	{
		ArrayData<TElement> result(*this);
	
		try
		{
			add<TElement>(result.m_data, m_data, a.m_data, m_numElements, Type2Type<TElement>());
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		return result;
	}

	template ArrayData<int> ArrayData<int>::operator + (ArrayData<int> &a);
	template ArrayData<float> ArrayData<float>::operator + (ArrayData<float> &a);
	template ArrayData<double> ArrayData<double>::operator + (ArrayData<double> &a);
	template ArrayData<ComplexFloat> ArrayData<ComplexFloat>::operator + (ArrayData<ComplexFloat> &a);
	template ArrayData<ComplexDouble> ArrayData<ComplexDouble>::operator + (ArrayData<ComplexDouble> &a);

	template <typename TElement, class TAllocator>
	void ArrayData<TElement, TAllocator>::conj()
	{
	}

	template void ArrayData<int>::conj();
	template void ArrayData<float>::conj();
	template void ArrayData<double>::conj();

	void ArrayData<ComplexFloat>::conj()
	{
		try
		{
			conjugate(m_data, m_data, m_numElements, Type2Type<ComplexFloat>());
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}
	}

	void ArrayData<ComplexDouble>::conj()
	{
		try
		{
			conjugate(m_data, m_data, m_numElements, Type2Type<ComplexDouble>());
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}
	}

	template <typename TElement, class TAllocator>
	void ArrayData<TElement, TAllocator>::transpose(int width, int height)
	{
		try
		{
			transp<TElement>(m_data, m_data, (width*height), Type2Type<TElement>());
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}
	}

	template void ArrayData<int>::transpose(int width, int height);
	template void ArrayData<float>::transpose(int width, int height);
	template void ArrayData<double>::transpose(int width, int height);
	template void ArrayData<ComplexFloat>::transpose(int width, int height);
	template void ArrayData<ComplexDouble>::transpose(int width, int height);

	template <typename TElement, class TAllocator>
	bool ArrayData<TElement, TAllocator>::AlmostEqual2sComplement(double A, double B)
	{
		// Make sure maxUlps is non-negative and small enough that the
		// default NAN won't compare as equal to anything.
		int maxUlps = CONFIDENCE_INTERVAL;
		double maxDiff = CONFIDENCE_INTERVAL_FLOAT;
		double maxRelDiff = 0.01;
		assert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);

		// Check if the numbers are really close -- needed
		// when comparing numbers near zero.
		double diff = fabs(A - B);
		if (diff <= maxDiff)
			return true;

		A = fabs(A);
		B = fabs(B);
		float largest = (B > A) ? B : A;
 
		if (diff <= largest * maxRelDiff)
			return true;
		return false;

		//int aInt = *(int*)&A;
		//// Make aInt lexicographically ordered as a twos-complement int
		//if (aInt < 0)
		//    aInt = 0x80000000 - aInt;
		//// Make bInt lexicographically ordered as a twos-complement int
		//int bInt = *(int*)&B;
		//if (bInt < 0)
		//    bInt = 0x80000000 - bInt;
		//int intDiff = abs(aInt - bInt);
		//if (intDiff <= maxUlps)
		//    return true;
		//return false;
	}

	//template bool ArrayData<int>::AlmostEqual2sComplement(int A, int B);
	//template bool ArrayData<float>::AlmostEqual2sComplement(float A, float B);
	template bool ArrayData<double>::AlmostEqual2sComplement(double A, double B);
	//template bool ArrayData<ComplexFloat>::AlmostEqual2sComplement(ComplexFloat A, ComplexFloat B);
	//template bool ArrayData<ComplexDouble>::AlmostEqual2sComplement(ComplexDouble A, ComplexDouble B);

	template <typename TElement, class TAllocator>
	bool ArrayData<TElement, TAllocator>::AlmostEqualRelativeAndAbs(TElement A, TElement B)
	{
		//TElement diff = boost::math::float_distance( A, B );
		//float maxDiff = 0.01, maxRelDiff = 0.01;
	 //   // Check if the numbers are really close -- needed
	 //   // when comparing numbers near zero.
	 //   TElement diff = fabs(A - B);
	 //   if (diff <= maxDiff)
	 //       return true;
	 //
	 //   A = fabs(A);
	 //   B = fabs(B);
	 //   TElement largest = (B > A) ? B : A;
	 //
	 //   if (diff <= largest * maxRelDiff)
	 //       return true;
		return false;
	}

	//template bool ArrayData<int>::AlmostEqualRelativeAndAbs(int A, int B);
	template bool ArrayData<float>::AlmostEqualRelativeAndAbs(float A, float B);
	template bool ArrayData<double>::AlmostEqualRelativeAndAbs(double A, double B);
	//template bool ArrayData<ComplexFloat>::AlmostEqualRelativeAndAbs(ComplexFloat A, ComplexFloat B);
	//template bool ArrayData<ComplexDouble>::AlmostEqualRelativeAndAbs(ComplexDouble A, ComplexDouble B);

	template<> bool ArrayData<ComplexFloat>::check_close(ArrayData<ComplexFloat> a)
	{
		bool auxReal, auxImag;
		for(index_t i = 0; i < m_numElements; i++) {
			auxReal = AlmostEqual2sComplement(m_data[i].real(), a.m_data[i].real());
			auxImag = AlmostEqual2sComplement(m_data[i].imag(), a.m_data[i].imag());
			if( auxReal == false || auxImag == false )
				return true;
		}

		return false;
	}

	template<> bool ArrayData<ComplexDouble>::check_close(ArrayData<ComplexDouble> a)
	{
		bool auxReal, auxImag;
		for(index_t i = 0; i < m_numElements; i++) {
			auxReal = AlmostEqual2sComplement(m_data[i].real(), a.m_data[i].real());
			auxImag = AlmostEqual2sComplement(m_data[i].imag(), a.m_data[i].imag());
			if( auxReal == false || auxImag == false )
				return true;
		}

		return false;
	}

	template <typename TElement, class TAllocator>
	bool ArrayData<TElement, TAllocator>::check_close (ArrayData<TElement, TAllocator> a)
	{
		for(index_t i = 0; i < m_numElements; i++) {
			if( AlmostEqual2sComplement(m_data[i], a.m_data[i]) == false )
				return true;
		}
		return false;
	}

	template bool ArrayData<int>::check_close(ArrayData<int> a);
	template bool ArrayData<float>::check_close(ArrayData<float> a);
	template bool ArrayData<double>::check_close(ArrayData<double> a);

	//template <typename TElement, class TAllocator>
	//bool ArrayData<TElement, TAllocator>::areEqualRel( TElement a, TElement b ) 
	//{
	//    return (fabs(a - b) <= std::numeric_limits<TElement>::epsilon * std::max(fabs(a), fabs(b)));
	//}
	//
	//template bool ArrayData<int>::areEqualRel(int a, int b);
	//template bool ArrayData<float>::areEqualRel(float a, float b);
	//template bool ArrayData<double>::areEqualRel(double a, double b);
	//template bool ArrayData<ComplexFloat>::areEqualRel(ComplexFloat a, ComplexFloat b);
	//template bool ArrayData<ComplexDouble>::areEqualRel(ComplexDouble a, ComplexDouble b);

	// Array implementation

	template <typename TElement>
	Array<TElement>::Array(	ArrayDescriptor &descriptor,
							const TElement& defaulTElement)
							: m_data(descriptor, defaulTElement),
							m_indexer(NULL),
							m_padded(false)
	{
		m_indexer = new LinearIndexer(descriptor);
	
		// TODO Pass description using pointer to avoid extra object instance
		ArrayUtil::ReleaseArrayDescriptor(descriptor);
	}

	template Array<int>::Array(	ArrayDescriptor &descriptor,
								const int& defaulTElement);
	template Array<float>::Array(	ArrayDescriptor &descriptor,
									const float& defaulTElement);
	template Array<double>::Array(	ArrayDescriptor &descriptor,
									const double& defaulTElement);
	template Array<ComplexFloat>::Array(	ArrayDescriptor &descriptor,
											const ComplexFloat& defaulTElement);
	template Array<ComplexDouble>::Array(	ArrayDescriptor &descriptor,
											const ComplexDouble& defaulTElement);

	template <typename TElement>
	Array<TElement>::Array(	Array &source)
							: m_data(source.m_data),
							m_indexer(NULL),
							m_padded(false)
	{
		m_indexer = new LinearIndexer(source.m_indexer->m_descriptor);
	}

	template Array<int>::Array(Array &source);
	template Array<float>::Array(Array &source);
	template Array<double>::Array(Array &source);
	template Array<ComplexFloat>::Array(Array &source);
	template Array<ComplexDouble>::Array(Array &source);

	template <typename TElement>
	Array<TElement>::~Array() { delete m_indexer; }

	template Array<int>::~Array();
	template Array<float>::~Array();
	template Array<double>::~Array();
	template Array<ComplexFloat>::~Array();
	template Array<ComplexDouble>::~Array();

	template <typename TElement>
	ArrayDescriptor& Array<TElement>::GetDescriptor()
	{
		return m_indexer->GetDescriptor();
	}

	template ArrayDescriptor& Array<int>::GetDescriptor();
	template ArrayDescriptor& Array<float>::GetDescriptor();
	template ArrayDescriptor& Array<double>::GetDescriptor();
	template ArrayDescriptor& Array<ComplexFloat>::GetDescriptor();
	template ArrayDescriptor& Array<ComplexDouble>::GetDescriptor();

	template <typename TElement>
	Array<TElement> Array<TElement>::operator + (Array<TElement> &a)
	{
		Array<TElement> result(*this);
		if(m_indexer->m_descriptor != a.m_indexer->m_descriptor)
			return result;
	
		cublasOperations<TElement> op;
		cublasStatus_t stat;
		try
		{
			stat = op.add( this, &a, &result );
			//stat = op.add2( this, &a, &result, std::string("+") );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		//result.m_data = m_data + a.m_data;
		return result;
	}

	template Array<int> Array<int>::operator + (Array<int> &a);
	template Array<float> Array<float>::operator + (Array<float> &a);
	template Array<double> Array<double>::operator + (Array<double> &a);
	template Array<ComplexFloat> Array<ComplexFloat>::operator + (Array<ComplexFloat> &a);
	template Array<ComplexDouble> Array<ComplexDouble>::operator + (Array<ComplexDouble> &a);

	//template <typename TElement>
	//Array<TElement> Array<TElement>::operator + (Array<float> &a)
	//{
	//	Array<TElement> result(*this);
	//	if(m_indexer->m_descriptor != a.m_indexer->m_descriptor)
	//		return result;
	//	
	//	cublasOperations<TElement> op;
	//	cublasStatus_t stat;
	//	try
	//	{
	//		stat = op.add( this, &a, &result );
	//		//stat = op.add2( this, &a, &result, '+' );
	//	}
	//	catch(std::exception &e)
	//	{
	//		std::cerr << boost::diagnostic_information(e);
	//	}
	//
	//	//result.m_data = m_data + a.m_data;
	//	return result;
	//}
	//template Array<double> Array<double>::operator + (Array<float> &a);

	template <typename TElement>
	Array<TElement> Array<TElement>::operator + (TElement a)
	{ 
		Array<TElement> result = *this;
		for( int i = 0; i < result.m_data.m_numElements; i++ )
			result.m_data.m_data[i] =  (this->m_data.m_data[i]) + a;
	
		return result; 
	}

	template Array<int> Array<int>::operator + (int a);
	template Array<float> Array<float>::operator + (float a);
	template Array<double> Array<double>::operator + (double a);
	template Array<ComplexFloat> Array<ComplexFloat>::operator + (ComplexFloat a);
	template Array<ComplexDouble> Array<ComplexDouble>::operator + (ComplexDouble a);

	//template <typename TElement>
	//Array<TElement>& Array<TElement>::transpose()
	//{
	//	if(m_indexer->GetDescriptor().GetNDim() != 2)
	//		return *this;
	//
	//	cublasOperations<TElement> op;
	//	cublasStatus_t stat;
	//	try
	//	{
	//		//this->print();
	//		stat = op.transpose( this );
	//		//ArrayData::transpose( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(0) );
	//		//transpose();
	//
	//		//m_data.transpose( this->GetDescriptor().GetDim(1), this->GetDescriptor().GetDim(0) );
	//	}
	//	catch(std::exception &e)
	//	{
	//		std::cerr << boost::diagnostic_information(e);
	//		return *this;
	//	}
	//}
	//
	//template Array<int>& Array<int>::transpose();
	//template Array<float>& Array<float>::transpose();
	//template Array<double>& Array<double>::transpose();
	//template Array<ComplexFloat>& Array<ComplexFloat>::transpose();
	//template Array<ComplexDouble>& Array<ComplexDouble>::transpose();

	//template Array<int>& Array<int>::transpose();
	//template Array<float>& Array<float>::transpose();
	//template Array<double>& Array<double>::transpose();
	//template Array<ComplexFloat>& Array<ComplexFloat>::transpose();
	//template Array<ComplexDouble>& Array<ComplexDouble>::transpose();

	template <typename TElement>
	bool Array<TElement>::operator == (Array<TElement>& a)
	{
		if(m_indexer->m_descriptor != a.m_indexer->m_descriptor)
			return false;

		if(m_data != a.m_data)
			return false;

		return true;
	}

	template bool Array<int>::operator == (Array<int>& a);
	template bool Array<float>::operator == (Array<float>& a);
	template bool Array<double>::operator == (Array<double>& a);
	template bool Array<ComplexFloat>::operator == (Array<ComplexFloat>& a);
	template bool Array<ComplexDouble>::operator == (Array<ComplexDouble>& a);

	template <typename TElement>
	bool Array<TElement>::operator != (Array<TElement>& a)
	{
		if(m_indexer->m_descriptor != a.m_indexer->m_descriptor)
			return true;

		if(m_data != a.m_data)
			return true;

		return false;
	}

	template bool Array<int>::operator != (Array<int>& a);
	template bool Array<float>::operator != (Array<float>& a);
	template bool Array<double>::operator != (Array<double>& a);
	template bool Array<ComplexFloat>::operator != (Array<ComplexFloat>& a);
	template bool Array<ComplexDouble>::operator != (Array<ComplexDouble>& a);

	//template <typename TElement>
	//bool Array<TElement>::operator > (Array<TElement>& a)
	//{
	//	if(m_indexer->m_descriptor != a.m_indexer->m_descriptor)
	//		return false;
	//
	//	if(m_data > a.m_data)
	//		return true;
	//
	//	return false;
	//}
	//
	//template bool Array<int>::operator > (Array<int>& a);
	//template bool Array<float>::operator > (Array<float>& a);
	//template bool Array<double>::operator > (Array<double>& a);
	//template bool Array<ComplexFloat>::operator > (Array<ComplexFloat>& a);
	//template bool Array<ComplexDouble>::operator > (Array<ComplexDouble>& a);
	//
	//template <typename TElement>
	//bool Array<TElement>::operator < (Array<TElement>& a)
	//{
	//	if(m_indexer->m_descriptor != a.m_indexer->m_descriptor)
	//		return false;
	//
	//	if(m_data < a.m_data)
	//		return true;
	//
	//	return false;
	//}
	//
	//template bool Array<int>::operator < (Array<int>& a);
	//template bool Array<float>::operator < (Array<float>& a);
	//template bool Array<double>::operator < (Array<double>& a);
	//template bool Array<ComplexFloat>::operator < (Array<ComplexFloat>& a);
	//template bool Array<ComplexDouble>::operator < (Array<ComplexDouble>& a);
	//
	//template <typename TElement>
	//bool Array<TElement>::operator >= (Array<TElement>& a)
	//{
	//	if(m_indexer->m_descriptor != a.m_indexer->m_descriptor)
	//		return false;
	//
	//	if(m_data >= a.m_data)
	//		return true;
	//
	//	return false;
	//}
	//
	//template bool Array<int>::operator >= (Array<int>& a);
	//template bool Array<float>::operator >= (Array<float>& a);
	//template bool Array<double>::operator >= (Array<double>& a);
	//template bool Array<ComplexFloat>::operator >= (Array<ComplexFloat>& a);
	//template bool Array<ComplexDouble>::operator >= (Array<ComplexDouble>& a);
	//
	//template <typename TElement>
	//bool Array<TElement>::operator <= (Array<TElement>& a)
	//{
	//	if(m_indexer->m_descriptor != a.m_indexer->m_descriptor)
	//		return false; // TODO EXIT_FAILURE!
	//
	//	if(m_data <= a.m_data)
	//		return true;
	//
	//	return false;
	//}
	//
	//template bool Array<int>::operator <= (Array<int>& a);
	//template bool Array<float>::operator <= (Array<float>& a);
	//template bool Array<double>::operator <= (Array<double>& a);
	//template bool Array<ComplexFloat>::operator <= (Array<ComplexFloat>& a);
	//template bool Array<ComplexDouble>::operator <= (Array<ComplexDouble>& a);
	
	template <typename TElement>
	const TElement& Array<TElement>::operator () (index_t u, ...) const
	{
		va_list args;
		va_start(args, u);

		m_indexer->m_pos[0] = u;

		int count = m_indexer->m_descriptor.m_count;
		if(m_padded)
		{
			m_indexer->m_pos[1] = 0;
			count--;
		}

		for(int i = 1; i < count; i++)
			m_indexer->m_pos[i] = va_arg(args, index_t);

		va_end(args);
		
		return m_data[m_indexer->GetIndex()];
	}

	template const int& Array<int>::operator () (index_t u, ...) const;
	template const float& Array<float>::operator () (index_t u, ...) const;
	template const double& Array<double>::operator () (index_t u, ...) const;
	template const ComplexFloat& Array<ComplexFloat>::operator () (index_t u, ...) const;
	template const ComplexDouble& Array<ComplexDouble>::operator () (index_t u, ...) const;

	template <typename TElement>
	TElement& Array<TElement>::operator () (index_t u, ...)
	{
		va_list args;
		va_start(args, u);

		m_indexer->m_pos[0] = u;

		int count = m_indexer->m_descriptor.m_count;
		if(m_padded)
		{
			m_indexer->m_pos[1] = 0;
			count--;
		}

		for(int i = 1; i < count; i++)
			m_indexer->m_pos[i] = va_arg(args, index_t);

		va_end(args);
		
		return m_data[m_indexer->GetIndex()];
	}

	template int& Array<int>::operator () (index_t u, ...);
	template float& Array<float>::operator () (index_t u, ...);
	template double& Array<double>::operator () (index_t u, ...);
	template ComplexFloat& Array<ComplexFloat>::operator () (index_t u, ...);
	template ComplexDouble& Array<ComplexDouble>::operator () (index_t u, ...);

	template <typename TElement>
	Array<TElement>& Array<TElement>::operator = (Array<TElement> &a)
	{
		if(m_indexer->m_descriptor != a.m_indexer->m_descriptor)
			return *this;

		m_data = a.m_data;
		
		return *this;
	}

	template Array<int>& Array<int>::operator = (Array<int> &a);
	template Array<float>& Array<float>::operator = (Array<float> &a);
	template Array<double>& Array<double>::operator = (Array<double> &a);
	template Array<ComplexFloat>& Array<ComplexFloat>::operator = (Array<ComplexFloat> &a);
	template Array<ComplexDouble>& Array<ComplexDouble>::operator = (Array<ComplexDouble> &a);

	template <typename TElement>
	Array<TElement>& Array<TElement>::operator = (TElement a)
	{ 
		for( int i = 0; i < this->m_data.m_numElements; i++ )
			this->m_data.m_data[i] = a;
		
		return *this;
	}

	//template Array<int>& Array<int>::operator = (int a);
	template Array<float>& Array<float>::operator = (float a);
	template Array<double>& Array<double>::operator = (double a);
	template Array<ComplexFloat>& Array<ComplexFloat>::operator = (ComplexFloat a);
	template Array<ComplexDouble>& Array<ComplexDouble>::operator = (ComplexDouble a);

	template <typename TElement>
	Array<TElement> Array<TElement>::operator - (Array<TElement> &a)
	{
		Array<TElement> result(*this);

		if(m_indexer->m_descriptor != a.m_indexer->m_descriptor)
			return result;

		cublasOperations<TElement> op;
		cublasStatus_t stat;
		try
		{
			stat = op.minus( this, &a, &result );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		//result.m_data = m_data - a.m_data;

		return result;
	}

	template Array<int> Array<int>::operator - (Array<int> &a);
	template Array<float> Array<float>::operator - (Array<float> &a);
	template Array<double> Array<double>::operator - (Array<double> &a);
	template Array<ComplexFloat> Array<ComplexFloat>::operator - (Array<ComplexFloat> &a);
	template Array<ComplexDouble> Array<ComplexDouble>::operator - (Array<ComplexDouble> &a);

	template <typename TElement>
	Array<TElement> Array<TElement>::operator - (TElement a)
	{ 
		Array<TElement> result = *this;
		for( int i = 0; i < result.m_data.m_numElements; i++ )
			result.m_data.m_data[i] = (this->m_data.m_data[i]) - a;
	
		return result; 
	}

	template Array<int> Array<int>::operator - (int a);
	template Array<float> Array<float>::operator - (float a);
	template Array<double> Array<double>::operator - (double a);
	template Array<ComplexFloat> Array<ComplexFloat>::operator - (ComplexFloat a);
	template Array<ComplexDouble> Array<ComplexDouble>::operator - (ComplexDouble a);

	// C = A x B
	template <typename TElement>
	Array<TElement> Array<TElement>::operator * (Array<TElement> &a)
	{
		if( (m_indexer->GetDescriptor().GetNDim() != 2) || (a.GetDescriptor().GetNDim() != 2) )
			BOOST_ASSERT("operation defined for vector and matrices only\n");

		if(	(m_indexer->GetDescriptor().GetDim(1) != a.m_indexer->m_descriptor.GetDim(0)) ||
				(m_indexer->GetDescriptor().GetNDim() != 2) ||
					(a.GetDescriptor().GetNDim() != 2))
			return *this;
	
		cublasOperations<TElement> op;
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		Array<TElement> result( m_indexer->GetDescriptor().GetDim(0), a.m_indexer->GetDescriptor().GetDim(1) );
		//Array<TElement> result = *this;

		try
		{
			// C = A x B
			stat = op.multiply( this, &a, &result );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		return result;
	}

	template Array<int> Array<int>::operator * (Array<int> &a);
	template Array<float> Array<float>::operator * (Array<float> &a);
	template Array<double> Array<double>::operator * (Array<double> &a);
	template Array<ComplexFloat> Array<ComplexFloat>::operator * (Array<ComplexFloat> &a);
	template Array<ComplexDouble> Array<ComplexDouble>::operator * (Array<ComplexDouble> &a);

	template <typename TElement>
	Array<TElement> Array<TElement>::operator * (TElement a)
	{ 
		Array<TElement> result = *this;
		for( int i = 0; i < result.m_data.m_numElements; i++ )
			result.m_data.m_data[i] =  (this->m_data.m_data[i])*a;
	
		return result; 
	}

	template Array<int> Array<int>::operator * (int a);
	template Array<float> Array<float>::operator * (float a);
	template Array<double> Array<double>::operator * (double a);
	template Array<ComplexFloat> Array<ComplexFloat>::operator * (ComplexFloat a);
	template Array<ComplexDouble> Array<ComplexDouble>::operator * (ComplexDouble a);

	template <typename TElement>
	Array<TElement> Array<TElement>::operator / (Array<TElement> &a)
	{
		if(	(m_indexer->GetDescriptor().GetDim(1) != a.m_indexer->m_descriptor.GetDim(0)) ||
				(m_indexer->GetDescriptor().GetNDim() != 2) ||
					(a.GetDescriptor().GetNDim() != 2))
			return *this;
	
		cublasOperations<TElement> op;
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		Array<TElement> result( m_indexer->GetDescriptor().GetDim(0), a.m_indexer->GetDescriptor().GetDim(1) );

		try
		{
			result = (*this)*a.invert();
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		return result;
	}

	template Array<int> Array<int>::operator / (Array<int> &a);
	template Array<float> Array<float>::operator / (Array<float> &a);
	template Array<double> Array<double>::operator / (Array<double> &a);
	template Array<ComplexFloat> Array<ComplexFloat>::operator / (Array<ComplexFloat> &a);
	template Array<ComplexDouble> Array<ComplexDouble>::operator / (Array<ComplexDouble> &a);

	template <typename TElement>
	Array<TElement> Array<TElement>::operator / (TElement a)
	{ 
		Array<TElement> result = *this;
		for( int i = 0; i < result.m_data.m_numElements; i++ )
			result.m_data.m_data[i] =  (this->m_data.m_data[i])/a;
	
		return result; 
	}

	template Array<int> Array<int>::operator / (int a);
	template Array<float> Array<float>::operator / (float a);
	template Array<double> Array<double>::operator / (double a);
	template Array<ComplexFloat> Array<ComplexFloat>::operator / (ComplexFloat a);
	template Array<ComplexDouble> Array<ComplexDouble>::operator / (ComplexDouble a);

	//template <typename TElement>
	//Array<TElement> Array<TElement>::operator * (float &a){ return *this; }
	//
	//template <typename TElement>
	//Array<TElement> Array<TElement>::operator * (double &a){ return *this; }

	//template Array<int> Array<int>::operator * (int &a);
	//template Array<float> Array<float>::operator * (int &a);
	//template Array<double> Array<double>::operator * (int &a);
	//template Array<ComplexFloat> Array<ComplexFloat>::operator * (int &a);
	//template Array<ComplexDouble> Array<ComplexDouble>::operator * (int &a);
	//template Array<int> Array<int>::operator * (float &a);
	//template Array<float> Array<float>::operator * (float &a);
	//template Array<double> Array<double>::operator * (float &a);
	//template Array<ComplexFloat> Array<ComplexFloat>::operator * (float &a);
	//template Array<ComplexDouble> Array<ComplexDouble>::operator * (float &a);
	//template Array<int> Array<int>::operator * (double &a);
	//template Array<float> Array<float>::operator * (double &a);
	//template Array<double> Array<double>::operator * (double &a);
	//template Array<ComplexFloat> Array<ComplexFloat>::operator * (double &a);
	//template Array<ComplexDouble> Array<ComplexDouble>::operator * (double &a);
	//template Array<ComplexFloat> Array<int>::operator * (ComplexFloat &a);
	//template Array<ComplexFloat> Array<float>::operator * (ComplexFloat &a);
	//template Array<ComplexFloat> Array<double>::operator * (ComplexFloat &a);
	//template Array<ComplexFloat> Array<ComplexFloat>::operator * (ComplexFloat &a);
	//template Array<ComplexFloat> Array<ComplexDouble>::operator * (ComplexFloat &a);
	//template Array<ComplexDouble> Array<int>::operator * (ComplexDouble &a);
	//template Array<ComplexDouble> Array<float>::operator * (ComplexDouble &a);
	//template Array<ComplexDouble> Array<double>::operator * (ComplexDouble &a);
	//template Array<ComplexDouble> Array<ComplexFloat>::operator * (ComplexDouble &a);
	//template Array<ComplexDouble> Array<ComplexDouble>::operator * (ComplexDouble &a);

	template <typename TElement>
	Array<TElement>& Array<TElement>::operator += (Array<TElement> &a)
	{
		if(m_indexer->m_descriptor != a.m_indexer->m_descriptor)
			return *this;

		*this = *this + a;
		//m_data += a.m_data;
		
		return *this;
	}

	template Array<int>& Array<int>::operator += (Array<int> &a);
	template Array<float>& Array<float>::operator += (Array<float> &a);
	template Array<double>& Array<double>::operator += (Array<double> &a);
	template Array<ComplexFloat>& Array<ComplexFloat>::operator += (Array<ComplexFloat> &a);
	template Array<ComplexDouble>& Array<ComplexDouble>::operator += (Array<ComplexDouble> &a);

	template <typename TElement>
	Array<TElement>& Array<TElement>::operator -= (Array<TElement> &a)
	{
		if(m_indexer->m_descriptor != a.m_indexer->m_descriptor)
			return *this;

		*this = *this - a;
		//m_data -= a.m_data;
		
		return *this;
	}

	template Array<int>& Array<int>::operator -= (Array<int> &a);
	template Array<float>& Array<float>::operator -= (Array<float> &a);
	template Array<double>& Array<double>::operator -= (Array<double> &a);
	template Array<ComplexFloat>& Array<ComplexFloat>::operator -= (Array<ComplexFloat> &a);
	template Array<ComplexDouble>& Array<ComplexDouble>::operator -= (Array<ComplexDouble> &a);

	template <typename TElement>
	Array<TElement>& Array<TElement>::operator *= (Array<TElement> &a)
	{
		if(	(m_indexer->GetDescriptor().GetDim(1) != a.m_indexer->m_descriptor.GetDim(0)) ||
				(m_indexer->GetDescriptor().GetNDim() != 2) ||
					(a.GetDescriptor().GetNDim() != 2))
			return *this;

		// TODO Resize this operand
		//if((m_indexer->GetDescriptor().GetDim(1) < a.m_indexer->m_descriptor.GetDim(1)))
		//	return *this;

		cublasOperations<TElement> op;
		cublasStatus_t stat;
		Array<TElement> aux = *this;
		try
		{	
			stat = op.multiply( &aux, &a, this );
			//multiply(	m_data.GetElements(),
			//			m_data.GetElements(),
			//			a.m_data.GetElements(),
			//			m_indexer->GetDescriptor().GetDim(1),
			//			m_indexer->GetDescriptor().GetDim(0),
			//			a.m_indexer->GetDescriptor().GetDim(1),
			//			Type2Type<TElement>());
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		return *this;
	}

	template Array<int>& Array<int>::operator *= (Array<int> &a);
	template Array<float>& Array<float>::operator *= (Array<float> &a);
	template Array<double>& Array<double>::operator *= (Array<double> &a);
	template Array<ComplexFloat>& Array<ComplexFloat>::operator *= (Array<ComplexFloat> &a);
	template Array<ComplexDouble>& Array<ComplexDouble>::operator *= (Array<ComplexDouble> &a);

	template <typename TElement>
	Array<TElement> Array<TElement>::transpose()
	{
		if(m_indexer->GetDescriptor().GetNDim() != 2)
			return *this;

		Array<TElement> result = *this;
		cublasOperations<TElement> op;
		cublasStatus_t stat;
		try
		{
			CUBLAS_CALL( op.transpose( &result ) );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
			result = *this;
		}

		return result;
	}

	template Array<int> Array<int>::transpose();
	template Array<float> Array<float>::transpose();
	template Array<double> Array<double>::transpose();
	template Array<ComplexFloat> Array<ComplexFloat>::transpose();
	template Array<ComplexDouble> Array<ComplexDouble>::transpose();

	template <typename TElement>
	Array<TElement> Array<TElement>::conjugate()
	{
		return this->conj();
	}

	template Array<int> Array<int>::conjugate();
	template Array<float> Array<float>::conjugate();
	template Array<double> Array<double>::conjugate();
	template Array<ComplexFloat> Array<ComplexFloat>::conjugate();
	template Array<ComplexDouble> Array<ComplexDouble>::conjugate();

	template <typename TElement>
	Array<TElement> Array<TElement>::conj()
	{
		Array<TElement> result = *this;
		result.m_data.conj();
		return result;
	}
	//{
	//	if(m_indexer->GetDescriptor().GetNDim() != 2)
	//		return *this;
	//
	//	cublasOperations<TElement> op;
	//	cublasStatus_t stat;
	//	try
	//	{
	//		stat = op.conjugate( this );
	//	}
	//	catch(std::exception &e)
	//	{
	//		std::cerr << boost::diagnostic_information(e);
	//		return *this;
	//	}
	//}

	template Array<int> Array<int>::conj();
	template Array<float> Array<float>::conj();
	template Array<double> Array<double>::conj();
	template Array<ComplexFloat> Array<ComplexFloat>::conj();
	template Array<ComplexDouble> Array<ComplexDouble>::conj();

	template <typename TElement>
	Array<TElement> Array<TElement>::hermitian()
	{
		if(m_indexer->GetDescriptor().GetNDim() != 2)
			return *this;

		cublasOperations<TElement> op;
		cublasStatus_t stat;
		Array<TElement> result = *this;
		try
		{
			CUBLAS_CALL( op.hermitian( &result ) );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
			result = *this;
		}

		return result;
	}

	//{
	//	transpose();
	//	conj();
	//	return *this;
	//}

	template Array<int> Array<int>::hermitian();
	template Array<float> Array<float>::hermitian();
	template Array<double> Array<double>::hermitian();
	template Array<ComplexFloat> Array<ComplexFloat>::hermitian();
	template Array<ComplexDouble> Array<ComplexDouble>::hermitian();

	// implementation of determinant

	template<> ComplexFloat Array<ComplexFloat>::determinant()
	{
		cublasOperations<ComplexFloat> op;
		cublasStatus_t stat;
		ComplexFloat det = ComplexFloat(1,0);

		if( m_data.m_numElements == 1)
			return (*this)(0,0);

		Array<ComplexFloat> LU( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );
		Array<ComplexFloat> Pivot = eye<ComplexFloat>( this->GetDescriptor().GetDim(0) );

		try
		{
			stat = op.LU( this, &LU, &Pivot );
		
			for( int i = 0; i < m_indexer->GetDescriptor().GetDim(1); i++ )
				det *= LU(i,i);
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		return det;
	}

	template<> ComplexDouble Array<ComplexDouble>::determinant()
	{
		cublasOperations<ComplexDouble> op;
		cublasStatus_t stat;
		ComplexDouble det = ComplexDouble(1,0);

		if( m_data.m_numElements == 1)
			return (*this)(0,0);

		Array<ComplexDouble> LU( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );
		Array<ComplexDouble> Pivot = eye<ComplexDouble>( this->GetDescriptor().GetDim(0) );

		try
		{
			stat = op.LU( this, &LU, &Pivot );
		
			for( int i = 0; i < m_indexer->GetDescriptor().GetDim(1); i++ )
				det *= LU(i,i);
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		return det;
	}

	template <typename TElement>
	TElement Array<TElement>::determinant()
	{
		cublasOperations<TElement> op;
		cublasStatus_t stat;
		TElement det = 1;

		if( m_data.m_numElements == 1)
			return (*this)(0,0);

		Array<TElement> LU( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );
		Array<TElement> Pivot = eye<TElement>( this->GetDescriptor().GetDim(0) );

		try
		{
			stat = op.LU( this, &LU, &Pivot );
		
			if( stat == CUBLAS_STATUS_SUCCESS ) {
				for( int i = 0; i < m_indexer->GetDescriptor().GetDim(1); i++ )
					det *= LU(i,i);
			}
			else
				det = 0;
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		return det;
	}

	template int Array<int>::determinant();
	template float Array<float>::determinant();
	template double Array<double>::determinant();

	// implementation of LU decomposition

	template<> void Array<ComplexFloat>::LU( Array<ComplexFloat> *L, Array<ComplexFloat> *U, Array<ComplexFloat> *P )
	{
		cublasOperations<ComplexFloat> op;
		cublasStatus_t stat;

		if( m_data.m_numElements == 1)
		{
			(*L)(0) = ComplexFloat(1,0);
			(*U)(0) = (*this)(0);
			return;
		}

		Array<ComplexFloat> LU( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );
		*P = eye<ComplexFloat>( this->GetDescriptor().GetDim(0) );

		try
		{
			stat = op.LU( this, &LU, P );
			if( stat == CUBLAS_STATUS_SUCCESS )
			{
				*L = eye<ComplexFloat>( LU.GetDescriptor().GetDim(0) );
				for( int i = 0; i < LU.GetDescriptor().GetDim(0); i++ ) {
					for( int j = 0; j < i; j++ )
						(*L)( i, j ) = LU( i, j );
				}
				for( int i = 0; i < LU.GetDescriptor().GetDim(0); i++ ) {
					for( int j = i; j < LU.GetDescriptor().GetDim(0); j++ )
						(*U)( i, j ) = LU( i, j );
				}
			}
			else
				std::cout << "LU decomposition failed" << std::endl;

		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}
	}

	template<> void Array<ComplexDouble>::LU( Array<ComplexDouble> *L, Array<ComplexDouble> *U, Array<ComplexDouble> *P )
	{
		cublasOperations<ComplexDouble> op;
		cublasStatus_t stat;

		if( m_data.m_numElements == 1)
		{
			(*L)(0) = ComplexDouble(1,0);
			(*U)(0) = (*this)(0);
			return;
		}

		Array<ComplexDouble> LU( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );
		*P = eye<ComplexDouble>( this->GetDescriptor().GetDim(0) );

		try
		{
			stat = op.LU( this, &LU, P );
			if( stat == CUBLAS_STATUS_SUCCESS )
			{
				*L = eye<ComplexDouble>( LU.GetDescriptor().GetDim(0) );
				for( int i = 0; i < LU.GetDescriptor().GetDim(0); i++ ) {
					for( int j = 0; j < i; j++ )
						(*L)( i, j ) = LU( i, j );
				}
				for( int i = 0; i < LU.GetDescriptor().GetDim(0); i++ ) {
					for( int j = i; j < LU.GetDescriptor().GetDim(0); j++ )
						(*U)( i, j ) = LU( i, j );
				}
			}
			else
				std::cout << "LU decomposition failed" << std::endl;

		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}
	}

	template <typename TElement>
	void Array<TElement>::LU( Array<TElement> *L, Array<TElement> *U, Array<TElement> *P )
	{
		cublasOperations<TElement> op;
		cublasStatus_t stat;
		TElement det = 1;

		if( m_data.m_numElements == 1)
		{
			(*L)(0) = 1;
			(*U)(0) = (*this)(0);
			return;
		}

		Array<TElement> LU( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );
		*P = eye<TElement>( this->GetDescriptor().GetDim(0) );

		try
		{
			stat = op.LU( this, &LU, P );
			if( stat == CUBLAS_STATUS_SUCCESS )
			{
				*L = eye<TElement>( LU.GetDescriptor().GetDim(0) );
				for( int i = 0; i < LU.GetDescriptor().GetDim(0); i++ ) {
					for( int j = 0; j < i; j++ )
						(*L)( i, j ) = LU( i, j );
				}
				for( int i = 0; i < LU.GetDescriptor().GetDim(0); i++ ) {
					for( int j = i; j < LU.GetDescriptor().GetDim(0); j++ )
						(*U)( i, j ) = LU( i, j );
				}
			}
			else
				std::cout << "LU decomposition failed" << std::endl;

		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}
	}

	template void Array<int>::LU( Array<int> *L, Array<int> *U, Array<int> *P );
	template void Array<float>::LU( Array<float> *L, Array<float> *U, Array<float> *P );
	template void Array<double>::LU( Array<double> *L, Array<double> *U, Array<double> *P );

	// implementation of invert

	template<> Array<ComplexFloat> Array<ComplexFloat>::invert()
	{
		cublasOperations<ComplexFloat> op;
		cublasStatus_t stat;

		if( m_data.m_numElements == 1)
		{
			Array<ComplexFloat> result(1);
			result(0) = ComplexFloat(1,0)/(*this)(0);
			return result;
		}
	
		Array<ComplexFloat> result = *this;
		Array<ComplexFloat> LU( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );
		Array<int> Pivot( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );

		try
		{
			stat = op.invert( &result, &LU, &Pivot );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		return result;
	}
	template<> Array<ComplexDouble> Array<ComplexDouble>::invert()
	{
		cublasOperations<ComplexDouble> op;
		cublasStatus_t stat;

		if( m_data.m_numElements == 1)
		{
			Array<ComplexDouble> result(1);
			result(0) = ComplexDouble(1,0)/(*this)(0);
			return result;
		}
	
		Array<ComplexDouble> result = *this;
		Array<ComplexDouble> LU( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );
		Array<int> Pivot( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );

		try
		{
			stat = op.invert( &result, &LU, &Pivot );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		return result;
	}

	template <typename TElement>
	Array<TElement> Array<TElement>::invert()
	{
		cublasOperations<TElement> op;
		cublasStatus_t stat;

		if( m_data.m_numElements == 1)
		{
			Array<TElement> result(1);
			result(0) = 1.0/(*this)(0);
			return result;
		}
	
		Array<TElement> result = *this;
		Array<TElement> LU( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );
		Array<int> Pivot( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );

		try
		{
			stat = op.invert( &result, &LU, &Pivot );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		return result;
	}

	template Array<int> Array<int>::invert();
	template Array<float> Array<float>::invert();
	template Array<double> Array<double>::invert();

	// implementation of LS solution

	//template<> Array<ComplexFloat> Array<ComplexFloat>::LS( Array<ComplexFloat> A )
	//{
	//	cublasOperations<ComplexFloat> op;
	//	cublasStatus_t stat;
	//
	//	if( m_data.m_numElements == 1)
	//	{
	//		Array<ComplexFloat> result(1);
	//		result(0) = ComplexFloat(1,0)/(*this)(0);
	//		return result;
	//	}
	//	
	//	Array<ComplexFloat> result = *this;
	//	Array<ComplexFloat> LU( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );
	//	Array<int> Pivot( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );
	//
	//	try
	//	{
	//		stat = op.invert( &result, &LU, &Pivot );
	//	}
	//	catch(std::exception &e)
	//	{
	//		std::cerr << boost::diagnostic_information(e);
	//	}
	//
	//	return result;
	//}
	//
	//template<> Array<ComplexDouble> Array<ComplexDouble>::LS( Array<ComplexDouble> A )
	//{
	//	cublasOperations<ComplexDouble> op;
	//	cublasStatus_t stat;
	//
	//	if( m_data.m_numElements == 1)
	//	{
	//		Array<ComplexDouble> result(1);
	//		result(0) = ComplexDouble(1,0)/(*this)(0);
	//		return result;
	//	}
	//	
	//	Array<ComplexDouble> result = *this;
	//	Array<ComplexDouble> LU( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );
	//	Array<int> Pivot( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );
	//
	//	try
	//	{
	//		stat = op.invert( &result, &LU, &Pivot );
	//	}
	//	catch(std::exception &e)
	//	{
	//		std::cerr << boost::diagnostic_information(e);
	//	}
	//
	//	return result;
	//}

	template <typename TElement>
	Array<TElement> Array<TElement>::LS( Array<TElement> A )
	{
		cublasOperations<TElement> op;
		cublasStatus_t stat;
	
		Array<TElement> result( A.GetDescriptor().GetDim( 1 ), this->GetDescriptor().GetDim( 1 ) );
		Array<TElement> aux = *this;

		try
		{
			stat = op.LS( A, &result, aux );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		return result;
	}

	template Array<int> Array<int>::LS( Array<int> A );
	template Array<float> Array<float>::LS( Array<float> A );
	template Array<double> Array<double>::LS( Array<double> A );
	template Array<ComplexFloat> Array<ComplexFloat>::LS( Array<ComplexFloat> A );
	template Array<ComplexDouble> Array<ComplexDouble>::LS( Array<ComplexDouble> A );

	// implementation of detrend

	template <typename TElement>
	Array<TElement> Array<TElement>::detrend()
	{
		Array<TElement> x( this->GetDescriptor().GetDim(0), 2 );
		Array<TElement> result( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );
		Array<TElement> aux( this->GetDescriptor().GetDim(0) );

		for( int i = 0; i < this->GetDescriptor().GetDim(1); i++ ) {

			for( int j = 0; j < x.GetDescriptor().GetDim(0); j++ ) {
				aux(j) = (*this)(j,i);
				x(j,0) = j;
				x(j,1) = 1;
			}

			Array<TElement> p = aux.LS( x );
			aux = aux - x*p;

			for( int j = 0; j < x.GetDescriptor().GetDim(0); j++ )
				result(j,i) = aux(j);
		}

		return result;
	}

	template Array<int> Array<int>::detrend();
	template Array<float> Array<float>::detrend();
	template Array<double> Array<double>::detrend();
	template Array<ComplexFloat> Array<ComplexFloat>::detrend();
	template Array<ComplexDouble> Array<ComplexDouble>::detrend();

	// implementation of diff

	template <typename TElement>
	Array<TElement> Array<TElement>::diff()
	{
		Array<TElement> result( this->GetDescriptor().GetDim(0) - 1, this->GetDescriptor().GetDim(1) );

		for( int i = 0; i < this->GetDescriptor().GetDim(1); i++ ) {
			for( int j = 0; j < this->GetDescriptor().GetDim(0) - 1; j++ )
				result(j,i) = (*this)(j+1,i) - (*this)(j,i);
		}

		return result;
	}

	template Array<int> Array<int>::diff();
	template Array<float> Array<float>::diff();
	template Array<double> Array<double>::diff();
	template Array<ComplexFloat> Array<ComplexFloat>::diff();
	template Array<ComplexDouble> Array<ComplexDouble>::diff();

	//template <typename TElement>
	//Array<TElement>& Array<TElement>::invert()
	//{
	//	if(m_indexer->GetDescriptor().GetNDim() != 2 ||
	//		m_indexer->GetDescriptor().GetDim(0) != m_indexer->GetDescriptor().GetDim(1))
	//			return *this;
	//
	//	cublasHandle_t handle;
	//	
	//	TElement *h_matrix = m_data.m_data;
	//	TElement *d_matrix = NULL;
	//	int rows = m_indexer->GetDescriptor().GetDim(0);
	//	int cols = m_indexer->GetDescriptor().GetDim(1);
	//	int size = rows*cols;
	//	
	//	try
	//	{
	//		if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
	//			throw std::exception();
	//
	//		if(cudaMalloc((void**)&d_matrix, size*sizeof(TElement)) != cudaSuccess)
	//			throw std::exception();
	//
	//		if(cublasSetMatrix(rows, cols, sizeof(TElement), h_matrix, rows, d_matrix, cols) != CUBLAS_STATUS_SUCCESS)
	//    		throw std::exception();
	//
	//		// Perform LU factorization
	//		invertLU(handle, rows, d_matrix);
	//
	//		if(cublasGetMatrix(rows, cols, sizeof(TElement), d_matrix, rows, h_matrix, cols) != CUBLAS_STATUS_SUCCESS)
	//    		throw std::exception();
	//	}
	//	catch(std::exception &e)
	//	{
	//		std::cerr << boost::diagnostic_information(e);
	//	}
	//		
	//	// Cleanup
	//	cudaFree(d_matrix);
	//	cublasDestroy(handle);
	//
	//	return *this;
	//}
	//
	//template Array<int>& Array<int>::invert();
	//template Array<float>& Array<float>::invert();
	//template Array<double>& Array<double>::invert();
	//
	//Array<ComplexFloat>& Array<ComplexFloat>::invert()
	//{
	//	if(m_indexer->GetDescriptor().GetNDim() != 2 ||
	//		m_indexer->GetDescriptor().GetDim(0) != m_indexer->GetDescriptor().GetDim(1))
	//			return *this;
	//
	//	cublasHandle_t handle;
	//	Array<float> *matrix = NULL;
	//	float *d_matrix = NULL;
	//
	//	try
	//	{
	//		int rows = m_indexer->GetDescriptor().GetDim(0) << 1;
	//		int cols = m_indexer->GetDescriptor().GetDim(1) << 1;
	//		int size = rows*cols;
	//	
	//		matrix = new Array<float>(rows, cols);
	//		int N = m_indexer->GetDescriptor().GetDim(0);
	//		for(int i = 0; i < N; i++)
	//		{
	//		   for(int j = 0; j < N; j++)
	//		   {
	//			   (*matrix)(i, j) = (*this)(i, j).real();
	//			   (*matrix)(i , j + N) = (*this)(i, j).imag();
	//
	//			   (*matrix)(i + N, j) = (*this)(i, j).imag() * (-1);
	//			   (*matrix)(i + N, j + N) = (*this)(i, j).real();
	//		   }
	//		}
	//	
	//		float *h_matrix = matrix->GetArrayData()->m_data;
	//		
	//		if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
	//			throw std::exception();
	//
	//		if(cudaMalloc((void**)&d_matrix, size*sizeof(float)) != cudaSuccess)
	//			throw std::exception();
	//
	//		if(cublasSetMatrix(rows, cols, sizeof(float), h_matrix, rows, d_matrix, cols) != CUBLAS_STATUS_SUCCESS)
	//    		throw std::exception();
	//
	//		// Perform LU factorization
	//		invertLU(handle, rows, d_matrix);
	//
	//		if(cublasGetMatrix(rows, cols, sizeof(float), d_matrix, rows, h_matrix, cols) != CUBLAS_STATUS_SUCCESS)
	//    		throw std::exception();
	//
	//		for(int i = 0; i < N; i++)
	//		{
	//		   for(int j = 0; j < N; j++)
	//		   {
	//			   (*this)(i, j).real((*matrix)(i , j));
	//			   (*this)(i, j).imag((*matrix)(i , j + N));
	//		   }
	//		}
	//	}
	//	catch(std::exception &e)
	//	{
	//		std::cerr << boost::diagnostic_information(e);
	//	}
	//		
	//	// Cleanup
	//	cudaFree(d_matrix);
	//	delete matrix;
	//	cublasDestroy(handle);
	//
	//	return *this;
	//}
	//
	//Array<ComplexDouble>& Array<ComplexDouble>::invert()
	//{
	//	if(m_indexer->GetDescriptor().GetNDim() != 2 ||
	//		m_indexer->GetDescriptor().GetDim(0) != m_indexer->GetDescriptor().GetDim(1))
	//			return *this;
	//
	//	cublasHandle_t handle;
	//	Array<double> *matrix = NULL;
	//	double *d_matrix = NULL;
	//	try
	//	{
	//
	//		int rows = m_indexer->GetDescriptor().GetDim(0) << 1;
	//		int cols = m_indexer->GetDescriptor().GetDim(1) << 1;
	//		int size = rows*cols;
	//	
	//		matrix = new Array<double>(rows, cols);
	//		int N = m_indexer->GetDescriptor().GetDim(0);
	//		for(int i = 0; i < N; i++)
	//		{
	//		   for(int j = 0; j < N; j++)
	//		   {
	//			   (*matrix)(i, j) = (*this)(i, j).real();
	//			   (*matrix)(i , j + N) = (*this)(i, j).imag();
	//
	//			   (*matrix)(i + N, j) = (*this)(i, j).imag() * (-1);
	//			   (*matrix)(i + N, j + N) = (*this)(i, j).real();
	//		   }
	//		}
	//	
	//		double *h_matrix = matrix->GetArrayData()->m_data;
	//		
	//		if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
	//			throw std::exception();
	//
	//		if(cudaMalloc((void**)&d_matrix, size*sizeof(double)) != cudaSuccess)
	//			throw std::exception();
	//
	//		if(cublasSetMatrix(rows, cols, sizeof(double), h_matrix, rows, d_matrix, cols) != CUBLAS_STATUS_SUCCESS)
	//    		throw std::exception();
	//
	//		// Perform LU factorization
	//		invertLU(handle, rows, d_matrix);
	//
	//		if(cublasGetMatrix(rows, cols, sizeof(double), d_matrix, rows, h_matrix, cols) != CUBLAS_STATUS_SUCCESS)
	//    		throw std::exception();
	//
	//		for(int i = 0; i < N; i++)
	//		{
	//		   for(int j = 0; j < N; j++)
	//		   {
	//			   (*this)(i, j).real((*matrix)(i , j));
	//			   (*this)(i, j).imag((*matrix)(i , j + N));
	//		   }
	//		}
	//	}
	//	catch(std::exception &e)
	//	{
	//		std::cerr << boost::diagnostic_information(e);
	//	}
	//		
	//	// Cleanup
	//	cudaFree(d_matrix);
	//	delete matrix;
	//	cublasDestroy(handle);
	//
	//	return *this;
	//}
	//
	//template <typename TElement>
	//void Array<TElement>::invertLU(void *hnd, int n, int *d_matrix)
	//{
	//	// TODO Throw a different exception type since critical
	//	throw std::exception("Operation not supported!!!");
	//}
	//
	//template <typename TElement>
	//void Array<TElement>::invertLU(void *hnd, int n, float *d_matrix)
	//{
	//	int *pivots = NULL;
	//	float *d_work = NULL;
	//
	//	try
	//	{
	//		cublasHandle_t handle = static_cast<cublasHandle_t>(hnd);
	//
	//		// Perform LU decomposition
	//		pivots = new int[n];
	//		for (int i = 0; i < n; i++)
	//		{
	//			pivots[i] = i;
	//		}
	//
	//		float factor[] = { 0.0 };
	//		float *offset = NULL;
	//		int pivot = 0;
	//		float alfa = 0;
	//		for(int i = 0; i < n - 1; i++)
	//		{
	//			offset = d_matrix + (i * n + i);
	//			cublasIsamax(handle, n - i, offset, 1, &pivot);
	//			pivot = i - 1 + pivot;
	//			if(pivot != i)
	//			{
	//				pivots[i] = pivot;
	//				cublasSswap(handle, n, d_matrix + pivot, n, d_matrix + i, n);
	//			}
	//
	//			cublasGetVector(1, sizeof(float), offset, 1, factor, 1);
	//        
	//			alfa = 1 / factor[0];
	//			cublasSscal(handle, n - i - 1, &alfa, offset + 1, 1);
	//		
	//			alfa = -1.0;
	//			cublasSger(handle, n - i - 1, n - i - 1, &alfa, offset + 1, 1, offset + n, n, offset + (n + 1), n);
	//		}
	//
	//		float det = 0;
	//		::determinant(&det, d_matrix, n);
	//		if(det == 0)
	//			throw std::exception("Determinant is zero!!! Inverse does not exist!!!");
	//
	//		// Perform inv(U)
	//		for(int i = 0; i < n; i++)
	//		{
	//			float *offset = d_matrix + (i * n);
	//			cublasGetVector(1, sizeof(float), offset + i, 1, factor, 1);
	//			factor[0] = 1 / factor[0];
	//			cublasSetVector(1, sizeof(float), factor, 1, offset + i, 1);
	//		
	//			cublasStrmv(handle,
	//						CUBLAS_FILL_MODE_UPPER,
	//						CUBLAS_OP_N,
	//						CUBLAS_DIAG_NON_UNIT,
	//						i, d_matrix, n, offset, 1);
	//        
	//			alfa = -factor[0];
	//			cublasSscal(handle, i, &alfa, offset, 1);
	//		}
	//
	//		float beta = 0;
	//	
	//		// Solve inv(A)*L = inv(U)
	//		if(cudaMalloc((void**)&d_work, (n - 1)*sizeof(float)) != cudaSuccess)
	//			throw std::exception();
	//		
	//		for(int i = n - 1; i > 0; i--)
	//        {
	//			float *offset = d_matrix + ((i - 1) * n + i);
	//
	//			if(cudaMemcpy(d_work, offset, (n - 1) * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess)
	//				throw std::exception();
	//			
	//			alfa = 0;
	//			cublasSscal(handle, n - i, &alfa, offset, 1);
	//            
	//			alfa = -1;
	//			beta = 1;
	//			cublasSgemv(handle, CUBLAS_OP_N,
	//						n, n - i, &alfa,
	//						d_matrix + (i * n), n,
	//						d_work, 1, &beta,
	//						d_matrix + ((i - 1) * n), 1);
	//		}
	//
	//		// Pivot back to original order
	//        for (int i = n - 1; i >= 0; i--)
	//        {
	//            if (i != pivots[i])
	//            {
	//                cublasSswap(handle, n, d_matrix + (i * n), 1, d_matrix + (pivots[i] * n), 1);
	//            }
	//        }
	//
	//	} catch(...)
	//	{
	//		delete[] pivots;
	//		cudaFree(d_work);
	//		throw;
	//	}
	//
	//	delete[] pivots;
	//	cudaFree(d_work);
	//}
	//
	//template <typename TElement>
	//void Array<TElement>::invertLU(void *hnd, int n, double *d_matrix)
	//{
	//	int *pivots = NULL;
	//	double *d_work = NULL;
	//	
	//	try
	//	{
	//		cublasHandle_t handle = static_cast<cublasHandle_t>(hnd);
	//
	//		// Perform LU decomposition
	//		pivots = new int[n];
	//		for (int i = 0; i < n; i++)
	//		{
	//			pivots[i] = i;
	//		}
	//
	//		float factor[] = { 0.0 };
	//		int pivot = 0;
	//		double *offset = NULL;
	//		double alfa = 0;
	//		for(int i = 0; i < n - 1; i++)
	//		{
	//			offset = d_matrix + (i * n + i);
	//			cublasIdamax(handle, n - i, offset, 1, &pivot);
	//			pivot = i - 1 + pivot;
	//			if(pivot != i)
	//			{
	//				pivots[i] = pivot;
	//				cublasDswap(handle, n, d_matrix + pivot, n, d_matrix + i, n);
	//			}
	//
	//			cublasGetVector(1, sizeof(double), offset, 1, factor, 1);
	//        
	//			alfa = 1 / factor[0];
	//			cublasDscal(handle, n - i - 1, &alfa, offset + 1, 1);
	//		
	//			alfa = -1.0;
	//			cublasDger(handle, n - i - 1, n - i - 1, &alfa, offset + 1, 1, offset + n, n, offset + (n + 1), n);
	//		}
	//
	//		double det = 0;
	//		::determinant(&det, d_matrix, n);
	//		if(det == 0)
	//			throw std::exception("Determinant is zero!!! Inverse does not exist!!!");
	//
	//		// Perform inv(U)
	//		for(int i = 0; i < n; i++)
	//		{
	//			double *offset = d_matrix + (i * n);
	//			cublasGetVector(1, sizeof(double), offset + i, 1, factor, 1);
	//			factor[0] = 1 / factor[0];
	//			cublasSetVector(1, sizeof(double), factor, 1, offset + i, 1);
	//		
	//			cublasDtrmv(handle,
	//						CUBLAS_FILL_MODE_UPPER,
	//						CUBLAS_OP_N,
	//						CUBLAS_DIAG_NON_UNIT,
	//						i, d_matrix, n, offset, 1);
	//        
	//			alfa = -factor[0];
	//			cublasDscal(handle, i, &alfa, offset, 1);
	//		}
	//
	//		double beta = 0;
	//		// Solve inv(A)*L = inv(U)
	//		if(cudaMalloc((void**)&d_work, (n - 1)*sizeof(double)) != cudaSuccess)
	//			throw std::exception();
	//		
	//		for(int i = n - 1; i > 0; i--)
	//        {
	//			double *offset = d_matrix + ((i - 1) * n + i);
	//
	//			if(cudaMemcpy(d_work, offset, (n - 1) * sizeof(double), cudaMemcpyDeviceToDevice) != cudaSuccess)
	//				throw std::exception();
	//			
	//			alfa = 0;
	//			cublasDscal(handle, n - i, &alfa, offset, 1);
	//            
	//			alfa = -1;
	//			beta = 1;
	//			cublasDgemv(handle, CUBLAS_OP_N,
	//						n, n - i, &alfa,
	//						d_matrix + (i * n), n,
	//						d_work, 1, &beta,
	//						d_matrix + ((i - 1) * n), 1);
	//		}
	//
	//		// Pivot back to original order
	//        for (int i = n - 1; i >= 0; i--)
	//        {
	//            if (i != pivots[i])
	//            {
	//                cublasDswap(handle, n, d_matrix + (i * n), 1, d_matrix + (pivots[i] * n), 1);
	//            }
	//        }
	//
	//	} catch(...)
	//	{
	//		delete[] pivots;
	//		cudaFree(d_work);
	//		throw;
	//	}
	//
	//	delete[] pivots;
	//	cudaFree(d_work);
	//}

	// implementation of minor

	template <typename TElement>
	Array<TElement> Array<TElement>::minor( const int row, const int column )
	{
		if(m_indexer->GetDescriptor().GetNDim() != 2 ||
		   !( row >= 0 && row < m_indexer->GetDescriptor().GetDim(0) && column >= 0 && column < m_indexer->GetDescriptor().GetDim(1) ))
			return *this;
	
		int dimRows = m_indexer->GetDescriptor().GetDim(0);
		int dimColumns = m_indexer->GetDescriptor().GetDim(1);
	 
		Array<TElement> res( dimRows - 1, dimColumns - 1 );
		try
		{   for ( int c = 0; c < m_indexer->GetDescriptor().GetDim(1); c++ )
			{
				if( c == column )
					continue;
				for ( int r = 0; r < m_indexer->GetDescriptor().GetDim(0); r++ )
				{
					if( r == row )
						continue;
					res( r - ( r > row ), ( c - ( c > column) ) ) = m_data[ c*m_indexer->GetDescriptor().GetDim(0) + r ];
				}			
				
			}
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}
	
		return res;
	}

	template Array<int> Array<int>::minor(const int row, const int column);
	template Array<float> Array<float>::minor(int row, const int column);
	template Array<double> Array<double>::minor(const int row, const int column);
	template Array<ComplexFloat> Array<ComplexFloat>::minor(const int row, const int column);
	template Array<ComplexDouble> Array<ComplexDouble>::minor(const int row, const int column);

	// implementation of print

	template <typename TElement>
	void Array<TElement>::print()
	{
		std::cout << std::endl;
		switch(GetDescriptor().GetNDim())
		{
			case 1:
				for(int i = 0; i < GetDescriptor().GetDim(0); i++)
					std::cout << (*this)(i) << " ";			
				std::cout << std::endl;
				break;
			case 2:
				for(int i = 0; i < GetDescriptor().GetDim(0); i++)
				{
					for(int j = 0; j < GetDescriptor().GetDim(1); j++)
						std::cout << (*this)(i, j) << " ";			
					std::cout << std::endl;
				}
				break;
			case 3:
				for(int i = 0; i < GetDescriptor().GetDim(0); i++)
				{
					for(int j = 0; j < GetDescriptor().GetDim(1); j++)
					{
						for(int k = 0; k < GetDescriptor().GetDim(2); k++)
							std::cout << (*this)(i, j, k) << " ";			
						std::cout << std::endl;
					}
					std::cout << std::endl;
				}
				break;
		}	
		std::cout << std::endl;

		return;
	}

	template void Array<int>::print();
	template void Array<float>::print();
	template void Array<double>::print();
	template void Array<ComplexFloat>::print();
	template void Array<ComplexDouble>::print();

	// implementation of norm

	template <typename TElement>
	TElement Array<TElement>::norm()
	{
		if(	(m_indexer->GetDescriptor().GetDim(1) != 1 &&
			 m_indexer->GetDescriptor().GetDim(0) != 1) ||
			(m_indexer->GetDescriptor().GetNDim() != 2) )
			return -1;
	
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		Array<TElement> A = *this;
		Array<TElement> B = this->hermitian();
		Array<TElement> result( 1 );

		if( A.GetDescriptor().GetDim(0) == 1 )
			result = A*B;
		else
			result = B*A;

		return std::sqrt( result(0) );
	}

	template int Array<int>::norm();
	template float Array<float>::norm();
	template double Array<double>::norm();
	template ComplexFloat Array<ComplexFloat>::norm();
	template ComplexDouble Array<ComplexDouble>::norm();

	// implementation of check_close

	template <typename TElement>
	bool Array<TElement>::check_close(Array<TElement> a)
	{	
		if(m_indexer->m_descriptor != a.m_indexer->m_descriptor)
			return true;

		if( !m_data.check_close( a.m_data ) )
			return true;

		return false;
	}

	template bool Array<int>::check_close(Array<int> a);
	template bool Array<float>::check_close(Array<float> a);
	template bool Array<double>::check_close(Array<double> a);
	template bool Array<ComplexFloat>::check_close(Array<ComplexFloat> a);
	template bool Array<ComplexDouble>::check_close(Array<ComplexDouble> a);

	// implementation of fft

	template<> Array<ComplexFloat> Array<ComplexFloat>::fft()
	{
		cufftOperations<ComplexFloat> op;
		cufftResult_t stat;
		Array<ComplexFloat> result( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );
	
		stat = op.fft( this, &result );
		return result;
	}

	template<> Array<ComplexDouble> Array<ComplexDouble>::fft()
	{
		cufftOperations<ComplexDouble> op;
		cufftResult_t stat;
		Array<ComplexDouble> result( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );

		stat = op.fft( this, &result );
		return result;
	}

	//Array<ComplexDouble> Array<double>::fft()
	//{
	//	cufftOperations<ComplexDouble> op;
	//	cufftResult_t stat;
	//	Array<ComplexDouble> result( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );
	//
	//	stat = op.fft( this, &result );
	//	return result;
	//}

	// implementation of trigonometric functions

	template <typename TElement>
	Array<TElement> Array<TElement>::sind()
	{
		Array<TElement> result( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );

		for( int i = 0; i < m_data.m_numElements; i++ )
			result.m_data.m_data[i] = boost::math::sin_pi(this->m_data.m_data[i]/180);

		return result;
	}

	template Array<float> Array<float>::sind();
	template Array<double> Array<double>::sind();
	//template Array<ComplexFloat> Array<ComplexFloat>::sind();
	//template Array<ComplexDouble> Array<ComplexDouble>::sind();

	template <typename TElement>
	Array<TElement> Array<TElement>::sin()
	{
		Array<TElement> result( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );

		for( int i = 0; i < m_data.m_numElements; i++ )
			result.m_data.m_data[i] = boost::math::sin_pi(this->m_data.m_data[i]/boost::math::constants::pi<TElement>());

		return result;
	}

	template Array<float> Array<float>::sin();
	template Array<double> Array<double>::sin();

	template <typename TElement>
	Array<TElement> Array<TElement>::cosd()
	{
		Array<TElement> result( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );

		for( int i = 0; i < m_data.m_numElements; i++ )
			result.m_data.m_data[i] = boost::math::cos_pi(this->m_data.m_data[i]/180);

		return result;
	}

	template Array<float> Array<float>::cosd();
	template Array<double> Array<double>::cosd();

	template <typename TElement>
	Array<TElement> Array<TElement>::cos()
	{
		Array<TElement> result( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );

		for( int i = 0; i < m_data.m_numElements; i++ )
			result.m_data.m_data[i] = boost::math::cos_pi(this->m_data.m_data[i]/boost::math::constants::pi<TElement>());

		return result;
	}

	template Array<float> Array<float>::cos();
	template Array<double> Array<double>::cos();

	template <typename TElement>
	Array<TElement> Array<TElement>::tand()
	{
		Array<TElement> result( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );

		for( int i = 0; i < m_data.m_numElements; i++ )
			result.m_data.m_data[i] = boost::math::sin_pi(this->m_data.m_data[i]/180)/boost::math::cos_pi(this->m_data.m_data[i]/180);

		return result;
	}

	template Array<float> Array<float>::tand();
	template Array<double> Array<double>::tand();

	template <typename TElement>
	Array<TElement> Array<TElement>::tan()
	{
		Array<TElement> result( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );

		for( int i = 0; i < m_data.m_numElements; i++ )
			result.m_data.m_data[i] = boost::math::sin_pi(this->m_data.m_data[i]/boost::math::constants::pi<TElement>())/boost::math::cos_pi(this->m_data.m_data[i]/boost::math::constants::pi<TElement>());

		return result;
	}

	template Array<float> Array<float>::tan();
	template Array<double> Array<double>::tan();

	template <typename TElement>
	Array<TElement> Array<TElement>::asind()
	{
		Array<TElement> result( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );

		for( int i = 0; i < m_data.m_numElements; i++ )
			result.m_data.m_data[i] = 180/boost::math::constants::pi<TElement>()*std::asin(this->m_data.m_data[i]);

		return result;
	}

	template Array<float> Array<float>::asind();
	template Array<double> Array<double>::asind();

	template <typename TElement>
	Array<TElement> Array<TElement>::asin()
	{
		Array<TElement> result( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );

		for( int i = 0; i < m_data.m_numElements; i++ )
			result.m_data.m_data[i] = std::asin(this->m_data.m_data[i]);

		return result;
	}

	template Array<float> Array<float>::asin();
	template Array<double> Array<double>::asin();

	template <typename TElement>
	Array<TElement> Array<TElement>::acosd()
	{
		Array<TElement> result( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );

		for( int i = 0; i < m_data.m_numElements; i++ )
			result.m_data.m_data[i] = 180/boost::math::constants::pi<TElement>()*std::acos(this->m_data.m_data[i]);

		return result;
	}

	template Array<float> Array<float>::acosd();
	template Array<double> Array<double>::acosd();

	template <typename TElement>
	Array<TElement> Array<TElement>::acos()
	{
		Array<TElement> result( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );

		for( int i = 0; i < m_data.m_numElements; i++ )
			result.m_data.m_data[i] = std::acos(this->m_data.m_data[i]);

		return result;
	}

	template Array<float> Array<float>::acos();
	template Array<double> Array<double>::acos();

	template <typename TElement>
	Array<TElement> Array<TElement>::atand()
	{
		Array<TElement> result( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );

		for( int i = 0; i < m_data.m_numElements; i++ )
			result.m_data.m_data[i] = 180/boost::math::constants::pi<TElement>()*std::atan(this->m_data.m_data[i]);

		return result;
	}

	template Array<float> Array<float>::atand();
	template Array<double> Array<double>::atand();

	template <typename TElement>
	Array<TElement> Array<TElement>::atan()
	{
		Array<TElement> result( this->GetDescriptor().GetDim(0), this->GetDescriptor().GetDim(1) );

		for( int i = 0; i < m_data.m_numElements; i++ )
			result.m_data.m_data[i] = std::atan(this->m_data.m_data[i]);

		return result;
	}

	template Array<float> Array<float>::atan();
	template Array<double> Array<double>::atan();

	// implementation of getColumn

	template <typename TElement>
	Array<TElement> Array<TElement>::getColumn( index_t col )
	{
		//va_list args;
		//va_start(args, u);
		//
		//m_indexer->m_pos[0] = u;
		//
		//int count = m_indexer->m_descriptor.m_count;
		//if(m_padded)
		//{
		//	m_indexer->m_pos[1] = 0;
		//	count--;
		//}
		//
		//for(int i = 1; i < count; i++)
		//	m_indexer->m_pos[i] = va_arg(args, index_t);
		//
		//va_end(args);
		//if( (m_indexer->GetDescriptor().GetNDim() != 2) )
		//	return *this;

		Array<TElement> result( this->GetDescriptor().GetDim( 0 ) );

		for( int i = 0; i < this->GetDescriptor().GetDim( 0 ); i++ )
			result( i ) = (*this)( i, col );

		return result;
	}

	template Array<int> Array<int>::getColumn(index_t col);
	template Array<float> Array<float>::getColumn(index_t col);
	template Array<double> Array<double>::getColumn(index_t col);
	template Array<ComplexFloat> Array<ComplexFloat>::getColumn(index_t col);
	template Array<ComplexDouble> Array<ComplexDouble>::getColumn(index_t col);

	// implementation of QR decomposition

	template <typename TElement>
	void Array<TElement>::QR( Array<TElement> *Q, Array<TElement> *R )
	{
		cublasOperations<TElement> op;
		cublasStatus_t stat;

		if( m_data.m_numElements == 1)
		{
			(*Q)(0) = 1;
			(*R)(0) = (*this)(0);
			return;
		}

		try
		{
			stat = op.QR( this, Q, R );
			if( stat != CUBLAS_STATUS_SUCCESS )
				std::cout << "QR decomposition failed" << std::endl;
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}
	}

	template void Array<int>::QR( Array<int> *Q, Array<int> *R );
	template void Array<float>::QR( Array<float> *Q, Array<float> *R );
	template void Array<double>::QR( Array<double> *Q, Array<double> *R );
	template void Array<ComplexFloat>::QR( Array<ComplexFloat> *Q, Array<ComplexFloat> *R );
	template void Array<ComplexDouble>::QR( Array<ComplexDouble> *Q, Array<ComplexDouble> *R );

	template <typename TElement>
	Array<TElement> Array<TElement>::eig( Array<TElement> *eigenvectors )
	{
		Array<TElement> result( this->GetDescriptor().GetDim( 0 ) );
		Array<TElement> aux = *this;

		cublasOperations<TElement> op;
		cublasStatus_t stat;

		if( m_data.m_numElements == 1)
		{
			(*eigenvectors)(0) = 1;
			result(0) = (*this)(0);
			return result;
		}

		try
		{
			stat = op.eig( &aux, eigenvectors );
			if( stat != CUBLAS_STATUS_SUCCESS )
				std::cout << "eigenvalues/eigenvectors calculation failed" << std::endl;
			else
			{
				for( int i = 0; i < this->GetDescriptor().GetDim(0); i++ )
					result(i) = aux( i, i );
			}
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}
	
		return result;
	}

	template Array<int> Array<int>::eig( Array<int> *eigenvectors );
	template Array<float> Array<float>::eig( Array<float> *eigenvectors );
	template Array<double> Array<double>::eig( Array<double> *eigenvectors );
	template Array<ComplexFloat> Array<ComplexFloat>::eig( Array<ComplexFloat> *eigenvectors );
	template Array<ComplexDouble> Array<ComplexDouble>::eig( Array<ComplexDouble> *eigenvectors );

	template <typename TElement>
	void Array<TElement>::Array2cuSparseCooMatrix( int n, int nnz, int *cooRowIndexHostPtr, 
															  int *cooColIndexHostPtr, TElement *cooValHostPtr )
	{
		cooRowIndexHostPtr = (int *)malloc(nnz*sizeof(cooRowIndexHostPtr[0])); 
		cooColIndexHostPtr = (int *)malloc(nnz*sizeof(cooColIndexHostPtr[0])); 
		cooValHostPtr = (TElement *)malloc(nnz*sizeof(cooValHostPtr[0])); 
		if ((!cooRowIndexHostPtr) || 
			(!cooColIndexHostPtr) || 
			(!cooValHostPtr))
		{ 
			printf("Host malloc failed (matrix)\n"); 
			exit(-1); 
		}

		int N = this->GetDescriptor().GetDim(0);
		int row = 0, col = 0, counter = 0;
		for (int i = 0; i < this->m_data.m_numElements; i++ ) {
			if( this->m_data.m_data[i] != 0 ) {
				cooColIndexHostPtr[counter] = i/this->GetDescriptor().GetDim(1);
				cooRowIndexHostPtr[counter] = i%this->GetDescriptor().GetDim(0);
				cooValHostPtr[counter++] = this->m_data.m_data[i];
			}
		}
	}

	template void Array<int>::Array2cuSparseCooMatrix( int n, int nnz, int *cooRowIndexHostPtr, 
															  int *cooColIndexHostPtr, int *cooValHostPtr );
	template void Array<float>::Array2cuSparseCooMatrix( int n, int nnz, int *cooRowIndexHostPtr, 
															  int *cooColIndexHostPtr, float *cooValHostPtr );
	template void Array<double>::Array2cuSparseCooMatrix( int n, int nnz, int *cooRowIndexHostPtr, 
															  int *cooColIndexHostPtr, double *cooValHostPtr );
	//template void Array<ComplexFloat>::Array2cuSparseCooMatrix( int n, int nnz, int *cooRowIndexHostPtr, 
	//														  int *cooColIndexHostPtr, ComplexFloat *cooValHostPtr );
	//template void Array<ComplexDouble>::Array2cuSparseCooMatrix( int n, int nnz, int *cooRowIndexHostPtr, 
	//														  int *cooColIndexHostPtr, ComplexDouble *cooValHostPtr );

	template <typename TElement>
	std::string Array<TElement>::arrayname2str()
	{
		return VAR_NAME_PRINTER(*this);
	}

	template std::string Array<float>::arrayname2str();
	template std::string Array<double>::arrayname2str();
	template std::string Array<ComplexFloat>::arrayname2str();
	template std::string Array<ComplexDouble>::arrayname2str();

	template <typename TElement>
	void Array<TElement>::write2file( std::string s )
	{
		std::ofstream myfile;
		myfile.open( s );
		if( myfile.good() )
		{
			for( int i = 0; i < this->GetDescriptor().GetDim( 0 ); i++ ) {
				for( int j = 0; j < this->GetDescriptor().GetDim( 1 ); j++ )
					myfile << (*this)( i, j ) << " ";
				myfile << std::endl;
			}

			myfile.close();
		}
	}

	template void Array<float>::write2file( std::string s );
	template void Array<double>::write2file( std::string s );

	template <typename TElement>
	Array<TElement> Array<TElement>::submatrix( const index_t rowBegin,
												const index_t rowEnd, 
												const index_t colBegin, 
												const index_t colEnd )
	{
		if( rowEnd < rowBegin || colEnd < colBegin ) {
			std::cout << "invalid paramenters for submatrix function" << std::endl;
			exit(-23);
		}

		Array<TElement> result( rowEnd - rowBegin + 1, colEnd - colBegin + 1 );
		for( int i = rowBegin; i <= rowEnd; i++ ) {
			for( int j = colBegin; j <= colEnd; j++ )
				result( i - rowBegin, j - colBegin ) = (*this)( i, j );
		}

		return result;
	}

	template Array<float> Array<float>::submatrix( index_t rowBegin, index_t rowEnd, index_t colBegin, index_t colEnd );
	template Array<double> Array<double>::submatrix( index_t rowBegin, index_t rowEnd, index_t colBegin, index_t colEnd );
	template Array<ComplexFloat> Array<ComplexFloat>::submatrix( index_t rowBegin, index_t rowEnd, index_t colBegin, index_t colEnd );
	template Array<ComplexDouble> Array<ComplexDouble>::submatrix( index_t rowBegin, index_t rowEnd, index_t colBegin, index_t colEnd );

	template <typename TElement>
	Array<TElement> Array<TElement>::max()
	{
		TElement *d_vec;
		Array<TElement> result( 1, this->GetDescriptor().GetDim( 1 ) );
		int rows = this->GetDescriptor().GetDim( 0 );

		CUDA_CALL( cudaMalloc((void**)&d_vec, rows * sizeof(TElement)) );

		TElement *pos = &(this->m_data.m_data[0]);
		int auxIdx;

		for( int i = 0; i < this->GetDescriptor().GetDim( 1 ); i++ )
		{
			CUDA_CALL( cudaMemcpy( d_vec, (void*)(pos), rows * sizeof(TElement), cudaMemcpyHostToDevice) );

			auxIdx = -1;
			result( 0, i ) = max_cuda( d_vec, &auxIdx, rows );

			pos += rows;
		}

		CUDA_CALL( cudaFree( d_vec ) );

		return result;
	}

	template Array<float> Array<float>::max();
	template Array<double> Array<double>::max();

	template <typename TElement>
	Array<TElement> Array<TElement>::max( Array<TElement> *idx )
	{
		TElement *d_vec;
		Array<TElement> result( 1, this->GetDescriptor().GetDim( 1 ) );
		int rows = this->GetDescriptor().GetDim( 0 );

		CUDA_CALL( cudaMalloc((void**)&d_vec, rows * sizeof(TElement)) );

		TElement *pos = &(this->m_data.m_data[0]);
		int auxIdx;

		for( int i = 0; i < this->GetDescriptor().GetDim( 1 ); i++ )
		{
			CUDA_CALL( cudaMemcpy( d_vec, (void*)(pos), rows * sizeof(TElement), cudaMemcpyHostToDevice) );

			auxIdx = -1;
			result( 0, i ) = max_cuda( d_vec, &auxIdx, rows );
			(*idx)( 0, i ) = auxIdx;

			pos += rows;
		}

		CUDA_CALL( cudaFree( d_vec ) );

		return result;
	}

	template Array<float> Array<float>::max( Array<float> *idx );
	template Array<double> Array<double>::max( Array<double> *idx );

	template <typename TElement>
	Array<TElement> Array<TElement>::min()
	{
		TElement *d_vec;
		Array<TElement> result( 1, this->GetDescriptor().GetDim( 1 ) );
		int rows = this->GetDescriptor().GetDim( 0 );

		CUDA_CALL( cudaMalloc((void**)&d_vec, rows * sizeof(TElement)) );

		TElement *pos = &(this->m_data.m_data[0]);
		int auxIdx;

		for( int i = 0; i < this->GetDescriptor().GetDim( 1 ); i++ )
		{
			CUDA_CALL( cudaMemcpy( d_vec, (void*)(pos), rows * sizeof(TElement), cudaMemcpyHostToDevice) );

			auxIdx = -1;
			result( 0, i ) = min_cuda( d_vec, &auxIdx, rows );

			pos += rows;
		}

		CUDA_CALL( cudaFree( d_vec ) );

		return result;
	}

	template Array<float> Array<float>::min();
	template Array<double> Array<double>::min();

	template <typename TElement>
	Array<TElement> Array<TElement>::min( Array<TElement> *idx )
	{
		TElement *d_vec;
		Array<TElement> result( 1, this->GetDescriptor().GetDim( 1 ) );
		int rows = this->GetDescriptor().GetDim( 0 );

		CUDA_CALL( cudaMalloc((void**)&d_vec, rows * sizeof(TElement)) );

		TElement *pos = &(this->m_data.m_data[0]);
		int auxIdx;

		for( int i = 0; i < this->GetDescriptor().GetDim( 1 ); i++ )
		{
			CUDA_CALL( cudaMemcpy( d_vec, (void*)(pos), rows * sizeof(TElement), cudaMemcpyHostToDevice) );

			auxIdx = -1;
			result( 0, i ) = min_cuda( d_vec, &auxIdx, rows );
			(*idx)( 0, i ) = auxIdx;

			pos += rows;
		}

		CUDA_CALL( cudaFree( d_vec ) );

		return result;
	}

	template Array<float> Array<float>::min( Array<float> *idx );
	template Array<double> Array<double>::min( Array<double> *idx );

	// TODO
	// find roots pf polynomial using bisection
	// method with tolerance TOL_BISECT_METHOD;

	template <typename TElement>
	Array<TElement> Array<TElement>::roots()
	{
		double TOL_BISECT_METHOD;
		if( (m_indexer->GetDescriptor().GetNDim() != 2) )
			return *this;

		Array<TElement> result( this->GetDescriptor().GetDim( 0 ) );

		return result;
	}

	//template Array<int> Array<int>::roots( Array<int> polynomial );
	template Array<float> Array<float>::roots();
	template Array<double> Array<double>::roots();
	template Array<ComplexFloat> Array<ComplexFloat>::roots();
	template Array<ComplexDouble> Array<ComplexDouble>::roots();

	ArrayDescriptor& ArrayUtil::GetArrayDescriptor(int count, ...)
	{
		ArrayDescriptor *arr_desc = new ArrayDescriptor(count);

		va_list args;
		va_start(args, count);

		for(int i = 0; i < count; i++)
			arr_desc->m_dim[i] = va_arg(args, index_t);

		va_end(args);

		return *arr_desc;
	}

	ArrayDescriptor& ArrayUtil::GetArrayDescriptor(ArrayDescriptor &source)
	{
		ArrayDescriptor *arr_desc = new ArrayDescriptor(source);
		return *arr_desc;
	}

	void ArrayUtil::ReleaseArrayDescriptor(ArrayDescriptor &descriptor)
	{
		delete &descriptor;
	}
}

// CudaDevice implementation
CudaDevice::CudaDevice()
{
	cudaSetDevice(0);
	cudaSetDeviceFlags(cudaDeviceMapHost);
};