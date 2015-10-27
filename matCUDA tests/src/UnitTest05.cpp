#include "unitTests.h"

#define sizeSmall 4
#define sizeLarge 1024

// test of call to Array
void call_to_array()
{
	Array<int> v1(sizeSmall);
	v1.print();

	Array<int> v2(1,sizeSmall);
	v2.print();
	for(int i = 0; i < sizeSmall; i++ )
		v2(0,i) = i;
	v2.print();

	Array<Complex> v3(1,sizeSmall*2);
	v3.print();
	for(int i = 0; i < sizeSmall*2; i++ )
		v3(0,i) = Complex(i,-i);
	v3.print();

	Array<Complex> m1(sizeSmall,sizeSmall);
	for(int i = 0; i < sizeSmall; i++ ) {
		for(int j = 0; j < sizeSmall; j++ )
			m1(i,j) = Complex(i,j);
	}
	m1.print();

	Array<Complex> m2 = m1;
	m2.print();
}

// small vector
void test_transpose_float_1() 
{
	Array<float> v1(sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = float(i)/10;

	Array<float> v2(1,sizeSmall);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 1 ); i++)
		v2(0,i) = float(i)/10;

	TEST_CALL( v1.transpose(), v2, BOOST_CURRENT_FUNCTION );
}

// large vector
void test_transpose_float_2() 
{
	Array<float> v1(sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = float(i)/10;

	Array<float> v2(1,sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 1 ); i++)
		v2(0,i) = float(i)/10;

	TEST_CALL( v1.transpose(), v2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_transpose_float_3() 
{	
	Array<float> m1(sizeSmall, sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = float(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<float> m2(sizeSmall, sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = float(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10;
	}

	TEST_CALL( m1.transpose(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_transpose_float_4()
{
	Array<float> m1(sizeLarge, sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = float(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<float> m2(sizeLarge, sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = float(j*m1.GetDescriptor().GetDim( 1 ) + i + 1)/10;
	}

	TEST_CALL( m1.transpose(), m2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_transpose_float_5() 
{	
	Array<float> m1(sizeSmall, 2*sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = float(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<float> m2(2*sizeSmall, sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = float(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10;
	}

	TEST_CALL( m1.transpose(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_transpose_float_6()
{
	Array<float> m1(sizeLarge, 2*sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = float(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<float> m2(2*sizeLarge, sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = float(j*m1.GetDescriptor().GetDim( 1 ) + i + 1)/10;
	}

	TEST_CALL( m1.transpose(), m2, BOOST_CURRENT_FUNCTION );
}

// small vector
void test_conjugate_float_1()  
{
	Array<float> v1(sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = float(i)/10;

	Array<float> v2(sizeSmall);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = float(i)/10;

	TEST_CALL( v1.conj(), v2, BOOST_CURRENT_FUNCTION );
}

// large vector
void test_conjugate_float_2()  
{
	Array<float> v1(sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = float(i)/10;

	Array<float> v2(sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = float(i)/10;

	TEST_CALL( v1.conj(), v2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_conjugate_float_3()  
{	
	Array<float> m1(sizeSmall, sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = float(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<float> m2(sizeSmall, sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = float(i*m2.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	TEST_CALL( m1.conj(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_conjugate_float_4() 
{
	Array<float> m1(sizeLarge, sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = float(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<float> m2(sizeLarge, sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = float(i*m2.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	TEST_CALL( m1.conj(), m2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_conjugate_float_5()  
{	
	Array<float> m1(sizeSmall, 2*sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = float(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<float> m2(sizeSmall, 2*sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = float(i*m2.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	TEST_CALL( m1.conj(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_conjugate_float_6() 
{
	Array<float> m1(sizeLarge, 2*sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = float(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<float> m2(sizeLarge, 2*sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = float(i*m2.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	TEST_CALL( m1.conj(), m2, BOOST_CURRENT_FUNCTION );
}

// small vector
void test_hermitian_float_1()  
{
	Array<float> v1(sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = float(i)/10;

	Array<float> v2(1,sizeSmall);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 1 ); i++)
		v2(0,i) = float(i)/10;

	TEST_CALL( v1.hermitian(), v2, BOOST_CURRENT_FUNCTION );
}

// large vector
void test_hermitian_float_2()  
{
	Array<float> v1(sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = float(i)/10;

	Array<float> v2(1,sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 1 ); i++)
		v2(0,i) = float(i)/10;

	TEST_CALL( v1.hermitian(), v2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_hermitian_float_3()  
{	
	Array<float> m1(sizeSmall, sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = float(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<float> m2(sizeSmall, sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = float(j*m2.GetDescriptor().GetDim( 1 ) + i + 1)/10;
	}

	TEST_CALL( m1.hermitian(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_hermitian_float_4() 
{
	Array<float> m1(sizeLarge, sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = float(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<float> m2(sizeLarge, sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = float(j*m2.GetDescriptor().GetDim( 1 ) + i + 1)/10;
	}

	TEST_CALL( m1.hermitian(), m2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_hermitian_float_5()  
{	
	Array<float> m1(sizeSmall, 2*sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = float(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<float> m2(2*sizeSmall, sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = float(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10;
	}

	TEST_CALL( m1.hermitian(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_hermitian_float_6() 
{
	Array<float> m1(sizeLarge, 2*sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = float(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<float> m2(2*sizeLarge, sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = float(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10;
	}

	TEST_CALL( m1.hermitian(), m2, BOOST_CURRENT_FUNCTION );
}

// small vector
void test_transpose_double_1() 
{
	Array<double> v1(sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = double(i)/10;

	Array<double> v2(1,sizeSmall);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 1 ); i++)
		v2(0,i) = double(i)/10;

	TEST_CALL( v1.transpose(), v2, BOOST_CURRENT_FUNCTION );
}

// large vector
void test_transpose_double_2() 
{
	Array<double> v1(sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = double(i)/10;

	Array<double> v2(1,sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 1 ); i++)
		v2(0,i) = double(i)/10;

	TEST_CALL( v1.transpose(), v2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_transpose_double_3() 
{	
	Array<double> m1(sizeSmall, sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<double> m2(sizeSmall, sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = double(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10;
	}

	TEST_CALL( m1.transpose(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_transpose_double_4()
{
	Array<double> m1(sizeLarge, sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<double> m2(sizeLarge, sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = double(j*m1.GetDescriptor().GetDim( 1 ) + i + 1)/10;
	}

	TEST_CALL( m1.transpose(), m2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_transpose_double_5() 
{	
	Array<double> m1(sizeSmall, 2*sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<double> m2(2*sizeSmall, sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = double(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10;
	}

	TEST_CALL( m1.transpose(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_transpose_double_6()
{
	Array<double> m1(sizeLarge, 2*sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<double> m2(2*sizeLarge, sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = double(j*m1.GetDescriptor().GetDim( 1 ) + i + 1)/10;
	}

	TEST_CALL( m1.transpose(), m2, BOOST_CURRENT_FUNCTION );
}

// small vector
void test_conjugate_double_1()  
{
	Array<double> v1(sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = double(i)/10;

	Array<double> v2(sizeSmall);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = double(i)/10;

	TEST_CALL( v1.conj(), v2, BOOST_CURRENT_FUNCTION );
}

// large vector
void test_conjugate_double_2()  
{
	Array<double> v1(sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = double(i)/10;

	Array<double> v2(sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = double(i)/10;

	TEST_CALL( v1.conj(), v2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_conjugate_double_3()  
{	
	Array<double> m1(sizeSmall, sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<double> m2(sizeSmall, sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = double(i*m2.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	TEST_CALL( m1.conj(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_conjugate_double_4() 
{
	Array<double> m1(sizeLarge, sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<double> m2(sizeLarge, sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = double(i*m2.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	TEST_CALL( m1.conj(), m2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_conjugate_double_5()  
{	
	Array<double> m1(sizeSmall, 2*sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<double> m2(sizeSmall, 2*sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = double(i*m2.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	TEST_CALL( m1.conj(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_conjugate_double_6() 
{
	Array<double> m1(sizeLarge, 2*sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<double> m2(sizeLarge, 2*sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = double(i*m2.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	TEST_CALL( m1.conj(), m2, BOOST_CURRENT_FUNCTION );
}

// small vector
void test_hermitian_double_1()  
{
	Array<double> v1(sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = double(i)/10;

	Array<double> v2(1,sizeSmall);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 1 ); i++)
		v2(0,i) = double(i)/10;

	TEST_CALL( v1.hermitian(), v2, BOOST_CURRENT_FUNCTION );
}

// large vector
void test_hermitian_double_2()  
{
	Array<double> v1(sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = double(i)/10;

	Array<double> v2(1,sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 1 ); i++)
		v2(0,i) = double(i)/10;

	TEST_CALL( v1.hermitian(), v2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_hermitian_double_3()  
{	
	Array<double> m1(sizeSmall, sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<double> m2(sizeSmall, sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = double(j*m2.GetDescriptor().GetDim( 1 ) + i + 1)/10;
	}

	TEST_CALL( m1.hermitian(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_hermitian_double_4() 
{
	Array<double> m1(sizeLarge, sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<double> m2(sizeLarge, sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = double(j*m2.GetDescriptor().GetDim( 1 ) + i + 1)/10;
	}

	TEST_CALL( m1.hermitian(), m2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_hermitian_double_5()  
{	
	Array<double> m1(sizeSmall, 2*sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<double> m2(2*sizeSmall, sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = double(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10;
	}

	TEST_CALL( m1.hermitian(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_hermitian_double_6() 
{
	Array<double> m1(sizeLarge, 2*sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	}

	Array<double> m2(2*sizeLarge, sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = double(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10;
	}

	TEST_CALL( m1.hermitian(), m2, BOOST_CURRENT_FUNCTION );
}

// small vector
void test_transpose_complex_1()  
{
	Array<Complex> v1(sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = Complex(i,i);

	Array<Complex> v2(1,sizeSmall);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 1 ); i++)
		v2(0,i) = Complex(i,i);

	TEST_CALL( v1.transpose(), v2, BOOST_CURRENT_FUNCTION );
}

// large vector
void test_transpose_complex_2() 
{
	Array<Complex> v1(sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = Complex(double(i)/10,double(i)/10);

	Array<Complex> v2(1,sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 1 ); i++)
		v2(0,i) = Complex(double(i)/10,double(i)/10);

	TEST_CALL( v1.transpose(), v2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_transpose_complex_3() 
{	
	Array<Complex> m1(sizeSmall, sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10,double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10);
	}

	Array<Complex> m2(sizeSmall, sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(double(j*m2.GetDescriptor().GetDim( 1 ) + i + 1)/10,double(j*m2.GetDescriptor().GetDim( 1 ) + i + 1)/10);
	}

	TEST_CALL( m1.transpose(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_transpose_complex_4()
{
	Array<Complex> m1(sizeLarge, sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10,double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10);
	}

	Array<Complex> m2(sizeLarge, sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(double(j*m2.GetDescriptor().GetDim( 1 ) + i + 1)/10,double(j*m2.GetDescriptor().GetDim( 1 ) + i + 1)/10);
	}

	TEST_CALL( m1.transpose(), m2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_transpose_complex_5() 
{	
	Array<Complex> m1(sizeSmall, 2*sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10,double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10);
	}

	Array<Complex> m2(2*sizeSmall, sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(double(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10,double(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10);
	}

	TEST_CALL( m1.transpose(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_transpose_complex_6()
{
	Array<Complex> m1(sizeLarge, 2*sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10,double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10);
	}

	Array<Complex> m2(2*sizeLarge, sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(double(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10,double(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10);
	}

	TEST_CALL( m1.transpose(), m2, BOOST_CURRENT_FUNCTION );
}

// small vector
void test_conjugate_complex_1()  
{
	Array<Complex> v1(sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = Complex(i,i);

	Array<Complex> v2(sizeSmall);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = Complex(i,-i);

	TEST_CALL( v1.conj(), v2, BOOST_CURRENT_FUNCTION );
}

// large vector
void test_conjugate_complex_2() 
{
	Array<Complex> v1(sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = Complex(double(i)/10,double(i)/10);

	Array<Complex> v2(sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = Complex(double(i)/10,double(-i)/10);

	TEST_CALL( v1.conj(), v2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_conjugate_complex_3() 
{	
	Array<Complex> m1(sizeSmall, sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10,double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10);
	}

	Array<Complex> m2(sizeSmall, sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(double(i*m2.GetDescriptor().GetDim( 1 ) + j + 1)/10,-double(i*m2.GetDescriptor().GetDim( 1 ) + j + 1)/10);
	}

	TEST_CALL( m1.conj(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_conjugate_complex_4()
{
	Array<Complex> m1(sizeLarge, sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10,double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10);
	}

	Array<Complex> m2(sizeLarge, sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(double(i*m2.GetDescriptor().GetDim( 1 ) + j + 1)/10,-double(i*m2.GetDescriptor().GetDim( 1 ) + j + 1)/10);
	}

	TEST_CALL( m1.conj(), m2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_conjugate_complex_5() 
{	
	Array<Complex> m1(sizeSmall, 2*sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10,double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10);
	}

	Array<Complex> m2(sizeSmall, 2*sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(double(i*m2.GetDescriptor().GetDim( 1 ) + j + 1)/10,-double(i*m2.GetDescriptor().GetDim( 1 ) + j + 1)/10);
	}

	TEST_CALL( m1.conj(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_conjugate_complex_6()
{
	Array<Complex> m1(sizeLarge, 2*sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10,double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10);
	}

	Array<Complex> m2(sizeLarge, 2*sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(double(i*m2.GetDescriptor().GetDim( 1 ) + j + 1)/10,-double(i*m2.GetDescriptor().GetDim( 1 ) + j + 1)/10);
	}

	TEST_CALL( m1.conj(), m2, BOOST_CURRENT_FUNCTION );
}

// small vector
void test_hermitian_complex_1()  
{
	Array<Complex> v1(sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = Complex(i,i);

	Array<Complex> v2(1,sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v2(0,i) = Complex(i,-i);

	TEST_CALL( v1.hermitian(), v2, BOOST_CURRENT_FUNCTION );
}

// large vector
void test_hermitian_complex_2() 
{
	Array<Complex> v1(sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = Complex(double(i)/10,double(i)/10);

	Array<Complex> v2(1,sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 1 ); i++)
		v2(0,i) = Complex(double(i)/10,-double(i)/10);

	TEST_CALL( v1.hermitian(), v2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_hermitian_complex_3() 
{	
	Array<Complex> m1(sizeSmall, sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10,double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10);
	}

	Array<Complex> m2(sizeSmall, sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(double(j*m2.GetDescriptor().GetDim( 1 ) + i + 1)/10,-double(j*m2.GetDescriptor().GetDim( 1 ) + i + 1)/10);
	}

	TEST_CALL( m1.hermitian(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_hermitian_complex_4()
{
	Array<Complex> m1(sizeLarge, sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10,double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10);
	}

	Array<Complex> m2(sizeLarge, sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(double(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10,-double(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10);
	}

	TEST_CALL( m1.hermitian(), m2, BOOST_CURRENT_FUNCTION );
}

// small matrix
void test_hermitian_complex_5() 
{	
	Array<Complex> m1(sizeSmall, 2*sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10,double(i*m1.GetDescriptor().GetDim( 0 ) + j + 1)/10);
	}

	Array<Complex> m2(2*sizeSmall, sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(double(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10,-double(j*m2.GetDescriptor().GetDim( 1 ) + i + 1)/10);
	}

	TEST_CALL( m1.hermitian(), m2, BOOST_CURRENT_FUNCTION );
}

// large matrix
void test_hermitian_complex_6()
{
	Array<Complex> m1(sizeLarge, 2*sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10,double(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10);
	}

	Array<Complex> m2(2*sizeLarge, sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(double(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10,-double(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10);
	}

	TEST_CALL( m1.hermitian(), m2, BOOST_CURRENT_FUNCTION );
}