#include "unitTests.h"

#define sizeSmall 4
#define sizeLarge 1024

void test_minus_float_1()
{
	// small vector
	Array<float> v1( sizeSmall );
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = 1;

	Array<float> v2( sizeSmall );
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = 1;
	
	Array<float> v3( sizeSmall );
	for (int i = 0; i < v3.GetDescriptor().GetDim( 0 ); i++)
		v3(i) = 5;
	
	Array<float> v4( sizeSmall );

	v3 = v1 - v2;
	
	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

void test_minus_float_2()
{
	// large vector
	Array<float> v1( sizeLarge );
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = 1;

	Array<float> v2( sizeLarge );
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = 1;
	
	Array<float> v3( sizeLarge );
	for (int i = 0; i < v3.GetDescriptor().GetDim( 0 ); i++)
		v3(i) = 5;
	
	Array<float> v4( sizeLarge );

	v3 = v1 - v2;
	
	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

void test_minus_float_3() 
{	
	Array<float> m1( sizeSmall, sizeSmall );
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 1.0;
	}

	Array<float> m2( sizeSmall, sizeSmall );
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = 1.0;
	}
	
	Array<float> m3( sizeSmall, sizeSmall );
	for (int i = 0; i < m3.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m3.GetDescriptor().GetDim( 1 ); j++)
			m3(i,j) = 5.0;
	}
	
	Array<float> m4( sizeSmall, sizeSmall );

	m3 = m1 - m2;
	
	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

void test_minus_float_4()
{
	// matrix
	Array<float> m1( sizeLarge, sizeLarge );
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 1.0;
	}

	Array<float> m2( sizeLarge, sizeLarge );
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = 1.0;
	}
	
	Array<float> m3( sizeLarge, sizeLarge );
	for (int i = 0; i < m3.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m3.GetDescriptor().GetDim( 1 ); j++)
			m3(i,j) = 5.0;
	}
	
	Array<float> m4( sizeLarge, sizeLarge );

	m3 = m1 - m2;
	
	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

void test_minus_double_1()
{
	// small vector
	Array<double> v1( sizeSmall );
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = 1;

	Array<double> v2( sizeSmall );
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = 1;
	
	Array<double> v3( sizeSmall );
	for (int i = 0; i < v3.GetDescriptor().GetDim( 0 ); i++)
		v3(i) = 5;
	
	Array<double> v4( sizeSmall );

	v3 = v1 - v2;
	
	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

void test_minus_double_2()
{
	// large vector
	Array<double> v1( sizeLarge );
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = 1;

	Array<double> v2( sizeLarge );
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = 1;
	
	Array<double> v3( sizeLarge );
	for (int i = 0; i < v3.GetDescriptor().GetDim( 0 ); i++)
		v3(i) = 5;
	
	Array<double> v4( sizeLarge );

	v3 = v1 - v2;
	
	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

void test_minus_double_3() 
{	
	Array<double> m1( sizeSmall, sizeSmall );
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 1.0;
	}

	Array<double> m2( sizeSmall, sizeSmall );
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = 1.0;
	}
	
	Array<double> m3( sizeSmall, sizeSmall );
	for (int i = 0; i < m3.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m3.GetDescriptor().GetDim( 1 ); j++)
			m3(i,j) = 5.0;
	}
	
	Array<double> m4( sizeSmall, sizeSmall );

	m3 = m1 - m2;
	
	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

void test_minus_double_4()
{
	// matrix
	Array<double> m1( sizeLarge, sizeLarge );
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 1.0;
	}

	Array<double> m2( sizeLarge, sizeLarge );
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = 1.0;
	}
	
	Array<double> m3( sizeLarge, sizeLarge );
	for (int i = 0; i < m3.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m3.GetDescriptor().GetDim( 1 ); j++)
			m3(i,j) = 5.0;
	}
	
	Array<double> m4( sizeLarge, sizeLarge );

	m3 = m1 - m2;
	
	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

void test_minus_complex_1()
{
	// small vector
	Array<Complex> v1( sizeSmall );
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = Complex(1,1);

	Array<Complex> v2( sizeSmall );
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = Complex(1,1);
	
	Array<Complex> v3( sizeSmall );
	for (int i = 0; i < v3.GetDescriptor().GetDim( 0 ); i++)
		v3(i) = Complex(5,5);
	
	Array<Complex> v4( sizeSmall );

	v3 = v1 - v2;
	
	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

void test_minus_complex_2()
{
	// large vector
	Array<Complex> v1( sizeLarge );
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = Complex(1,1);

	Array<Complex> v2( sizeLarge );
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = Complex(1,1);
	
	Array<Complex> v3( sizeLarge );
	for (int i = 0; i < v3.GetDescriptor().GetDim( 0 ); i++)
		v3(i) = Complex(5,5);
	
	Array<Complex> v4( sizeLarge );

	v3 = v1 - v2;
	
	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

void test_minus_complex_3()
{	
	Array<Complex> m1( sizeSmall, sizeSmall );
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(1,1);
	}

	Array<Complex> m2( sizeSmall, sizeSmall );
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(1,1);
	}
	
	Array<Complex> m3( sizeSmall, sizeSmall );
	for (int i = 0; i < m3.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m3.GetDescriptor().GetDim( 1 ); j++)
			m3(i,j) = Complex(5,5);
	}
	
	Array<Complex> m4( sizeSmall, sizeSmall );

	m3 = m1 - m2;
	
	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

void test_minus_complex_4()
{
	// matrix
	Array<Complex> m1( sizeLarge, sizeLarge );
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(1,1);
	}

	Array<Complex> m2( sizeLarge, sizeLarge );
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(1,1);
	}
	
	Array<Complex> m3( sizeLarge, sizeLarge );
	for (int i = 0; i < m3.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m3.GetDescriptor().GetDim( 1 ); j++)
			m3(i,j) = Complex(5,5);
	}
	
	Array<Complex> m4( sizeLarge, sizeLarge );

	m3 = m1 - m2;
	
	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}
