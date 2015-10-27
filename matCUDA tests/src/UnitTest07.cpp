#include "unitTests.h"

#define sizeSmall 4
#define sizeLarge 1024


/////// float type
// v.transpose()*matrix = v.transpose
void test_times_equal_float_1() 
{
	Array<float> v1(1,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ )
		v1(0,i) = 0.1;

	Array<float> m1(sizeSmall,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ ) {
		for( int j = 0; j < sizeSmall; j++ )
			m1(i,j) = 0.1;
	}

	Array<float> v2(1,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ )
		v2(0,i) = float(sizeSmall)/100;

	v1 *= m1;

	TEST_CALL( v1, v2, BOOST_CURRENT_FUNCTION );
}

void test_times_equal_float_2()
{
	Array<float> v1(1,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ )
		v1(0,i) = 0.1;

	Array<float> m1(sizeLarge,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ ) {
		for( int j = 0; j < sizeLarge; j++ )
			m1(i,j) = 0.1;
	}

	Array<float> v2(1,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ )
		v2(0,i) = float(sizeLarge)/100;

	v1 *= m1;

	TEST_CALL( v1, v2, BOOST_CURRENT_FUNCTION );
}

// matrix*matrix = matrix
void test_times_equal_float_3()
{
	Array<float> m1(sizeSmall,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ ) {
		for( int j = 0; j < sizeSmall; j++ )
			m1(i,j) = 0.1;
	}

	Array<float> m2(sizeSmall,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ ) {
		for( int j = 0; j < sizeSmall; j++ )
			m2(i,j) = 0.1;
	}

	Array<float> m3(sizeSmall,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ ) {
		for( int j = 0; j < sizeSmall; j++ )
			m3(i,j) = float(sizeSmall)/100;
	}

	m1 *= m2;

	TEST_CALL( m1, m3, BOOST_CURRENT_FUNCTION );
}

void test_times_equal_float_4() 
{
	Array<float> m1(sizeLarge,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ ) {
		for( int j = 0; j < sizeLarge; j++ )
			m1(i,j) = 0.1;
	}

	Array<float> m2(sizeLarge,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ ) {
		for( int j = 0; j < sizeLarge; j++ )
			m2(i,j) = 0.1;
	}

	Array<float> m3(sizeLarge,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ ) {
		for( int j = 0; j < sizeLarge; j++ )
			m3(i,j) = float(sizeLarge)/100;
	}

	m1 *= m2;

	TEST_CALL( m1, m3, BOOST_CURRENT_FUNCTION );
}

//////// double type

// v.transpose()*matrix = v.transpose
void test_times_equal_double_1() 
{
	Array<double> v1(1,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ )
		v1(0,i) = 0.1;

	Array<double> m1(sizeSmall,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ ) {
		for( int j = 0; j < sizeSmall; j++ )
			m1(i,j) = 0.1;
	}

	Array<double> v2(1,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ )
		v2(0,i) = double(sizeSmall)/100;

	v1 *= m1;

	TEST_CALL( v1, v2, BOOST_CURRENT_FUNCTION );
}

void test_times_equal_double_2()
{
	Array<double> v1(1,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ )
		v1(0,i) = 0.1;

	Array<double> m1(sizeLarge,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ ) {
		for( int j = 0; j < sizeLarge; j++ )
			m1(i,j) = 0.1;
	}

	Array<double> v2(1,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ )
		v2(0,i) = double(sizeLarge)/100;

	v1 *= m1;

	TEST_CALL( v1, v2, BOOST_CURRENT_FUNCTION );	
}

// matrix*matrix = matrix
void test_times_equal_double_3()
{
	Array<double> m1(sizeSmall,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ ) {
		for( int j = 0; j < sizeSmall; j++ )
			m1(i,j) = 0.1;
	}

	Array<double> m2(sizeSmall,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ ) {
		for( int j = 0; j < sizeSmall; j++ )
			m2(i,j) = 0.1;
	}

	Array<double> m3(sizeSmall,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ ) {
		for( int j = 0; j < sizeSmall; j++ )
			m3(i,j) = double(sizeSmall)/100;
	}

	m1 *= m2;

	TEST_CALL( m1, m3, BOOST_CURRENT_FUNCTION );
}

void test_times_equal_double_4() 
{
	Array<double> m1(sizeLarge,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ ) {
		for( int j = 0; j < sizeLarge; j++ )
			m1(i,j) = 0.1;
	}

	Array<double> m2(sizeLarge,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ ) {
		for( int j = 0; j < sizeLarge; j++ )
			m2(i,j) = 0.1;
	}

	Array<double> m3(sizeLarge,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ ) {
		for( int j = 0; j < sizeLarge; j++ )
			m3(i,j) = double(sizeLarge)/100;
	}

	m1 *= m2;

	TEST_CALL( m1, m3, BOOST_CURRENT_FUNCTION );
}



//////// complex type
// v.transpose()*matrix = v.transpose
void test_times_equal_complex_1()  
{
	Array<Complex> v1(1,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ )
		v1(0,i) = Complex(0.1,0.1);

	Array<Complex> m1(sizeSmall,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ ) {
		for( int j = 0; j < sizeSmall; j++ )
			m1(i,j) = Complex(0.1,0.1);
	}

	Array<Complex> v2(1,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ )
		v2(0,i) = Complex(0,float(sizeSmall)*2/100);

	v1 *= m1;

	TEST_CALL( v1, v2, BOOST_CURRENT_FUNCTION );
}

void test_times_equal_complex_2()
{
	Array<Complex> v1(1,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ )
		v1(0,i) = Complex(0.1,0.1);

	Array<Complex> m1(sizeLarge,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ ) {
		for( int j = 0; j < sizeLarge; j++ )
			m1(i,j) = Complex(0.1,0.1);
	}

	Array<Complex> v2(1,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ )
		v2(0,i) = Complex(0,float(sizeLarge)*2/100);

	v1 *= m1;

	TEST_CALL( v1, v2, BOOST_CURRENT_FUNCTION );
}

// matrix*matrix = matrix
void test_times_equal_complex_3()
{
	Array<Complex> m1(sizeSmall,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ ) {
		for( int j = 0; j < sizeSmall; j++ )
			m1(i,j) = Complex(0.1,0.1);
	}

	Array<Complex> m2(sizeSmall,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ ) {
		for( int j = 0; j < sizeSmall; j++ )
			m2(i,j) = Complex(0.1,0.1);
	}

	Array<Complex> m3(sizeSmall,sizeSmall);
	for( int i = 0; i < sizeSmall; i++ ) {
		for( int j = 0; j < sizeSmall; j++ )
			m3(i,j) = Complex(0,float(sizeSmall)*2/100);
	}

	m1 *= m2;

	TEST_CALL( m1, m3, BOOST_CURRENT_FUNCTION );
}

void test_times_equal_complex_4() 
{
	Array<Complex> m1(sizeLarge,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ ) {
		for( int j = 0; j < sizeLarge; j++ )
			m1(i,j) = Complex(0.1,0.1);
	}

	Array<Complex> m2(sizeLarge,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ ) {
		for( int j = 0; j < sizeLarge; j++ )
			m2(i,j) = Complex(0.1,0.1);
	}

	Array<Complex> m3(sizeLarge,sizeLarge);
	for( int i = 0; i < sizeLarge; i++ ) {
		for( int j = 0; j < sizeLarge; j++ )
			m3(i,j) = Complex(0,float(sizeLarge)*2/100);
	}
	
	m1 *= m2;

	TEST_CALL( m1, m3, BOOST_CURRENT_FUNCTION );
}
