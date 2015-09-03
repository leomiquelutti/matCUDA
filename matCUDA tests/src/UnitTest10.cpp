#include "unitTests.h"
#include "boost/date_time/posix_time/posix_time.hpp"
//#include <thrust\complex.h>

#define sizeSmall 4
#define sizeLarge 1024

//// minor

// int type
void test_minor_int_1()
{
	Array<int> m( 2, 2 );
	m( 0, 0 ) = 0;
	m( 0, 1 ) = 1;
	m( 1, 0 ) = 2;
	m( 1, 1 ) = 3;
	m.print();

	Array<int> control( 1, 1 );

	control = m.minor( 1, 0 );
	control.print();
}

void test_minor_int_2()
{
	Array<int> m( 3, 5 );
	for( int i = 0; i < m.GetDescriptor().GetDim(0); i++ )
	{
		for( int j = 0; j < m.GetDescriptor().GetDim(1); j++ )
			m( i, j ) = j + i*m.GetDescriptor().GetDim(1);
	}
	m.print();

	Array<int> control( 2, 4 );

	control = m.minor( 1, 0 );
	control.print();
}

// float type
void test_minor_float_1()
{
	Array<float> m( 2, 2 );
	m( 0, 0 ) = 0;
	m( 0, 1 ) = 0.1f;
	m( 1, 0 ) = 0.2f;
	m( 1, 1 ) = 0.3f;

	Array<float> control( 1, 1 );

	control( 0, 0 ) = m( 0, 1 );

	TEST_CALL( m.minor(1,0), control, BOOST_CURRENT_FUNCTION );
}

void test_minor_float_2()
{
	Array<float> m( 3, 5 );
	for( int i = 0; i < m.GetDescriptor().GetDim(0); i++ )
	{
		for( int j = 0; j < m.GetDescriptor().GetDim(1); j++ )
			m( i, j ) = (float)(j + i*m.GetDescriptor().GetDim(1))/10;
	}

	Array<float> control( 2, 4 );
	for( int i = 0; i < control.GetDescriptor().GetDim(0); i++ )
	{
		for( int j = 0; j < control.GetDescriptor().GetDim(1); j++ )
			control( i, j ) = (float)(j + 1 + 2.5*i*control.GetDescriptor().GetDim(1))/10;
	}

	TEST_CALL( m.minor(1,0), control, BOOST_CURRENT_FUNCTION );
}

// double type
void test_minor_double_1()
{
	Array<double> m( 2, 2 );
	m( 0, 0 ) = 0;
	m( 0, 1 ) = 0.1;
	m( 1, 0 ) = 0.2;
	m( 1, 1 ) = 0.3;

	Array<double> control( 1, 1 );

	control( 0, 0 ) = m( 0, 1 );

	TEST_CALL( m.minor(1,0), control, BOOST_CURRENT_FUNCTION );
}

void test_minor_double_2()
{
	Array<double> m( 3, 5 );
	for( int i = 0; i < m.GetDescriptor().GetDim(0); i++ )
	{
		for( int j = 0; j < m.GetDescriptor().GetDim(1); j++ )
			m( i, j ) = (double)(j + i*m.GetDescriptor().GetDim(1))/10;
	}

	Array<double> control( 2, 4 );
	for( int i = 0; i < control.GetDescriptor().GetDim(0); i++ )
	{
		for( int j = 0; j < control.GetDescriptor().GetDim(1); j++ )
			control( i, j ) = (double)(j + 1 + 2.5*i*control.GetDescriptor().GetDim(1))/10;
	}

	TEST_CALL( m.minor(1,0), control, BOOST_CURRENT_FUNCTION );
}

// complex type
void test_minor_complex_1()
{
	Array<Complex> m( 2, 2 );
	m( 0, 0 ) = Complex(0,0);
	m( 0, 1 ) = Complex(0,1);
	m( 1, 0 ) = Complex(1,0);
	m( 1, 1 ) = Complex(1,1);

	Array<Complex> control( 1, 1 );

	control( 0, 0 ) = m( 0, 1 );

	TEST_CALL( m.minor(1,0), control, BOOST_CURRENT_FUNCTION );
}

void test_minor_complex_2()
{
	Array<Complex> m( 3, 5 );
	for( int i = 0; i < m.GetDescriptor().GetDim(0); i++ )
	{
		for( int j = 0; j < m.GetDescriptor().GetDim(1); j++ )
			m( i, j ) = Complex((double)(j + i*m.GetDescriptor().GetDim(1))/10,(double)(j + i*m.GetDescriptor().GetDim(1))/10);
	}

	Array<Complex> control( 2, 4 );
	for( int i = 0; i < control.GetDescriptor().GetDim(0); i++ )
	{
		for( int j = 0; j < control.GetDescriptor().GetDim(1); j++ )
			control( i, j ) =  Complex((double)(j + 1 + 2.5*i*control.GetDescriptor().GetDim(1))/10,(double)(j + 1 + 2.5*i*control.GetDescriptor().GetDim(1))/10);
	}

	TEST_CALL( m.minor(1,0), control, BOOST_CURRENT_FUNCTION );
}

//// norm

// int type
void test_norm_int_1()
{
	Array<int> m( sizeSmall );
	m( 0 ) = 0;
	m( 1 ) = 3;
	m( 2 ) = 4;
	m( 3 ) = 12;
	int control = 13;

	TEST_CALL( m.norm(), control, BOOST_CURRENT_FUNCTION );
}

// float type
void test_norm_float_1()
{
	Array<float> m( sizeSmall );
	m( 0 ) = 0;
	m( 1 ) = 3.0;
	m( 2 ) = 4.0;
	m( 3 ) = 12.0;

	float control = 13.0f;

	TEST_CALL( m.norm(), control, BOOST_CURRENT_FUNCTION );
}

// double type
void test_norm_double_1()
{
	Array<double> m( sizeSmall );
	m( 0 ) = 0;
	m( 1 ) = 3.0;
	m( 2 ) = 4.0;
	m( 3 ) = 12.0;
	
	double control = 13.0;

	TEST_CALL( m.norm(), control, BOOST_CURRENT_FUNCTION );
}

// complex type
void test_norm_complex_1()
{
	Array<Complex> m( sizeSmall );
	m( 0 ) = 0;
	m( 1 ) = Complex( 3.0, 4.0);
	m( 2 ) = Complex( 6.0, 8.0);
	m( 3 ) = Complex( 6.0, 8.0);

	Complex control( 15.0, 0 );

	TEST_CALL( m.norm(), control, BOOST_CURRENT_FUNCTION );
}

//// eye

// int type
void test_eye_int_1()
{
	Array<int> m = eye<int>( sizeSmall );
	Array<int> control( sizeSmall, sizeSmall );
	for( int i = 0; i < sizeSmall; i++ ) {
		for( int j = i; j < i + 1; j++ )
			control(i,j) = i == j;
	}

	m.print();
	
	if( m != control )
		BOOST_FAIL("m != control!!!");
}

void test_eye_int_2()
{
	Array<int> m = eye<int>( sizeLarge );
	Array<int> control( sizeLarge, sizeLarge );
	for( int i = 0; i < sizeLarge; i++ ) {
		for( int j = i; j < i + 1; j++ )
			control(i,j) = i == j;
	}
	
	if( m != control )
		BOOST_FAIL("m != control!!!");
}

// float type
void test_eye_float_1()
{
	Array<float> m = eye<float>( sizeSmall );
	Array<float> control( sizeSmall, sizeSmall );
	for( int i = 0; i < sizeSmall; i++ ) {
		for( int j = i; j < i + 1; j++ )
			control(i,j) = i == j;
	}

	TEST_CALL( m, control, BOOST_CURRENT_FUNCTION );
}

void test_eye_float_2()
{
	Array<float> m = eye<float>( sizeLarge );
	Array<float> control( sizeLarge, sizeLarge );
	for( int i = 0; i < sizeLarge; i++ ) {
		for( int j = i; j < i + 1; j++ )
			control(i,j) = i == j;
	}

	TEST_CALL( m, control, BOOST_CURRENT_FUNCTION );
}

// double type
void test_eye_double_1()
{
	Array<double> m = eye<double>( sizeSmall );
	Array<double> control( sizeSmall, sizeSmall );
	for( int i = 0; i < sizeSmall; i++ ) {
		for( int j = i; j < i + 1; j++ )
			control(i,j) = i == j;
	}

	TEST_CALL( m, control, BOOST_CURRENT_FUNCTION );
}

void test_eye_double_2()
{
	Array<double> m = eye<double>( sizeLarge );
	Array<double> control( sizeLarge, sizeLarge );
	for( int i = 0; i < sizeLarge; i++ ) {
		for( int j = i; j < i + 1; j++ )
			control(i,j) = i == j;
	}

	TEST_CALL( m, control, BOOST_CURRENT_FUNCTION );
}

// complex type
void test_eye_complex_1()
{
	Array<Complex> m = eye<Complex>( sizeSmall );
	Array<Complex> control( sizeSmall, sizeSmall );
	for( int i = 0; i < sizeSmall; i++ ) {
		for( int j = i; j < i + 1; j++ )
			control(i,j) = Complex(i == j,0);
	}

	TEST_CALL( m, control, BOOST_CURRENT_FUNCTION );
}

void test_eye_complex_2()
{
	Array<Complex> m = eye<Complex>( sizeLarge );
	Array<Complex> control( sizeLarge, sizeLarge );
	for( int i = 0; i < sizeLarge; i++ ) {
		for( int j = i; j < i + 1; j++ )
			control(i,j) = Complex(i == j,0);
	}

	TEST_CALL( m, control, BOOST_CURRENT_FUNCTION );
}

void test_times_TElement_int() {}

void test_times_TElement_float()
{
	float a = 1.5;

	Array<float> v( 10,30 );
	for( int i = 0; i < v.GetDescriptor().GetDim(0); i++ ) {
		for( int j = 0; j < v.GetDescriptor().GetDim(1); j++ )
			v( i, j ) = 1;
	}	

	Array<float> control( 10,30 );
	for( int i = 0; i < control.GetDescriptor().GetDim(0); i++ ) {
		for( int j = 0; j < control.GetDescriptor().GetDim(1); j++ )
			control( i, j ) = 1*a;
	}

	v = v*a;

	TEST_CALL( v, control, BOOST_CURRENT_FUNCTION );
}

void test_times_TElement_double()
{
	double a = 1.5;
	Array<double> v( 10,30 );

	for( int i = 0; i < v.GetDescriptor().GetDim(0); i++ ) {
		for( int j = 0; j < v.GetDescriptor().GetDim(1); j++ )
			v( i, j ) = 1;
	}

	Array<double> control( 10,30 );
	for( int i = 0; i < control.GetDescriptor().GetDim(0); i++ ) {
		for( int j = 0; j < control.GetDescriptor().GetDim(1); j++ )
			control( i, j ) = 1*a;
	}

	v = v*a;

	TEST_CALL( v, control, BOOST_CURRENT_FUNCTION );
}
void test_times_TElement_complex()
{
	Complex a(1.5,2);
	Array<Complex> v( 10,30 );

	for( int i = 0; i < v.GetDescriptor().GetDim(0); i++ ) {
		for( int j = 0; j < v.GetDescriptor().GetDim(1); j++ )
			v( i, j ) = 1;
	}

	Array<Complex> control( 10,30 );
	for( int i = 0; i < control.GetDescriptor().GetDim(0); i++ ) {
		for( int j = 0; j < control.GetDescriptor().GetDim(1); j++ )
			control( i, j ) = a;
	}

	v = v*a;

	TEST_CALL( v, control, BOOST_CURRENT_FUNCTION );
}
