#include "unitTests.h"

#define sizeSmall 4
#define sizeLarge 4096

//////// float type
// v.tranpose()*v = scalar

void test_times_float_1() 
{
	Array<float> v1(1,sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 1 ); i++)
		v1(0,i) = 0.1f;

	Array<float> v2(sizeSmall,1);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i,0) = 0.1f;
	
	int scalar1 = 0, scalar2 = 0;
	Array<float> scalar3(1);
	Array<float> scalar4(1);
	scalar4(0) = float(sizeSmall)/100;

	scalar3 = v1*v2;

	TEST_CALL( scalar3, scalar4, BOOST_CURRENT_FUNCTION );
}

void test_times_float_2() 
{
	Array<float> v1(1,sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 1 ); i++)
		v1(0,i) = 0.1f;

	Array<float> v2(sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = 0.1f;
	
	int scalar1 = 0, scalar2 = 0;
	Array<float> scalar3(1);
	Array<float> scalar4(1);
	scalar4(0) = float(sizeLarge)/100;

	scalar3 = v1*v2;

	TEST_CALL( scalar3, scalar4, BOOST_CURRENT_FUNCTION );
}

// v*v.tranpose() = matrix
void test_times_float_3() 
{
	Array<float> v1(sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = 0.1f;

	Array<float> v2(1,sizeSmall);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 1 ); i++)
		v2(0,i) = 0.1f;
	
	Array<float> m3(sizeSmall,sizeSmall);
	Array<float> m4(sizeSmall,sizeSmall);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = 0.01f;
	}

	m3 = v1*v2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

void test_times_float_4() 
{
	Array<float> v1(sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = 0.1f;

	Array<float> v2(1,sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 1 ); i++)
		v2(0,i) = 0.1f;
	
	Array<float> m3(sizeLarge,sizeLarge);
	Array<float> m4(sizeLarge,sizeLarge);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = 0.01f;
	}

	m3 = v1*v2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

// matrix*v = v
void test_times_float_5() 
{
	Array<float> m1(2*sizeSmall,sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 0.1f;
	}

	Array<float> v2(sizeSmall);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = 0.1f;

	Array<float> v3(2*sizeSmall);
	Array<float> v4(2*sizeSmall);
	for (int i = 0; i < v4.GetDescriptor().GetDim( 0 ); i++)
		v4(i) = float(sizeSmall)/100;

	v3 = m1*v2;

	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

void test_times_float_6()
{
	Array<float> m1(sizeLarge,sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 0.1f;
	}

	Array<float> v2(sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = 0.1f;

	Array<float> v3(sizeLarge);
	Array<float> v4(sizeLarge);
	for (int i = 0; i < v4.GetDescriptor().GetDim( 0 ); i++)
		v4(i) = float(sizeLarge)/100;

	v3 = m1*v2;

	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

// v.transpose()*matrix = v.transpose
void test_times_float_7()
{	
	Array<float> v1(1,sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 1 ); i++)
		v1(0,i) = 0.1f;

	//v1.transpose();

	Array<float> m2(sizeSmall,sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = 0.1f;
	}

	Array<float> v3(1,sizeSmall);
	Array<float> v4(1,sizeSmall);
	for (int i = 0; i < v4.GetDescriptor().GetDim( 1 ); i++)
		v4(0,i) = float(sizeSmall)/100;

	v3 = v1*m2;

	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

void test_times_float_8()
{
	Array<float> v1(1,sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 1 ); i++)
		v1(0,i) = 0.1f;

	Array<float> m2(sizeLarge,sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = 0.1f;
	}

	Array<float> v3(1,sizeLarge);
	Array<float> v4(1,sizeLarge);
	for (int i = 0; i < v4.GetDescriptor().GetDim( 1 ); i++)
		v4(0,i) = float(sizeLarge)/100;

	v3 = v1*m2;

	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

// matrix*matrix = matrix
void test_times_float_9()
{
	Array<float> m1(sizeSmall,sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 0.1f;
	}

	Array<float> m2(sizeSmall,sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = 0.1f;
	}

	Array<float> m3(sizeSmall,sizeSmall);
	Array<float> m4(sizeSmall,sizeSmall);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = float(sizeSmall)/100;
	}

	m3 = m1*m2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

void test_times_float_10()
{	
	Array<float> m1(sizeLarge,sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 0.1f;
	}

	Array<float> m2(sizeLarge,sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = 0.1f;
	}

	Array<float> m3(sizeLarge,sizeLarge);
	Array<float> m4(sizeLarge,sizeLarge);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = float(sizeLarge)/100;
	}

	m3 = m1*m2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

// v.tranpose()*v = scalar
void test_times_float_11() 
{
	Array<float> v1(1,sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 1 ); i++)
		v1(0,i) = float(i)/10;

	Array<float> v2(sizeSmall);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = float(i)/10;
	
	int scalar1 = 0, scalar2 = 0;
	Array<float> scalar3(1);
	Array<float> scalar4(1);
	scalar4(0) = 0.14f;

	scalar3 = v1*v2;

	TEST_CALL( scalar3, scalar4, BOOST_CURRENT_FUNCTION );
}

// matrix*matrix = matrix
void test_times_float_12()
{
	Array<float> m1(sizeSmall,sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = float(i*sizeSmall + j + 1)/10;
	}

	Array<float> m2(sizeSmall,sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = i == j;
	}

	Array<float> m3(sizeSmall,sizeSmall);
	Array<float> m4 = m1;

	m3 = m1*m2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

// matrix n x m
void test_times_float_13()
{
	Array<float> m1(sizeSmall,2*sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 0.15f;
	}

	Array<float> m2(2*sizeSmall,sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = i == j;
	}

	Array<float> m4(sizeSmall,sizeSmall);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = 0.15f;
	}
	
	Array<float> m3 = m1*m2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

void test_times_float_14()
{
	const size_t size = 500;
	Array<float> m1(size,2*size);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 0.15f;
	}

	Array<float> m2(2*size,size);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = i == j;
	}

	Array<float> m4(size,size);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = 0.15f;
	}
	
	Array<float> m3 = m1*m2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

// matrix m x n
void test_times_float_15()
{
	Array<float> m1(2*sizeSmall,sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 0.15f;
	}

	Array<float> m2(sizeSmall,2*sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = i == j;
	}

	Array<float> m4(2*sizeSmall,2*sizeSmall);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = 0.15*(j < sizeSmall);
	}
	
	Array<float> m3 = m1*m2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

void test_times_float_16()
{
	const size_t size = 2000;
	Array<float> m1(size,2*size);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 0.15f;
	}

	Array<float> m2(2*size,size);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = i == j;
	}

	Array<float> m4(size,size);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = 0.15f;
	}
	
	// tic();
	Array<float> m3 = m1*m2;
	// toc();

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}


//////// double type
// v.tranpose()*v = scalar
void test_times_double_1() 
{
	Array<double> v1(1,sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 1 ); i++)
		v1(0,i) = 0.1;

	Array<double> v2(sizeSmall);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = 0.1;
	
	int scalar1 = 0, scalar2 = 0;
	Array<double> scalar3(1);
	Array<double> scalar4(1);
	scalar4(0) = double(sizeSmall)/100;

	scalar3 = v1*v2;

	TEST_CALL( scalar3, scalar4, BOOST_CURRENT_FUNCTION );
}

void test_times_double_2() 
{
	Array<double> v1(1,sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 1 ); i++)
		v1(0,i) = 0.1;

	Array<double> v2(sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = 0.1;
	
	int scalar1 = 0, scalar2 = 0;
	Array<double> scalar3(1);
	Array<double> scalar4(1);
	scalar4(0) = double(sizeLarge)/100;
	
	scalar3 = v1*v2;

	TEST_CALL( scalar3, scalar4, BOOST_CURRENT_FUNCTION );
}

// v*v.tranpose() = matrix
void test_times_double_3() 
{
	Array<double> v1(sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = 0.1;

	Array<double> v2(1,sizeSmall);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 1 ); i++)
		v2(0,i) = 0.1;
	
	Array<double> m3(sizeSmall,sizeSmall);
	Array<double> m4(sizeSmall,sizeSmall);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = 0.01;
	}

	m3 = v1*v2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

void test_times_double_4() 
{
	Array<double> v1(sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = 0.1;

	Array<double> v2(1,sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 1 ); i++)
		v2(0,i) = 0.1;
	
	Array<double> m3(sizeLarge,sizeLarge);
	Array<double> m4(sizeLarge,sizeLarge);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = 0.01;
	}

	m3 = v1*v2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

// matrix*v = v
void test_times_double_5() 
{
	Array<double> m1(sizeSmall,sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 0.1;
	}

	Array<double> v2(sizeSmall);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = 0.1;

	Array<double> v3(sizeSmall);
	Array<double> v4(sizeSmall);
	for (int i = 0; i < v4.GetDescriptor().GetDim( 0 ); i++)
		v4(i) = double(sizeSmall)/100;
	
	v3 = m1*v2;

	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

void test_times_double_6()
{
	Array<double> m1(sizeLarge,sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 0.1;
	}

	Array<double> v2(sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = 0.1;

	Array<double> v3(sizeLarge);
	Array<double> v4(sizeLarge);
	for (int i = 0; i < v4.GetDescriptor().GetDim( 0 ); i++)
		v4(i) = double(sizeLarge)/100;

	v3 = m1*v2;

	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

// v.transpose()*matrix = v.transpose
void test_times_double_7()
{
	Array<double> v1(1,sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 1 ); i++)
		v1(0,i) = 0.1;

	Array<double> m2(sizeSmall,sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = 0.1;
	}

	Array<double> v3(1,sizeSmall);
	Array<double> v4(1,sizeSmall);
	for (int i = 0; i < v4.GetDescriptor().GetDim( 1 ); i++)
		v4(0,i) = double(sizeSmall)/100;

	v3 = v1*m2;

	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

void test_times_double_8()
{
	Array<double> v1(1,sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 1 ); i++)
		v1(0,i) = 0.1;

	Array<double> m2(sizeLarge,sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = 0.1;
	}

	Array<double> v3(1,sizeLarge);
	Array<double> v4(1,sizeLarge);
	for (int i = 0; i < v4.GetDescriptor().GetDim( 1 ); i++)
		v4(0,i) = double(sizeLarge)/100;

	v3 = v1*m2;

	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

// matrix*matrix = matrix
void test_times_double_9()
{
	Array<double> m1(sizeSmall,sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 0.1;
	}

	Array<double> m2(sizeSmall,sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = 0.1;
	}

	Array<double> m3(sizeSmall,sizeSmall);
	Array<double> m4(sizeSmall,sizeSmall);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = double(sizeSmall)/100;
	}

	m3 = m1*m2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

void test_times_double_10()
{	
	Array<double> m1(sizeLarge,sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 0.1;
	}

	Array<double> m2(sizeLarge,sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = 0.1;
	}

	Array<double> m3(sizeLarge,sizeLarge);
	Array<double> m4(sizeLarge,sizeLarge);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = double(sizeLarge)/100;
	}

	m3 = m1*m2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

void test_times_double_11()
{	
	Array<double> m1(sizeLarge,sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 0.15;
	}

	Array<double> m2(sizeLarge,sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = i == j;
	}

	Array<double> m3(sizeLarge,sizeLarge);
	Array<double> m4(sizeLarge,sizeLarge);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = 0.15;
	}

	m3 = m1*m2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

// matrix n x m
void test_times_double_12()
{
	Array<double> m1(sizeSmall,2*sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 0.15;
	}

	Array<double> m2(2*sizeSmall,sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = i == j;
	}

	Array<double> m4(sizeSmall,sizeSmall);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = 0.15;
	}
	
	Array<double> m3 = m1*m2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

void test_times_double_13()
{
	const size_t size = 2000;
	Array<double> m1(size,2*size);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 0.15;
	}

	Array<double> m2(2*size,size);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = i == j;
	}

	Array<double> m4(size,size);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = 0.15;
	}
	
	Array<double> m3 = m1*m2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

// dot product
void test_times_double_14()
{	
	Array<double> m1(1,sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 1;
	}

	Array<double> m2(sizeSmall,1);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = 1;
	}

	Array<double> m3 = m1*m2;
	Array<double> m4(1); 
	m4(0) = sizeSmall;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

//////// complex type

// v.tranpose()*v = scalar
void test_times_complex_1() 
{
	Array<Complex> v1(1,sizeSmall);
	Array<Complex> v2(sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 1 ); i++) {
		v1(0,i) = Complex(i+3,i*2-2);
		v2(i) = Complex((float)i/1.4,i*5-5);
	}

	int scalar1 = 0, scalar2 = 0;
	Array<Complex> scalar3(1);
	Array<Complex> scalar4(1);
	scalar4(0) = Complex(-37.1429,81.4286);

	scalar3 = v1*v2;
	TEST_CALL( scalar3, scalar4, BOOST_CURRENT_FUNCTION );
}

void test_times_complex_2() 
{
	Array<Complex> v1(1,sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 1 ); i++)
		v1(0,i) = Complex(-i*0.1,0.1);

	Array<Complex> v2(sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = Complex(0.01,i/2);
	
	int scalar1 = 0, scalar2 = 0;
	Array<Complex> scalar3(1);
	Array<Complex> scalar4(1);
	scalar4(0) = Complex(-4.277145600000000e+05,-1.144905211904000e+09);
	
	scalar3 = v1*v2;
	TEST_CALL( scalar3, scalar4, BOOST_CURRENT_FUNCTION );
}

// v*v.tranpose() = matrix
void test_times_complex_3() 
{
	Array<Complex> v1(sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = Complex(0.1,0.1);

	Array<Complex> v2(1,sizeSmall);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 1 ); i++)
		v2(0,i) = Complex(0.1,0.1);
	
	Array<Complex> m3(sizeSmall,sizeSmall);
	Array<Complex> m4(sizeSmall,sizeSmall);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = Complex(0,double(sizeSmall)/2/100);
	}

	m3 = v1*v2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

void test_times_complex_4() 
{
	Array<Complex> v1(sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 0 ); i++)
		v1(i) = Complex(0.1,0.1);

	Array<Complex> v2(1,sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 1 ); i++)
		v2(0,i) = Complex(0.1,0.1);
	
	Array<Complex> m3(sizeLarge,sizeLarge);
	Array<Complex> m4(sizeLarge,sizeLarge);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = Complex(0,double(0.02));
	}

	m3 = v1*v2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

// matrix*v = v
void test_times_complex_5() 
{
	Array<Complex> m1(sizeSmall,sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(0.1,0.1);
	}

	Array<Complex> v2(sizeSmall);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = Complex(0.1,0.1);

	Array<Complex> v3(sizeSmall);
	Array<Complex> v4(sizeSmall);
	for (int i = 0; i < v4.GetDescriptor().GetDim( 0 ); i++)
		v4(i) = Complex(0,double(sizeSmall)*2/100);
	
	v3 = m1*v2;

	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

void test_times_complex_6()
{
	Array<Complex> m1(sizeLarge,sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(0.1,0.1);
	}

	Array<Complex> v2(sizeLarge);
	for (int i = 0; i < v2.GetDescriptor().GetDim( 0 ); i++)
		v2(i) = Complex(0.1,0.1);

	Array<Complex> v3(sizeLarge);
	Array<Complex> v4(sizeLarge);
	for (int i = 0; i < v4.GetDescriptor().GetDim( 0 ); i++)
		v4(i) = Complex(0,double(sizeLarge)*2/100);

	v3 = m1*v2;

	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

// v.transpose()*matrix = v.transpose
void test_times_complex_7()
{
	Array<Complex> v1(1,sizeSmall);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 1 ); i++)
		v1(0,i) = Complex(0.1,0.1);

	Array<Complex> m2(sizeSmall,sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(0.1,0.1);
	}

	Array<Complex> v3(1,sizeSmall);
	Array<Complex> v4(1,sizeSmall);
	for (int i = 0; i < v4.GetDescriptor().GetDim( 1 ); i++)
		v4(0,i) = Complex(0,double(sizeSmall)*2/100);

	v3 = v1*m2;

	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

void test_times_complex_8()
{
	Array<Complex> v1(1,sizeLarge);
	for (int i = 0; i < v1.GetDescriptor().GetDim( 1 ); i++)
		v1(0,i) = Complex(0.1,0.1);

	Array<Complex> m2(sizeLarge,sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(0.1,0.1);
	}

	Array<Complex> v3(1,sizeLarge);
	Array<Complex> v4(1,sizeLarge);
	for (int i = 0; i < v4.GetDescriptor().GetDim( 1 ); i++)
		v4(0,i) = Complex(0,double(sizeLarge)*2/100);

	v3 = v1*m2;

	TEST_CALL( v3, v4, BOOST_CURRENT_FUNCTION );
}

// matrix*matrix = matrix
void test_times_complex_9()
{
	Array<Complex> m1(sizeSmall,sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(0.1,0.1);
	}

	Array<Complex> m2(sizeSmall,sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(0.1,0.1);
	}

	Array<Complex> m3(sizeSmall,sizeSmall);
	Array<Complex> m4(sizeSmall,sizeSmall);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = Complex(0,double(sizeSmall)*2/100);
	}

	m3 = m1*m2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

void test_times_complex_10()
{	
	Array<Complex> m1(sizeLarge,sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(0.1,0.1);
	}

	Array<Complex> m2(sizeLarge,sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(0.1,0.1);
	}

	Array<Complex> m3(sizeLarge,sizeLarge);
	Array<Complex> m4(sizeLarge,sizeLarge);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = Complex(0,double(sizeLarge)*2/100);
	}

	m3 = m1*m2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

void test_times_complex_11()
{
	Array<Complex> m1(sizeLarge,sizeLarge);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = Complex(0.4,0.12);
	}

	Array<Complex> m2(sizeLarge,sizeLarge);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = Complex(i == j,0);
	}

	Array<Complex> m3(sizeLarge,sizeLarge);
	Array<Complex> m4 = m1;

	m3 = m1*m2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

// matrix n x m
void test_times_complex_12()
{
	Array<Complex> m1(sizeSmall,2*sizeSmall);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 0.15;
	}

	Array<Complex> m2(2*sizeSmall,sizeSmall);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = i == j;
	}

	Array<Complex> m4(sizeSmall,sizeSmall);
	for (int i = 0; i < m4.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m4.GetDescriptor().GetDim( 1 ); j++)
			m4(i,j) = 0.15;
	}
	
	Array<Complex> m3 = m1*m2;

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}

void test_times_complex_13()
{	
	const size_t size = 2000;
	Array<Complex> m1(2*size,size);
	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
			m1(i,j) = 0.15;
	}

	Array<Complex> m2(size,2*size);
	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
			m2(i,j) = i == j;
	}
	Array<Complex> m3 = m1*m2;

	Array<Complex> m4(m1.GetDescriptor().GetDim(0),m2.GetDescriptor().GetDim(1));
	for (int i = 0; i < 2*size; i++) {
		for (int j = 0; j < size; j++)
			m4(i,j) = 0.15;
	}

	TEST_CALL( m3, m4, BOOST_CURRENT_FUNCTION );
}