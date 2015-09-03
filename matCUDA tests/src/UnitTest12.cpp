#include "unitTests.h"

#define size 2048

void test_sin_float_1()
{
	Array<float> a = (eye<float>(3)*7*PI/2).sin();
	Array<float> control_a = eye<float>(3)*(-1);

	TEST_CALL( a, control_a, BOOST_CURRENT_FUNCTION );
}

void test_sind_float_1()
{
	Array<float> b = (eye<float>(3)*90).sind();
	Array<float> control_b = eye<float>(3);

	TEST_CALL( b, control_b, BOOST_CURRENT_FUNCTION );
}

void test_sin_double_1()
{
	Array<double> a = (eye<double>(3)*7*PI/2).sin();
	Array<double> control_a = eye<double>(3)*(-1);

	TEST_CALL( a, control_a, BOOST_CURRENT_FUNCTION );
}

void test_sind_double_1()
{
	Array<double> b = (eye<double>(3)*90).sind();
	Array<double> control_b = eye<double>(3);

	TEST_CALL( b, control_b, BOOST_CURRENT_FUNCTION );
}

void test_cos_float_1()
{
	Array<float> c = (eye<float>(3)*5*PI/2).cos();
	Array<float> control_c = eye<float>(3);
	for(int i = 0; i < control_c.GetDescriptor().GetDim(0); i++ ) {
		for(int j = 0; j < control_c.GetDescriptor().GetDim(1); j++ ) {
			if( i != j )
				control_c( i, j ) = 1;
			else
				control_c( i, j ) = 0;
		}
	}

	TEST_CALL( c, control_c, BOOST_CURRENT_FUNCTION );
}

void test_cosd_float_1()
{
	Array<float> d = (eye<float>(3)*60).cosd();
	Array<float> control_d = eye<float>(3);
	for(int i = 0; i < control_d.GetDescriptor().GetDim(0); i++ ) {
		for(int j = 0; j < control_d.GetDescriptor().GetDim(1); j++ ) {
			if( i != j )
				control_d( i, j ) = 1;
			else
				control_d( i, j ) = 0.5;
		}
	}

	TEST_CALL( d, control_d, BOOST_CURRENT_FUNCTION );
}

void test_cos_double_1()
{
	Array<double> c = (eye<double>(3)*5*PI/2).cos();
	Array<double> control_c = eye<double>(3);
	for(int i = 0; i < control_c.GetDescriptor().GetDim(0); i++ ) {
		for(int j = 0; j < control_c.GetDescriptor().GetDim(1); j++ ) {
			if( i != j )
				control_c( i, j ) = 1;
			else
				control_c( i, j ) = 0;
		}
	}

	TEST_CALL( c, control_c, BOOST_CURRENT_FUNCTION );
}

void test_cosd_double_1()
{
	Array<double> d = (eye<double>(3)*60).cosd();
	Array<double> control_d = eye<double>(3);
	for(int i = 0; i < control_d.GetDescriptor().GetDim(0); i++ ) {
		for(int j = 0; j < control_d.GetDescriptor().GetDim(1); j++ ) {
			if( i != j )
				control_d( i, j ) = 1;
			else
				control_d( i, j ) = 0.5;
		}
	}

	TEST_CALL( d, control_d, BOOST_CURRENT_FUNCTION );
}

void test_tan_float_1()
{
	Array<float> a = (eye<float>(3)*3*PI/4).tan();
	Array<float> control_a = eye<float>(3)*(-1);

	TEST_CALL( a, control_a, BOOST_CURRENT_FUNCTION );
}

void test_tand_float_1()
{
	Array<float> b = (eye<float>(3)*45).tand();
	Array<float> control_b = eye<float>(3);

	TEST_CALL( b, control_b, BOOST_CURRENT_FUNCTION );
}

void test_tan_double_1()
{
	Array<double> a = (eye<double>(3)*3*PI/4).tan();
	Array<double> control_a = eye<double>(3)*(-1);

	TEST_CALL( a, control_a, BOOST_CURRENT_FUNCTION );
}

void test_tand_double_1()
{
	Array<double> b = (eye<double>(3)*45).tand();
	Array<double> control_b = eye<double>(3);

	TEST_CALL( b, control_b, BOOST_CURRENT_FUNCTION );
}

void test_asin_float_1()
{
	Array<float> a = eye<float>(3)*PI/2*(-1);
	Array<float> control_a = eye<float>(3)*(-1);

	TEST_CALL( a, control_a.asin(), BOOST_CURRENT_FUNCTION );
}

void test_asind_float_1()
{
	Array<float> b = eye<float>(3)*90;
	Array<float> control_b = eye<float>(3);

	TEST_CALL( b, control_b.asind(), BOOST_CURRENT_FUNCTION );
}

void test_asin_double_1()
{
	Array<double> a = eye<double>(3)*PI/2*(-1);
	Array<double> control_a = eye<double>(3)*(-1);

	TEST_CALL( a, control_a.asin(), BOOST_CURRENT_FUNCTION );
}

void test_asind_double_1()
{
	Array<double> b = eye<double>(3)*90;
	Array<double> control_b = eye<double>(3);

	TEST_CALL( b, control_b.asind(), BOOST_CURRENT_FUNCTION );
}

void test_acos_float_1()
{
	Array<float> c = eye<float>(3)*PI/2;
	Array<float> control_c = eye<float>(3);
	for(int i = 0; i < control_c.GetDescriptor().GetDim(0); i++ ) {
		for(int j = 0; j < control_c.GetDescriptor().GetDim(1); j++ ) {
			if( i != j )
				control_c( i, j ) = 1;
			else
				control_c( i, j ) = 0;
		}
	}

	TEST_CALL( c, control_c.acos(), BOOST_CURRENT_FUNCTION );
}

void test_acosd_float_1()
{
	Array<float> d = eye<float>(3)*60;
	Array<float> control_d = eye<float>(3);
	for(int i = 0; i < control_d.GetDescriptor().GetDim(0); i++ ) {
		for(int j = 0; j < control_d.GetDescriptor().GetDim(1); j++ ) {
			if( i != j )
				control_d( i, j ) = 1;
			else
				control_d( i, j ) = 0.5;
		}
	}

	TEST_CALL( d, control_d.acosd(), BOOST_CURRENT_FUNCTION );
}

void test_acos_double_1()
{
	Array<double> c = (eye<double>(3)*PI/2);
	Array<double> control_c = eye<double>(3);
	for(int i = 0; i < control_c.GetDescriptor().GetDim(0); i++ ) {
		for(int j = 0; j < control_c.GetDescriptor().GetDim(1); j++ ) {
			if( i != j )
				control_c( i, j ) = 1;
			else
				control_c( i, j ) = 0;
		}
	}

	TEST_CALL( c, control_c.acos(), BOOST_CURRENT_FUNCTION );
}

void test_acosd_double_1()
{
	Array<double> d = (eye<double>(3)*60);
	Array<double> control_d = eye<double>(3);
	for(int i = 0; i < control_d.GetDescriptor().GetDim(0); i++ ) {
		for(int j = 0; j < control_d.GetDescriptor().GetDim(1); j++ ) {
			if( i != j )
				control_d( i, j ) = 1;
			else
				control_d( i, j ) = 0.5;
		}
	}

	TEST_CALL( d, control_d.acosd(), BOOST_CURRENT_FUNCTION );
}

void test_atan_float_1()
{
	Array<float> a = eye<float>(3)*PI/4*(-1);
	Array<float> control_a = eye<float>(3)*(-1);

	TEST_CALL( a, control_a.atan(), BOOST_CURRENT_FUNCTION );
}

void test_atand_float_1()
{
	Array<float> b = eye<float>(3)*45;
	Array<float> control_b = eye<float>(3);

	TEST_CALL( b, control_b.atand(), BOOST_CURRENT_FUNCTION );
}

void test_atan_double_1()
{
	Array<double> a = eye<double>(3)*PI/4*(-1);
	Array<double> control_a = eye<double>(3)*(-1);

	TEST_CALL( a, control_a.atan(), BOOST_CURRENT_FUNCTION );
}

void test_atand_double_1()
{
	Array<double> b = eye<double>(3)*45;
	Array<double> control_b = eye<double>(3);

	TEST_CALL( b, control_b.atand(), BOOST_CURRENT_FUNCTION );
}