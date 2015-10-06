#include "unitTests.h"

void test_LS_float_1()
{
	Array<float> A( 5, 3 );
	Array<float> B( 5, 2 );
	Array<float> control(3,2);

	A(0,0) = 1;	A(0,1) = 1;	A(0,2) = 1;
	A(1,0) = 2;	A(1,1) = 3;	A(1,2) = 4;
	A(2,0) = 3;	A(2,1) = 5;	A(2,2) = 2;
	A(3,0) = 4;	A(3,1) = 2;	A(3,2) = 5;
	A(4,0) = 5;	A(4,1) = 4;	A(4,2) = 3;

	B(0,0) = -10; B(0,1) = -3;
	B(1,0) = 12; B(1,1) = 14;
	B(2,0) = 14; B(2,1) = 12;
	B(3,0) = 16; B(3,1) = 16;
	B(4,0) = 18; B(4,1) = 16;

	control(0,0) = 2; control(0,1) = 1;
	control(1,0) = 1; control(1,1) = 1;
	control(2,0) = 1; control(2,1) = 2;

	TEST_CALL( B.LS( A ), control, BOOST_CURRENT_FUNCTION );
}

void test_LS_float_2()
{
	Array<float> y = read_file_vector<float>( std::string("unit_tests_files//test_LS_float_2//y.txt") );
	Array<float> x = read_file_matrix<float>( std::string("unit_tests_files//test_LS_float_2//x.txt") );
	Array<float> control = read_file_vector<float>( std::string("unit_tests_files//test_LS_float_2//control.txt") );

	Array<float> data = y.LS( x );

	TEST_CALL( data, control, BOOST_CURRENT_FUNCTION );
}

void test_LS_double_1()
{
	Array<double> A( 5, 3 );
	Array<double> B( 5, 2 );
	Array<double> control(3,2);

	A(0,0) = 1;	A(0,1) = 1;	A(0,2) = 1;
	A(1,0) = 2;	A(1,1) = 3;	A(1,2) = 4;
	A(2,0) = 3;	A(2,1) = 5;	A(2,2) = 2;
	A(3,0) = 4;	A(3,1) = 2;	A(3,2) = 5;
	A(4,0) = 5;	A(4,1) = 4;	A(4,2) = 3;

	B(0,0) = -10; B(0,1) = -3;
	B(1,0) = 12; B(1,1) = 14;
	B(2,0) = 14; B(2,1) = 12;
	B(3,0) = 16; B(3,1) = 16;
	B(4,0) = 18; B(4,1) = 16;

	control(0,0) = 2; control(0,1) = 1;
	control(1,0) = 1; control(1,1) = 1;
	control(2,0) = 1; control(2,1) = 2;

	TEST_CALL( B.LS( A ), control, BOOST_CURRENT_FUNCTION );
}

void test_LS_double_2()
{
	Array<double> y = read_file_vector<double>( std::string("unit_tests_files//test_LS_double_2//y.txt") );
	Array<double> x = read_file_matrix<double>( std::string("unit_tests_files//test_LS_double_2//x.txt") );
	Array<double> control = read_file_vector<double>( std::string("unit_tests_files//test_LS_double_2//control.txt") );

	Array<double> data = y.LS( x );

	TEST_CALL( data, control, BOOST_CURRENT_FUNCTION );
}

void test_LS_complex_1()
{
	Array<Complex> A( 5, 3 );
	Array<Complex> B( 5, 2 );
	Array<Complex> control(3,2);

	A(0,0) = 1;	A(0,1) = 1;	A(0,2) = 1;
	A(1,0) = 2;	A(1,1) = 3;	A(1,2) = 4;
	A(2,0) = 3;	A(2,1) = 5;	A(2,2) = 2;
	A(3,0) = 4;	A(3,1) = 2;	A(3,2) = 5;
	A(4,0) = 5;	A(4,1) = 4;	A(4,2) = 3;

	B(0,0) = -10; B(0,1) = -3;
	B(1,0) = 12; B(1,1) = 14;
	B(2,0) = 14; B(2,1) = 12;
	B(3,0) = 16; B(3,1) = 16;
	B(4,0) = 18; B(4,1) = 16;

	control(0,0) = 2; control(0,1) = 1;
	control(1,0) = 1; control(1,1) = 1;
	control(2,0) = 1; control(2,1) = 2;

	TEST_CALL( B.LS( A ), control, BOOST_CURRENT_FUNCTION );
}

void test_LS_complex_2()
{
	Array<Complex> y = read_file_matrix<Complex>( std::string("unit_tests_files//test_LS_complex_2//y.txt") );
	Array<Complex> x = read_file_matrix<Complex>( std::string("unit_tests_files//test_LS_complex_2//x.txt") );
	Array<Complex> control = read_file_matrix<Complex>( std::string("unit_tests_files//test_LS_complex_2//control.txt") );

	Array<Complex> data = y.LS( x );

	TEST_CALL( data, control, BOOST_CURRENT_FUNCTION );
}

void test_LS_complex_3()
{
	Array<Complex> y = read_file_matrix<Complex>( std::string("unit_tests_files//test_LS_complex_3//y.txt") );
	Array<Complex> x = read_file_matrix<Complex>( std::string("unit_tests_files//test_LS_complex_3//x.txt") );
	Array<Complex> control = read_file_matrix<Complex>( std::string("unit_tests_files//test_LS_complex_3//control.txt") );

	Array<Complex> data = y.LS( x );

	TEST_CALL( data, control, BOOST_CURRENT_FUNCTION );
}
