#include "unitTests.h"

// float type
void test_invert_float_1x1_1()
{
	Array<float> m(1);
	m(0) = 4;

	Array<float> control(1);
	control(0) = 0.25;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

// m.det = 0
void test_invert_float_1x1_2()
{
	Array<float> m(1);
	m(0) = 0;

	Array<float> inverse(1);
	Array<float> control(1);
	control(0) = 0;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_float_2x2_1()
{
	Array<float> m(2,2);
	m(0,0) = 1;
	m(0,1) = 0;
	m(1,0) = 0;
	m(1,1) = 1;

	Array<float> inverse(2,2);
	Array<float> control(2,2);
	control(0,0) = 1;
	control(0,1) = 0;
	control(1,0) = 0;
	control(1,1) = 1;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_float_2x2_2()
{
	Array<float> m(2,2);
	m(0,0) = 5.0;
	m(0,1) = 3.0;
	m(1,0) = 3.0;
	m(1,1) = 2.0;

	Array<float> inverse(2,2);
	Array<float> control(2,2);
	control(0,0) = 2.0;
	control(0,1) = -3.0;
	control(1,0) = -3.0;
	control(1,1) = 5.0;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

// m.det() = 0
void test_invert_float_2x2_3()
{
	Array<float> m(2,2);
	m(0,0) = 5.0;
	m(0,1) = 3.0;
	m(1,0) = 10.0;
	m(1,1) = 6.0;

	TEST_CALL( m.invert(), m, BOOST_CURRENT_FUNCTION );
}

void test_invert_float_3x3_1()
{
	Array<float> m(3,3);
	m(0,0) = 1;
	m(0,1) = 0;
	m(0,2) = 0;
	m(1,0) = 0;
	m(1,1) = 1;
	m(1,2) = 0;
	m(2,0) = 0;
	m(2,1) = 0;
	m(2,2) = 1;

	Array<float> inverse(3,3);
	Array<float> control(3,3);
	control(0,0) = 1;
	control(0,1) = 0;
	control(0,2) = 0;
	control(1,0) = 0;
	control(1,1) = 1;
	control(1,2) = 0;
	control(2,0) = 0;
	control(2,1) = 0;
	control(2,2) = 1;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_float_3x3_2()
{
	Array<float> m(3,3);
	m(0,0) = 1.0;
	m(0,1) = 2.0;
	m(0,2) = 3.0;
	m(1,0) = 0.0;
	m(1,1) = 1.0;
	m(1,2) = 4.0;
	m(2,0) = 5.0;
	m(2,1) = 6.0;
	m(2,2) = 0.0;

	Array<float> inverse(3,3);
	Array<float> control(3,3);
	control(0,0) = -24.0;
	control(0,1) = 18.0;
	control(0,2) = 5.0;
	control(1,0) = 20.0;
	control(1,1) = -15.0;
	control(1,2) = -4.0;
	control(2,0) = -5.0;
	control(2,1) = 4.0;
	control(2,2) = 1.0;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_float_3x3_3()
{
	Array<float> m(3,3);
	m(0,0) = 1;
	m(0,1) = 0;
	m(0,2) = 1;
	m(1,0) = 0;
	m(1,1) = 0.5;
	m(1,2) = 1;
	m(2,0) = 0;
	m(2,1) = 0;
	m(2,2) = 1;

	Array<float> inverse(3,3);
	Array<float> control(3,3);
	control(0,0) = 1;
	control(0,1) = 0;
	control(0,2) = -1;
	control(1,0) = 0;
	control(1,1) = 2;
	control(1,2) = -2;
	control(2,0) = 0;
	control(2,1) = 0;
	control(2,2) = 1;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_float_20x20_1()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_invert_float_20x20_1//data.txt") );
	Array<float> control = read_file_matrix<float>( std::string("unit_tests_files//test_invert_float_20x20_1//control.txt") );
	TEST_CALL( data.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_float_20x20_2()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_invert_float_20x20_2//data.txt") );
	Array<float> control = read_file_matrix<float>( std::string("unit_tests_files//test_invert_float_20x20_2//control.txt") );

	TEST_CALL( data.invert(), control, BOOST_CURRENT_FUNCTION );
}


//// double type
void test_invert_double_1x1_1()
{
	Array<double> m(1);
	m(0) = 4;

	Array<double> inverse(1);
	Array<double> control(1);
	control(0) = 0.25;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

// m.det = 0
void test_invert_double_1x1_2()
{
	Array<double> m(1);
	m(0) = 0;

	Array<double> inverse(1);
	Array<double> control(1);
	control(0) = 0;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_double_2x2_1()
{
	Array<double> m(2,2);
	m(0,0) = 1;
	m(0,1) = 0;
	m(1,0) = 0;
	m(1,1) = 1;

	Array<double> inverse(2,2);
	Array<double> control(2,2);
	control(0,0) = 1;
	control(0,1) = 0;
	control(1,0) = 0;
	control(1,1) = 1;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_double_2x2_2()
{
	Array<double> m(2,2);
	m(0,0) = 5.0;
	m(0,1) = 3.0;
	m(1,0) = 3.0;
	m(1,1) = 2.0;

	Array<double> inverse(2,2);
	Array<double> control(2,2);
	control(0,0) = 2.0;
	control(0,1) = -3.0;
	control(1,0) = -3.0;
	control(1,1) = 5.0;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

// m.det() = 0
void test_invert_double_2x2_3()
{
	Array<double> m(2,2);
	m(0,0) = 5.0;
	m(0,1) = 3.0;
	m(1,0) = 10.0;
	m(1,1) = 6.0;

	TEST_CALL( m.invert(), m, BOOST_CURRENT_FUNCTION );
}

void test_invert_double_3x3_1()
{
	Array<double> m(3,3);
	m(0,0) = 1;
	m(0,1) = 0;
	m(0,2) = 0;
	m(1,0) = 0;
	m(1,1) = 1;
	m(1,2) = 0;
	m(2,0) = 0;
	m(2,1) = 0;
	m(2,2) = 1;

	Array<double> inverse(3,3);
	Array<double> control(3,3);
	control(0,0) = 1;
	control(0,1) = 0;
	control(0,2) = 0;
	control(1,0) = 0;
	control(1,1) = 1;
	control(1,2) = 0;
	control(2,0) = 0;
	control(2,1) = 0;
	control(2,2) = 1;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_double_3x3_2()
{
	Array<double> m(3,3);
	m(0,0) = 1.0;
	m(0,1) = 2.0;
	m(0,2) = 3.0;
	m(1,0) = 0.0;
	m(1,1) = 1.0;
	m(1,2) = 4.0;
	m(2,0) = 5.0;
	m(2,1) = 6.0;
	m(2,2) = 0.0;

	Array<double> control(3,3);
	control(0,0) = -24.0;
	control(0,1) = 18.0;
	control(0,2) = 5.0;
	control(1,0) = 20.0;
	control(1,1) = -15.0;
	control(1,2) = -4.0;
	control(2,0) = -5.0;
	control(2,1) = 4.0;
	control(2,2) = 1.0;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_double_3x3_3()
{
	Array<double> m(3,3);
	m(0,0) = 1;
	m(0,1) = 0;
	m(0,2) = 1;
	m(1,0) = 0;
	m(1,1) = 0.5;
	m(1,2) = 1;
	m(2,0) = 0;
	m(2,1) = 0;
	m(2,2) = 1;

	Array<double> control(3,3);
	control(0,0) = 1;
	control(0,1) = 0;
	control(0,2) = -1;
	control(1,0) = 0;
	control(1,1) = 2;
	control(1,2) = -2;
	control(2,0) = 0;
	control(2,1) = 0;
	control(2,2) = 1;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_double_20x20_1()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_invert_double_20x20_1//data.txt") );
	Array<double> control = read_file_matrix<double>( std::string("unit_tests_files//test_invert_double_20x20_1//control.txt") );

	TEST_CALL( data.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_double_20x20_2()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_invert_double_20x20_2//data.txt") );
	Array<double> control = read_file_matrix<double>( std::string("unit_tests_files//test_invert_double_20x20_2//control.txt") );

	TEST_CALL( data.invert(), control, BOOST_CURRENT_FUNCTION );
}


//// complex type
void test_invert_complex_1x1_1()
{
	Array<Complex> m(1);
	m(0) = 4;

	Array<Complex> control(1);
	control(0) = 0.25;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

// m.det = 0
void test_invert_complex_1x1_2()
{
	Array<Complex> m(1);
	m(0) = 0.5;

	Array<Complex> inverse(1);
	Array<Complex> control(1);
	control(0) = 2;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_complex_2x2_1()
{
	Array<Complex> m(2,2);
	m(0,0) = 1;
	m(0,1) = 0;
	m(1,0) = 0;
	m(1,1) = 1;

	Array<Complex> inverse(2,2);
	Array<Complex> control(2,2);
	control(0,0) = 1;
	control(0,1) = 0;
	control(1,0) = 0;
	control(1,1) = 1;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_complex_2x2_2()
{
	Array<Complex> m(2,2);
	m(0,0) = 5.0;
	m(0,1) = 3.0;
	m(1,0) = 3.0;
	m(1,1) = 2.0;

	Array<Complex> control(2,2);
	control(0,0) = 2.0;
	control(0,1) = -3.0;
	control(1,0) = -3.0;
	control(1,1) = 5.0;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

// m.det() = 0
void test_invert_complex_2x2_3()
{
	Array<Complex> m(2,2);
	m(0,0) = 5.0;
	m(0,1) = 3.0;
	m(1,0) = 10.0;
	m(1,1) = 6.0;

	TEST_CALL( m.invert(), m, BOOST_CURRENT_FUNCTION );
}

void test_invert_complex_3x3_1()
{
	Array<Complex> m(3,3);
	m(0,0) = 1;
	m(0,1) = 0;
	m(0,2) = 0;
	m(1,0) = 0;
	m(1,1) = 1;
	m(1,2) = 0;
	m(2,0) = 0;
	m(2,1) = 0;
	m(2,2) = 1;

	Array<Complex> control(3,3);
	control(0,0) = 1;
	control(0,1) = 0;
	control(0,2) = 0;
	control(1,0) = 0;
	control(1,1) = 1;
	control(1,2) = 0;
	control(2,0) = 0;
	control(2,1) = 0;
	control(2,2) = 1;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_complex_3x3_2()
{
	Array<Complex> m(3,3);
	m(0,0) = 1.0;
	m(0,1) = 2.0;
	m(0,2) = 3.0;
	m(1,0) = 0.0;
	m(1,1) = 1.0;
	m(1,2) = 4.0;
	m(2,0) = 5.0;
	m(2,1) = 6.0;
	m(2,2) = 0.0;

	Array<Complex> control(3,3);
	control(0,0) = -24.0;
	control(0,1) = 18.0;
	control(0,2) = 5.0;
	control(1,0) = 20.0;
	control(1,1) = -15.0;
	control(1,2) = -4.0;
	control(2,0) = -5.0;
	control(2,1) = 4.0;
	control(2,2) = 1.0;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_complex_3x3_3()
{
	Array<Complex> m(3,3);
	m(0,0) = 1;
	m(0,1) = 0;
	m(0,2) = 1;
	m(1,0) = 0;
	m(1,1) = 0.5;
	m(1,2) = 1;
	m(2,0) = 0;
	m(2,1) = 0;
	m(2,2) = 1;

	Array<Complex> control(3,3);
	control(0,0) = 1;
	control(0,1) = 0;
	control(0,2) = -1;
	control(1,0) = 0;
	control(1,1) = 2;
	control(1,2) = -2;
	control(2,0) = 0;
	control(2,1) = 0;
	control(2,2) = 1;

	TEST_CALL( m.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_complex_20x20_1()
{
	Array<Complex> data = read_file_matrix<Complex>( std::string("unit_tests_files//test_invert_complex_20x20_1//data.txt") );
	Array<Complex> control = read_file_matrix<Complex>( std::string("unit_tests_files//test_invert_complex_20x20_1//control.txt") );

	TEST_CALL( data.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_complex_20x20_2()
{
	Array<Complex> data = read_file_matrix<Complex>( std::string("unit_tests_files//test_invert_complex_20x20_2//data.txt") );
	Array<Complex> control = read_file_matrix<Complex>( std::string("unit_tests_files//test_invert_complex_20x20_2//control.txt") );

	TEST_CALL( data.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_complex_20x20_3()
{
	Array<Complex> data = read_file_matrix<Complex>( std::string("unit_tests_files//test_invert_complex_20x20_3//data.txt") );
	Array<Complex> control = read_file_matrix<Complex>( std::string("unit_tests_files//test_invert_complex_20x20_3//control.txt") );

	TEST_CALL( data.invert(), control, BOOST_CURRENT_FUNCTION );
}

void test_invert_complex_100x100_1()
{
	Array<Complex> data = read_file_matrix<Complex>( std::string("unit_tests_files//test_invert_complex_100x100_1//data.txt") );
	Array<Complex> control = read_file_matrix<Complex>( std::string("unit_tests_files//test_invert_complex_100x100_1//control.txt") );

	TEST_CALL( data.invert(), control, BOOST_CURRENT_FUNCTION );
}