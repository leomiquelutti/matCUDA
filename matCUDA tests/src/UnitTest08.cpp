#include "unitTests.h"

void test_determinant_float_1x1_1() 
{
	Array<float> m( 1, 1 );
	float control = 1;
	m( 0, 0 ) = 1;

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

// m.det() = 0
void test_determinant_float_1x1_2() 
{
	Array<float> m( 1, 1 );
	float control = 0;
	m( 0, 0 ) = 0;

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

void test_determinant_float_2x2_1()
{
	Array<float> m( 2, 2 );
	m( 0, 0 ) = 1;
	m( 0, 1 ) = 0;
	m( 1, 0 ) = 0;
	m( 1, 1 ) = 1;

	float control = 1;

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

// m.det() = 0
void test_determinant_float_2x2_2()
{
	Array<float> m( 2, 2 );
	m( 0, 0 ) = 1;
	m( 0, 1 ) = 1;
	m( 1, 0 ) = 1;
	m( 1, 1 ) = 1;

	float control = 0;

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

void test_determinant_float_2x2_3()
{
	Array<float> m( 2, 2 );
	m( 0, 0 ) = 1;
	m( 0, 1 ) = 2;
	m( 1, 0 ) = 3;
	m( 1, 1 ) = 4;

	float control = 2;

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

void test_determinant_float_3x3_1()
{
	Array<float> m( 3, 3 );
	m( 0, 0 ) = 1;
	m( 0, 1 ) = 0;
	m( 0, 2 ) = 0;
	m( 1, 0 ) = 0;
	m( 1, 1 ) = 1;
	m( 1, 2 ) = 0;
	m( 2, 0 ) = 0;
	m( 2, 1 ) = 0;
	m( 2, 2 ) = 1;

	float control = 1;

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

// m.det() = 0
void test_determinant_float_3x3_2()
{
	Array<float> m( 3, 3 );
	m( 0, 0 ) = 1;
	m( 0, 1 ) = 1;
	m( 0, 2 ) = 1;
	m( 1, 0 ) = 1;
	m( 1, 1 ) = 1;
	m( 1, 2 ) = 1;
	m( 2, 0 ) = 1;
	m( 2, 1 ) = 1;
	m( 2, 2 ) = 1;

	float control = 0;

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

void test_determinant_float_20x20_1()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_determinant_float_20x20_1//data.txt") );
	
	float control = 9.974225382144941e+03;

	TEST_CALL( abs(data.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

// m.det() = 0
void test_determinant_float_20x20_2()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_determinant_float_20x20_2//data.txt") );
	
	float control = 0;

	TEST_CALL( abs(data.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

void test_determinant_float_100x100_1()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_determinant_float_100x100_1//data.txt") );
	
	float control = 2.094843919208893e+03;

	TEST_CALL( abs(data.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

void test_determinant_double_1x1_1() 
{
	Array<double> m( 1, 1 );
	m( 0, 0 ) = 1;
	
	double control = 1;

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

// m.det() = 0
void test_determinant_double_1x1_2() 
{
	Array<double> m( 1, 1 );
	m( 0, 0 ) = 0;
	
	double control = 0;

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

void test_determinant_double_2x2_1()
{
	Array<double> m( 2, 2 );
	m( 0, 0 ) = 1;
	m( 0, 1 ) = 0;
	m( 1, 0 ) = 0;
	m( 1, 1 ) = 1;
	
	double control = 1;

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

// m.det() = 0
void test_determinant_double_2x2_2()
{
	Array<double> m( 2, 2 );
	m( 0, 0 ) = 1;
	m( 0, 1 ) = 1;
	m( 1, 0 ) = 1;
	m( 1, 1 ) = 1;
	
	double control = 0;

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

void test_determinant_double_3x3_1()
{
	Array<double> m( 3, 3 );
	m( 0, 0 ) = 1;
	m( 0, 1 ) = 0;
	m( 0, 2 ) = 0;
	m( 1, 0 ) = 0;
	m( 1, 1 ) = 1;
	m( 1, 2 ) = 0;
	m( 2, 0 ) = 0;
	m( 2, 1 ) = 0;
	m( 2, 2 ) = 1;
	
	double control = 1;

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

// m.det() = 0
void test_determinant_double_3x3_2()
{
	Array<double> m( 3, 3 );
	m( 0, 0 ) = 1;
	m( 0, 1 ) = 1;
	m( 0, 2 ) = 1;
	m( 1, 0 ) = 1;
	m( 1, 1 ) = 1;
	m( 1, 2 ) = 1;
	m( 2, 0 ) = 1;
	m( 2, 1 ) = 1;
	m( 2, 2 ) = 1;
	
	double control = 0;

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

void test_determinant_double_20x20_1()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_determinant_double_20x20_1//data.txt") );
	
	double control = 9.974225382144941e+03;

	TEST_CALL( abs(data.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

// m.det() = 0
void test_determinant_double_20x20_2()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_determinant_double_20x20_2//data.txt") );
	
	double control = 0;

	TEST_CALL( abs(data.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

void test_determinant_double_100x100_1()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_determinant_double_100x100_1//data.txt") );
	
	double control = 3.899089460444517e+02;

	TEST_CALL( abs(data.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

void test_determinant_complex_1x1_1()
{
	Array<Complex> m( 1, 1 );
	m( 0, 0 ) = Complex( 1, 1 );
	Complex control = Complex( 1, 1 );

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

// m.det() = 0
void test_determinant_complex_1x1_2()
{
	Array<Complex> m( 1, 1 );
	m( 0, 0 ) = Complex( 0, 0 );
	
	Complex det = m.determinant();
	Complex control = Complex( 0, 0);

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

void test_determinant_complex_2x2_1()
{
	Array<Complex> m( 2, 2 );
	m( 0, 0 ) = Complex( 1, 1 );
	m( 0, 1 ) = Complex( 1, 0 );
	m( 1, 0 ) = Complex( 1, 0 );
	m( 1, 1 ) = Complex( 1, 1 );

	Complex control = Complex( -1, 2);

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

// m.det() = 0
void test_determinant_complex_2x2_2()
{
	Array<Complex> m( 2, 2 );
	m( 0, 0 ) = Complex( 1, 1 );
	m( 0, 1 ) = Complex( 1, 0 );
	m( 1, 0 ) = Complex( 2, 2 );
	m( 1, 1 ) = Complex( 2, 0 );
	
	Complex det = m.determinant();
	Complex control = Complex( 0, 0);

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

void test_determinant_complex_3x3_1()
{
	Array<Complex> m( 3, 3 );
	m( 0, 0 ) = Complex( 1, 1 );
	m( 0, 1 ) = Complex( 1, 0 );
	m( 0, 2 ) = Complex( 0, 1 );
	m( 1, 0 ) = Complex( 2, 2 );
	m( 1, 1 ) = Complex( 1, 0 );
	m( 1, 2 ) = Complex( 2, -2 );
	m( 2, 0 ) = Complex( 3, 0 );
	m( 2, 1 ) = Complex( 1, 0 );
	m( 2, 2 ) = Complex( 0, 4 );
	
	Complex det = m.determinant();
	Complex control = Complex( 4, -11);

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

// m.det() = 0
void test_determinant_complex_3x3_2()
{
	Array<Complex> data = read_file_matrix<Complex>( std::string("unit_tests_files//test_determinant_complex_3x3_1//data.txt") );
	
	Complex control = Complex(  -2.465963761762253e-15, 7.725952660079466e-16 );

	TEST_CALL( abs(data.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

void test_determinant_complex_4x4_1()
{
	Array<Complex> m( 4, 4 );
	m( 0, 0 ) = Complex( 1, 1 );
	m( 0, 1 ) = Complex( 1, 0 );
	m( 0, 2 ) = Complex( 0, 1 );
	m( 0, 3 ) = Complex( 0, 1 );
	m( 1, 0 ) = Complex( 2, 2 );
	m( 1, 1 ) = Complex( 1, 0 );
	m( 1, 2 ) = Complex( 2, -2 );
	m( 1, 3 ) = Complex( 0, 1 );
	m( 2, 0 ) = Complex( 3, 0 );
	m( 2, 1 ) = Complex( 1, 0 );
	m( 2, 2 ) = Complex( 0, 4 );
	m( 2, 3 ) = Complex( 0, 1 );
	m( 3, 0 ) = Complex( 3, 0 );
	m( 3, 1 ) = Complex( 1, 1 );
	m( 3, 2 ) = Complex( 0, 4 );
	m( 3, 3 ) = Complex( 0, 2 );

	Complex det = m.determinant();
	Complex control = Complex( 15, -7);

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

// m.det() = 0
void test_determinant_complex_4x4_2()
{
	Array<Complex> m( 4, 4 );
	m( 0, 0 ) = Complex( 1, 1 );
	m( 0, 1 ) = Complex( 1, 0 );
	m( 0, 2 ) = Complex( 0, 1 );
	m( 0, 3 ) = Complex( 0, 1 );
	m( 1, 0 ) = Complex( 2, 2 );
	m( 1, 1 ) = Complex( 1, 0 );
	m( 1, 2 ) = Complex( 2, -2 );
	m( 1, 3 ) = Complex( 0, 1 );
	m( 2, 0 ) = Complex( 3, 0 );
	m( 2, 1 ) = Complex( 1, 0 );
	m( 2, 2 ) = Complex( 0, 4 );
	m( 2, 3 ) = Complex( 0, 1 );
	m( 3, 0 ) = Complex( 3, 0 );
	m( 3, 1 ) = Complex( 1, 0 );
	m( 3, 2 ) = Complex( 0, 4 );
	m( 3, 3 ) = Complex( 0, 1 );

	TEST_CALL( m.determinant(), Complex( 0, 0), BOOST_CURRENT_FUNCTION );
}

void test_determinant_complex_5x5_1()
{
	Array<Complex> m( 5, 5 );
	m(0,0) = Complex(    7.20062084171543270000e-01,    1.34233246597730590000e+00);
	m(0,1) = Complex(    8.60555690148311130000e-01,    3.92421335311681220000e-02);
	m(0,2) = Complex(    2.18668479048691470000e-01,    1.72773644586722130000e+00);
	m(0,3) = Complex(    4.57375168040441650000e-01,    2.86312044167152280000e-01);
	m(0,4) = Complex(    1.33808518016311080000e+00,    1.83364254050747610000e+00);
	m(1,0) = Complex(    9.08424743706797160000e-01,    1.19917109624562770000e+00);
	m(1,1) = Complex(    1.38750514826782690000e+00,    8.70351091393903650000e-01);
	m(1,2) = Complex(    7.79861314312970010000e-01,    1.95395836322886260000e-01);
	m(1,3) = Complex(    1.66837812211379900000e+00,    1.11874114480600780000e+00);
	m(1,4) = Complex(    1.00042264856220480000e+00,    1.97393654956731620000e+00);
	m(2,0) = Complex(    7.72779797184344640000e-01,    1.11952314775580990000e-01);
	m(2,1) = Complex(    1.89042697433692060000e+00,    1.66444295056563530000e+00);
	m(2,2) = Complex(    1.18180946081067280000e+00,    1.81610440637353830000e+00);
	m(2,3) = Complex(    3.12893853930540370000e-02,    9.15924789464694910000e-03);
	m(2,4) = Complex(    4.35987597500307130000e-01,    1.01026620359764510000e+00);
	m(3,0) = Complex(    1.55110928450107100000e+00,    1.12686037086713540000e-01);
	m(3,1) = Complex(    1.56846519646348170000e+00,    1.23478034290816850000e+00);
	m(3,2) = Complex(    9.18760095932650730000e-01,    2.16033388273517920000e-01);
	m(3,3) = Complex(    1.72742173011173140000e+00,    1.53336399724297490000e+00);
	m(3,4) = Complex(    1.14323145080817820000e+00,    5.42843248835030100000e-01);
	m(4,0) = Complex(    1.46854221137187870000e+00,    3.05001274004940590000e-01);
	m(4,1) = Complex(    1.41114371626213760000e+00,    1.04025883062148390000e+00);
	m(4,2) = Complex(    1.00679973477908250000e-01,    1.03399351619388980000e+00);
	m(4,3) = Complex(    1.56138106064868550000e-01,    1.69741845291656480000e+00);
	m(4,4) = Complex(    2.44378301846839510000e-01,    2.01501023842471260000e-01);

	Complex control = Complex(   -1.00028861738829940000e+01,    9.93608334223188020000e+00);

	TEST_CALL( abs(m.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

void test_determinant_complex_20x20_1()
{
	Array<Complex> data = read_file_matrix<Complex>( std::string("unit_tests_files//test_determinant_complex_20x20_1//data.txt") );
	
	Complex control = Complex( 8.0892e+07, 2.0180e+06 );

	TEST_CALL( abs(data.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

// m.det() = 0
void test_determinant_complex_20x20_2()
{
	Array<Complex> data = read_file_matrix<Complex>( std::string("unit_tests_files//test_determinant_complex_20x20_2//data.txt") );
	
	Complex control = 0;

	TEST_CALL( abs(data.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}

void test_determinant_complex_100x100_1()
{
	Array<Complex> data = read_file_matrix<Complex>( std::string("unit_tests_files//test_determinant_complex_100x100_1//data.txt") );
	
	Complex control = Complex( -2.2809e-04, 2.5320e-04 );

	TEST_CALL( abs(data.determinant()), abs(control), BOOST_CURRENT_FUNCTION );
}
