#include "unitTests.h"

using namespace boost::unit_test;

test_suite*
init_unit_test_suite( int argc, char* argv[] )
{
	test_suite* ts1 = BOOST_TEST_SUITE( "Array test suite" );
		
	//ts1->add( BOOST_TEST_CASE( &draft ) );

	// UnitTest1.cpp
	ts1->add( BOOST_TEST_CASE( &test_plus_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_float_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_float_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_float_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_double_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_double_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_double_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_complex_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_complex_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_complex_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_complex_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_complex_5 ) );

	// UnitTest2.cpp
	ts1->add( BOOST_TEST_CASE( &test_plus_equal_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_equal_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_equal_float_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_equal_float_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_equal_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_equal_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_equal_double_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_equal_double_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_equal_complex_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_equal_complex_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_equal_complex_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_plus_equal_complex_4 ) );

	// UnitTest3.cpp
	ts1->add( BOOST_TEST_CASE( &test_minus_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_float_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_float_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_double_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_double_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_complex_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_complex_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_complex_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_complex_4 ) );

	// UnitTest4.cpp
	ts1->add( BOOST_TEST_CASE( &test_minus_equal_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_equal_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_equal_float_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_equal_float_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_equal_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_equal_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_equal_double_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_equal_double_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_equal_complex_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_equal_complex_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_equal_complex_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_minus_equal_complex_4 ) );
	
	// UnitTest5.cpp	
	ts1->add( BOOST_TEST_CASE( &test_transpose_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_transpose_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_transpose_float_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_transpose_float_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_transpose_float_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_transpose_float_6 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_float_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_float_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_float_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_float_6 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_float_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_float_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_float_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_float_6 ) );
	ts1->add( BOOST_TEST_CASE( &test_transpose_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_transpose_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_transpose_double_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_transpose_double_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_transpose_double_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_transpose_double_6 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_double_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_double_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_double_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_double_6 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_double_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_double_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_double_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_double_6 ) );
	ts1->add( BOOST_TEST_CASE( &test_transpose_complex_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_transpose_complex_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_transpose_complex_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_transpose_complex_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_transpose_complex_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_transpose_complex_6 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_complex_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_complex_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_complex_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_complex_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_complex_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_conjugate_complex_6 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_complex_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_complex_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_complex_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_complex_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_complex_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_hermitian_complex_6 ) );
	
	// UnitTest6.cpp
	ts1->add( BOOST_TEST_CASE( &test_times_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_float_11 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_float_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_float_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_float_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_float_6 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_float_7 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_float_8 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_float_9 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_float_10 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_float_12 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_float_13 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_float_14 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_float_15 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_float_16 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_double_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_double_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_double_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_double_6 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_double_7 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_double_8 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_double_9 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_double_10 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_double_11 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_double_12 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_double_13 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_double_14 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_complex_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_complex_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_complex_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_complex_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_complex_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_complex_6 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_complex_7 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_complex_8 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_complex_9 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_complex_10 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_complex_11 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_complex_12 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_complex_13 ) );

	// UnitTest7.cpp
	ts1->add( BOOST_TEST_CASE( &test_times_equal_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_equal_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_equal_float_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_equal_float_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_equal_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_equal_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_equal_double_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_equal_double_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_equal_complex_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_equal_complex_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_equal_complex_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_equal_complex_4 ) );

	// UnitTest8.cpp
	ts1->add( BOOST_TEST_CASE( &test_determinant_float_1x1_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_float_1x1_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_float_2x2_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_float_2x2_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_float_2x2_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_float_3x3_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_float_3x3_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_float_20x20_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_float_20x20_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_float_100x100_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_double_1x1_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_double_1x1_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_double_2x2_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_double_2x2_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_double_3x3_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_double_3x3_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_double_20x20_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_double_20x20_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_double_100x100_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_complex_1x1_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_complex_1x1_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_complex_2x2_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_complex_2x2_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_complex_3x3_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_complex_3x3_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_complex_4x4_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_complex_4x4_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_complex_5x5_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_complex_20x20_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_complex_20x20_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_determinant_complex_100x100_1 ) );

	// UnitTest9.cpp
	ts1->add( BOOST_TEST_CASE( &test_invert_float_1x1_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_float_1x1_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_float_2x2_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_float_2x2_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_float_2x2_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_float_3x3_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_float_3x3_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_float_3x3_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_float_20x20_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_float_20x20_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_double_1x1_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_double_1x1_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_double_2x2_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_double_2x2_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_double_2x2_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_double_3x3_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_double_3x3_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_double_3x3_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_double_20x20_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_double_20x20_2 ) );	
	ts1->add( BOOST_TEST_CASE( &test_invert_complex_1x1_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_complex_1x1_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_complex_2x2_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_complex_2x2_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_complex_2x2_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_complex_3x3_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_complex_3x3_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_complex_3x3_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_complex_20x20_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_complex_20x20_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_complex_20x20_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_invert_complex_100x100_1 ) );

	// UnitTest10.cpp
	ts1->add( BOOST_TEST_CASE( &test_minor_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_minor_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_minor_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_minor_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_minor_complex_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_minor_complex_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_norm_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_norm_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_norm_complex_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_eye_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_eye_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_eye_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_eye_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_eye_complex_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_eye_complex_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_times_TElement_float ) );
	ts1->add( BOOST_TEST_CASE( &test_times_TElement_double ) );
	ts1->add( BOOST_TEST_CASE( &test_times_TElement_complex ) );

	//UnitTest11.cpp
	ts1->add( BOOST_TEST_CASE( &test_LU_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_LU_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_LU_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_LU_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_LU_complex_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_LU_complex_2 ) );

	// UnitTest12.cpp
	ts1->add( BOOST_TEST_CASE( &test_sin_float_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_sind_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_sin_double_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_sind_double_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_cos_float_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_cosd_float_1 ) );		
	ts1->add( BOOST_TEST_CASE( &test_cos_double_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_cosd_double_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_tan_float_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_tand_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_tan_double_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_tand_double_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_asin_float_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_asind_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_asin_double_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_asind_double_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_acos_float_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_acosd_float_1 ) );		
	ts1->add( BOOST_TEST_CASE( &test_acos_double_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_acosd_double_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_atan_float_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_atand_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_atan_double_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_atand_double_1 ) );	

	// UnitTest13.cpp
	ts1->add( BOOST_TEST_CASE( &test_FFT_C2C_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_FFT_C2C_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_FFT_C2C_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_FFT_R2C_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_FFT_R2C_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_FFT_R2C_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_FFT_D2Z_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_FFT_D2Z_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_FFT_D2Z_3 ) );

	// UnitTest14.cpp
	ts1->add( BOOST_TEST_CASE( &test_LS_float_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_LS_float_2 ) );	
	ts1->add( BOOST_TEST_CASE( &test_LS_double_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_LS_double_2 ) );	
	ts1->add( BOOST_TEST_CASE( &test_LS_complex_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_LS_complex_2 ) );	
	ts1->add( BOOST_TEST_CASE( &test_LS_complex_3 ) );

	// UnitTest15.cpp
	ts1->add( BOOST_TEST_CASE( &test_detrend_float_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_detrend_float_2 ) );	
	ts1->add( BOOST_TEST_CASE( &test_detrend_float_3 ) );	
	ts1->add( BOOST_TEST_CASE( &test_detrend_float_4 ) );	
	ts1->add( BOOST_TEST_CASE( &test_detrend_double_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_detrend_double_2 ) );	
	ts1->add( BOOST_TEST_CASE( &test_detrend_double_3 ) );	
	ts1->add( BOOST_TEST_CASE( &test_detrend_double_4 ) );	

	// UnitTest16.cpp
	ts1->add( BOOST_TEST_CASE( &test_diff_float_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_diff_float_2 ) );	
	ts1->add( BOOST_TEST_CASE( &test_diff_double_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_diff_double_2 ) );	
	ts1->add( BOOST_TEST_CASE( &test_diff_complex_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_diff_complex_2 ) );	

	// UnitTest17.cpp
	ts1->add( BOOST_TEST_CASE( &test_dpss_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_submatrix_float_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_submatrix_float_2 ) );	
	ts1->add( BOOST_TEST_CASE( &test_submatrix_double_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_submatrix_double_2 ) );	
	ts1->add( BOOST_TEST_CASE( &test_submatrix_complex_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_submatrix_complex_2 ) );	

	// UnitTest18.cpp
	ts1->add( BOOST_TEST_CASE( &test_QR_float_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_QR_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_QR_float_3 ) );	
	ts1->add( BOOST_TEST_CASE( &test_QR_double_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_QR_double_2 ) );	
	ts1->add( BOOST_TEST_CASE( &test_QR_double_3 ) );	
	ts1->add( BOOST_TEST_CASE( &test_QR_complex_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_QR_complex_2 ) );	
	ts1->add( BOOST_TEST_CASE( &test_QR_complex_3 ) );

	//// UnitTest19.cpp
	ts1->add( BOOST_TEST_CASE( &test_rand_float_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_rand_float_2 ) );	
	ts1->add( BOOST_TEST_CASE( &test_rand_float_3 ) );	
	ts1->add( BOOST_TEST_CASE( &test_rand_float_4 ) );	
	ts1->add( BOOST_TEST_CASE( &test_rand_double_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_rand_double_2 ) );	
	ts1->add( BOOST_TEST_CASE( &test_rand_double_3 ) );	
	ts1->add( BOOST_TEST_CASE( &test_rand_double_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_rand_complex_1 ) );	
	ts1->add( BOOST_TEST_CASE( &test_rand_complex_2 ) );	
	ts1->add( BOOST_TEST_CASE( &test_rand_complex_3 ) );	
	ts1->add( BOOST_TEST_CASE( &test_rand_complex_4 ) );

	// UnitTest20.cpp	
	ts1->add( BOOST_TEST_CASE( &test_max_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_max_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_max_float_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_max_float_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_max_float_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_max_float_6 ) );
	ts1->add( BOOST_TEST_CASE( &test_max_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_max_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_max_double_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_max_double_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_max_double_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_max_double_6 ) );
	ts1->add( BOOST_TEST_CASE( &test_min_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_min_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_min_float_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_min_float_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_min_float_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_min_float_6 ) );
	ts1->add( BOOST_TEST_CASE( &test_min_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_min_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_min_double_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_min_double_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_min_double_5 ) );
	ts1->add( BOOST_TEST_CASE( &test_min_double_6 ) );

	// UnitTest21.cpp	
	ts1->add( BOOST_TEST_CASE( &test_elementwise_multiplication_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_multiplication_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_multiplication_float_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_multiplication_float_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_multiplication_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_multiplication_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_multiplication_double_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_multiplication_double_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_multiplication_complex_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_multiplication_complex_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_multiplication_complex_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_multiplication_complex_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_division_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_division_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_division_float_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_division_float_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_division_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_division_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_division_double_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_division_double_4 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_division_complex_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_division_complex_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_division_complex_3 ) );
	ts1->add( BOOST_TEST_CASE( &test_elementwise_division_complex_4 ) );

	// UnitTest21.cpp	
	ts1->add( BOOST_TEST_CASE( &test_remove_row_int_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_remove_row_int_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_remove_row_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_remove_row_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_remove_row_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_remove_row_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_remove_row_complex_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_remove_row_complex_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_remove_col_int_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_remove_col_int_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_remove_col_float_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_remove_col_float_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_remove_col_double_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_remove_col_double_2 ) );
	ts1->add( BOOST_TEST_CASE( &test_remove_col_complex_1 ) );
	ts1->add( BOOST_TEST_CASE( &test_remove_col_complex_2 ) );

	framework::master_test_suite().add( ts1 );

	return 0;
}