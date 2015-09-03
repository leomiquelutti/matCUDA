#ifndef UNIT_TESTS_H
#define UNIT_TESTS_H

//#include "../../matCUDA lib/inc/
#include "matCUDA.h"

#include <boost/current_function.hpp>
#include <boost/preprocessor.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

using namespace matCUDA;

typedef ComplexDouble Complex;

template <typename TElement>
void TEST_CALL(Array<TElement> a, Array<TElement> b, char *s )
{		           															          
	if ( !a.check_close( b ) )
		BOOST_FAIL("");
	else									          	
		fprintf(stderr, "%s success\n",s);	
}

template <typename TElement>
void TEST_CALL(TElement a, TElement b, char *s )
{
	Array<TElement> A(1), B(1);
	A(0) = a;
	B(0) = b;
	TEST_CALL( A, B, s );
}

// UnitTest01.cpp
void test_plus_float_1();
void test_plus_float_2();
void test_plus_float_3();
void test_plus_float_4();
void test_plus_float_5();
void test_plus_double_1();
void test_plus_double_2();
void test_plus_double_3();
void test_plus_double_4();
void test_plus_double_5();
void test_plus_complex_1();
void test_plus_complex_2();
void test_plus_complex_3();
void test_plus_complex_4();
void test_plus_complex_5();

// UnitTest02.cpp
void test_plus_equal_float_1();
void test_plus_equal_float_2();
void test_plus_equal_float_3();
void test_plus_equal_float_4();
void test_plus_equal_double_1();
void test_plus_equal_double_2();
void test_plus_equal_double_3();
void test_plus_equal_double_4();
void test_plus_equal_complex_1();
void test_plus_equal_complex_2();
void test_plus_equal_complex_3();
void test_plus_equal_complex_4();

// UnitTest03.cpp
void test_minus_float_1();
void test_minus_float_2();
void test_minus_float_3();
void test_minus_float_4();
void test_minus_double_1();
void test_minus_double_2();
void test_minus_double_3();
void test_minus_double_4();
void test_minus_complex_1();
void test_minus_complex_2();
void test_minus_complex_3();
void test_minus_complex_4();

// UnitTest04.cpp
void test_minus_equal_float_1();
void test_minus_equal_float_2();
void test_minus_equal_float_3();
void test_minus_equal_float_4();
void test_minus_equal_double_1();
void test_minus_equal_double_2();
void test_minus_equal_double_3();
void test_minus_equal_double_4();
void test_minus_equal_complex_1();
void test_minus_equal_complex_2();
void test_minus_equal_complex_3();
void test_minus_equal_complex_4();

// UnitTest05.cpp
void call_to_array();
void test_transpose_float_1();
void test_transpose_float_2();
void test_transpose_float_3();
void test_transpose_float_4();
void test_transpose_float_5();
void test_transpose_float_6();
void test_conjugate_float_1();
void test_conjugate_float_2();
void test_conjugate_float_3();
void test_conjugate_float_4();
void test_conjugate_float_5();
void test_conjugate_float_6();
void test_hermitian_float_1();
void test_hermitian_float_2();
void test_hermitian_float_3();
void test_hermitian_float_4();
void test_hermitian_float_5();
void test_hermitian_float_6();
void test_transpose_double_1();
void test_transpose_double_2();
void test_transpose_double_3();
void test_transpose_double_4();
void test_transpose_double_5();
void test_transpose_double_6();
void test_conjugate_double_1();
void test_conjugate_double_2();
void test_conjugate_double_3();
void test_conjugate_double_4();
void test_conjugate_double_5();
void test_conjugate_double_6();
void test_hermitian_double_1();
void test_hermitian_double_2();
void test_hermitian_double_3();
void test_hermitian_double_4();
void test_hermitian_double_5();
void test_hermitian_double_6();
void test_transpose_complex_1();
void test_transpose_complex_2();
void test_transpose_complex_3();
void test_transpose_complex_4();
void test_transpose_complex_5();
void test_transpose_complex_6();
void test_conjugate_complex_1();
void test_conjugate_complex_2();
void test_conjugate_complex_3();
void test_conjugate_complex_4();
void test_conjugate_complex_5();
void test_conjugate_complex_6();
void test_hermitian_complex_1();
void test_hermitian_complex_2();
void test_hermitian_complex_3();
void test_hermitian_complex_4();
void test_hermitian_complex_5();
void test_hermitian_complex_6();

// UnitTest06.cpp
void test_times_float_1();
void test_times_float_2();
void test_times_float_3();
void test_times_float_4();
void test_times_float_5();
void test_times_float_6();
void test_times_float_7();
void test_times_float_8();
void test_times_float_9();
void test_times_float_10();
void test_times_float_11();
void test_times_float_12();
void test_times_float_13();
void test_times_float_14();
void test_times_float_15();
void test_times_float_16();
void test_times_double_1();
void test_times_double_2();
void test_times_double_3();
void test_times_double_4();
void test_times_double_5();
void test_times_double_6();
void test_times_double_7();
void test_times_double_8();
void test_times_double_9();
void test_times_double_10();
void test_times_double_11();
void test_times_double_12();
void test_times_double_13();
void test_times_double_14();
void test_times_complex_1();
void test_times_complex_2();
void test_times_complex_3();
void test_times_complex_4();
void test_times_complex_5();
void test_times_complex_6();
void test_times_complex_7();
void test_times_complex_8();
void test_times_complex_9();
void test_times_complex_10();
void test_times_complex_11();
void test_times_complex_12();
void test_times_complex_13();

// UnitTest07.cpp
void test_times_equal_float_1();
void test_times_equal_float_2();
void test_times_equal_float_3();
void test_times_equal_float_4();
void test_times_equal_double_1();
void test_times_equal_double_2();
void test_times_equal_double_3();
void test_times_equal_double_4();
void test_times_equal_complex_1();
void test_times_equal_complex_2();
void test_times_equal_complex_3();
void test_times_equal_complex_4();

// UnitTest08.cpp
void test_determinant_float_1x1_1();
void test_determinant_float_1x1_2();
void test_determinant_float_2x2_1();
void test_determinant_float_2x2_2();
void test_determinant_float_2x2_3();
void test_determinant_float_3x3_1();
void test_determinant_float_3x3_2();
void test_determinant_float_20x20_1();
void test_determinant_float_20x20_2();
void test_determinant_float_100x100_1();
void test_determinant_double_1x1_1();
void test_determinant_double_1x1_2();
void test_determinant_double_2x2_1();
void test_determinant_double_2x2_2();
void test_determinant_double_3x3_1();
void test_determinant_double_3x3_2();
void test_determinant_double_20x20_1();
void test_determinant_double_20x20_2();
void test_determinant_double_100x100_1();
void test_determinant_complex_1x1_1();
void test_determinant_complex_1x1_2();
void test_determinant_complex_2x2_1();
void test_determinant_complex_2x2_2();
void test_determinant_complex_3x3_1();
void test_determinant_complex_3x3_2();
void test_determinant_complex_4x4_1();
void test_determinant_complex_4x4_2();
void test_determinant_complex_5x5_1();
void test_determinant_complex_20x20_1();
void test_determinant_complex_20x20_2();
void test_determinant_complex_100x100_1();

// UnitTest09.cpp
void test_invert_float_1x1_1();
void test_invert_float_1x1_2();
void test_invert_float_2x2_1();
void test_invert_float_2x2_2();
void test_invert_float_2x2_3();
void test_invert_float_3x3_1();
void test_invert_float_3x3_2();
void test_invert_float_3x3_3();
void test_invert_float_20x20_1();
void test_invert_float_20x20_2();
void test_invert_double_1x1_1();
void test_invert_double_1x1_2();
void test_invert_double_2x2_1();
void test_invert_double_2x2_2();
void test_invert_double_2x2_3();
void test_invert_double_3x3_1();
void test_invert_double_3x3_2();
void test_invert_double_3x3_3();
void test_invert_double_20x20_1();
void test_invert_double_20x20_2();
void test_invert_complex_1x1_1();
void test_invert_complex_1x1_2();
void test_invert_complex_2x2_1();
void test_invert_complex_2x2_2();
void test_invert_complex_2x2_3();
void test_invert_complex_3x3_1();
void test_invert_complex_3x3_2();
void test_invert_complex_3x3_3();
void test_invert_complex_20x20_1();
void test_invert_complex_20x20_2();
void test_invert_complex_20x20_3();
void test_invert_complex_100x100_1();

// UnitTest10.cpp
void test_minor_float_1();
void test_minor_float_2();
void test_minor_double_1();
void test_minor_double_2();
void test_minor_complex_1();
void test_minor_complex_2();
void test_norm_float_1();
void test_norm_double_1();
void test_norm_complex_1();
void test_eye_float_1();
void test_eye_float_2();
void test_eye_double_1();
void test_eye_double_2();
void test_eye_complex_1();
void test_eye_complex_2();
void test_times_TElement_int();
void test_times_TElement_float();
void test_times_TElement_double();
void test_times_TElement_complex();
void test_divided_TElement_float();
void test_divided_TElement_double();
void test_divided_TElement_complex();
void test_add_TElement_float();
void test_add_TElement_double();
void test_add_TElement_complex();
void test_minus_TElement_float();
void test_minus_TElement_double();
void test_minus_TElement_complex();
void test_equal_TElement_float();
void test_equal_TElement_double();
void test_equal_TElement_complex();

// UnitTest11.cpp
void test_LU_float_1();
void test_LU_float_2();
void test_LU_double_1();
void test_LU_double_2();
void test_LU_complex_1();
void test_LU_complex_2();

// UnitTest12.cpp
void test_sin_float_1();
void test_sind_float_1();
void test_sin_double_1();
void test_sind_double_1();
void test_cos_float_1();
void test_cosd_float_1();
void test_cos_double_1();
void test_cosd_double_1();
void test_tan_float_1();
void test_tand_float_1();
void test_tan_double_1();
void test_tand_double_1();
void test_asin_float_1();
void test_asind_float_1();
void test_asin_double_1();
void test_asind_double_1();
void test_acos_float_1();
void test_acosd_float_1();
void test_acos_double_1();
void test_acosd_double_1();
void test_atan_float_1();
void test_atand_float_1();
void test_atan_double_1();
void test_atand_double_1();

// UnitTest13.cpp
void test_FFT_C2C_1();
void test_FFT_C2C_2();
void test_FFT_C2C_3();
void test_FFT_R2C_1();
void test_FFT_R2C_2();
void test_FFT_R2C_3();
void test_FFT_D2Z_1();
void test_FFT_D2Z_2();
void test_FFT_D2Z_3();

// UnitTest14.cpp
void test_LS_float_1();
void test_LS_float_2();
void test_LS_double_1();
void test_LS_double_2();
void test_LS_complex_1();
void test_LS_complex_2();
void test_LS_complex_3();

// UnitTest15.cpp
void test_detrend_float_1();
void test_detrend_float_2();
void test_detrend_float_3();
void test_detrend_float_4();
void test_detrend_double_1();
void test_detrend_double_2();
void test_detrend_double_3();
void test_detrend_double_4();

// UnitTest16.cpp
void test_diff_float_1();
void test_diff_float_2();
void test_diff_double_1();
void test_diff_double_2();
void test_diff_complex_1();
void test_diff_complex_2();

// UnitTest17.cpp
void test_dpss_float_1();
void test_dpss_float_2();
void test_dpss_double_1();
void test_dpss_double_2();
void test_submatrix_float_1();
void test_submatrix_float_2();
void test_submatrix_double_1();
void test_submatrix_double_2();
void test_submatrix_complex_1();
void test_submatrix_complex_2();

// UnitTest18.cpp
void test_QR_float_1();
void test_QR_float_2();
void test_QR_float_3();
void test_QR_double_1();
void test_QR_double_2();
void test_QR_double_3();
void test_QR_complex_1();
void test_QR_complex_2();
void test_QR_complex_3();

// UnitTest19.cpp
void test_rand_float_1();
void test_rand_float_2();
void test_rand_double_1();
void test_rand_double_2();
void test_rand_complex_1();
void test_rand_complex_2();

// UnitTest20.cpp
void test_max_float_1();
void test_max_float_2();
void test_max_float_3();
void test_max_float_4();
void test_max_float_5();
void test_max_float_6();
void test_max_double_1();
void test_max_double_2();
void test_max_double_3();
void test_max_double_4();
void test_max_double_5();
void test_max_double_6();
void test_min_float_1();
void test_min_float_2();
void test_min_float_3();
void test_min_float_4();
void test_min_float_5();
void test_min_float_6();
void test_min_double_1();
void test_min_double_2();
void test_min_double_3();
void test_min_double_4();
void test_min_double_5();
void test_min_double_6();

// UnitTestDraft.cpp
void draft();

#endif