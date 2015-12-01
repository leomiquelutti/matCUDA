#include "unitTests.h"

void draft()
{	
	// here you do your stuff

	size_t size = 5e3, nIter = 10;

	Array<double> a = rand<double>( size, size );
	Array<double> b = rand<double>( size, size );

	b = a.elementWiseMultiply( &b );
}


//void draft()
//{	
//	// here you do your stuff
//
//	size_t size = 5e3, nIter = 10;
//
//	/////* test 01 - rand *//////
//	//Array<ComplexDouble> m1 = rand<ComplexDouble>( size, size );
//
//	/////* test 02 - add *//////
//	//Array<ComplexDouble> m1 = rand<ComplexDouble>( size, size );
//	//Array<ComplexDouble> m2 = rand<ComplexDouble>( size, size );
//	//Array<ComplexDouble> m3( size, size );
//	//m3 = m2 + m1;	
//
//	/////* test 03 - times *//////
//	//Array<ComplexDouble> m1 = rand<ComplexDouble>( size, size );
//	//Array<ComplexDouble> m2 = rand<ComplexDouble>( size, size );
//	//Array<ComplexDouble> m3( size, size );
//	//m3 = m2*m1;	
//
//	/////* test 04 - lu *//////
//	//Array<ComplexDouble> m = rand<ComplexDouble>( size, size );
//	//Array<ComplexDouble> l( size, size );
//	//Array<ComplexDouble> u( size, size );
//	//m.lu( &l, &u );	
//
//	/////* test 05 - fft *//////
//	//Array<ComplexDouble> v = rand<ComplexDouble>( pow(2,26) );
//	//Array<ComplexDouble> V( pow(2,26) );
//	// V = v.fft();	
//
//	/////* test 06 - ls *//////
//	//size_t size1 = 1e4, size2 = 1e2, size3 = 1e2;
//	//Array<ComplexDouble> a = rand<ComplexDouble>( size1, size2 );
//	//Array<ComplexDouble> x = rand<ComplexDouble>( size1, size3 );
//	//Array<ComplexDouble> p( size2, size3 );
//	//p = x.ls( &a );
//
//	/////* test 07 - inv *//////
//	//Array<ComplexDouble> m = rand<ComplexDouble>( size, size );
//	//Array<ComplexDouble> m_inv( size, size );
//	//m_inv = m.invert();
//}

//#include "matCUDA.h"
//using namespace matCUDA;
//
//typedef std::complex<double> ComplexDouble;
//
//void example()
//{ 
//	size_t N = 1024;
//
//	// creates complex-vector with N random elements
//	Array<ComplexDouble> v1 = rand<ComplexDouble>( N );
//
//	// creates complex-matrix with N x N elements
//	Array<ComplexDouble> m1( N, N ), m2( N, N );
//
//	// fills in m1 and m2 with random numbers
//	m1 = rand<ComplexDouble>( N, N );
//	
//	// multiplies m1 times v1 and stores result in newly declarated v2 
//	Array<ComplexDouble> v2 = m1*v1; 
//
//	// compares if v1 and v2 are equal element-wise and returns true of false 
//	bool trueOrFalse = v1 == v2; 
//	
//	// multiplies m1 times hermitian of m1 and stores the result in m2
//	m2 = m1*m1.hermitian();
//
//	// adds v1 to vector m2*v2 and stores in v1
//	// equivalent to v1 = v1+ m2*v2
//	v1 += m2*v2;
//
//	// multiplies m2 times the inverse of m1 and stores the result in m1 
//	// equivalente to m1 = m2*m1.inverse();
//	m1 = m2/m1;
//     
//	// multiplies all elements of m1 times 4, adds to m2 and then
//	// subtracts 5 of all elements and stores the result in m2
//	m2 = m1*4 + m2 - 5; 
//	
//	// evaluates m2 QR decomposition
//	Array<ComplexDouble> q( N, N );
//	Array<ComplexDouble> r( N, N );
//	m2.qr( &q, &r );
//
//	// solves linear problems represented by Ap = x by least square method 
//	// in this example, A -> m1; p -> v1; x -> v2
//	v1 = v2.ls( &m1 );
//
//	// print v1 on screen
//	v1.print();
//}