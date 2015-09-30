#include "unitTests.h"

#include "matCUDA.h"
using namespace matCUDA;

void draft()
{
	size_t size = 2;
	Array<double> a( size, size );
	a( 0, 0 ) = 4;
	a( 0, 1 ) = 3;
	a( 1, 0 ) = 6;
	a( 1, 1 ) = 3;

	Array<double> l( size, size );
	Array<double> u( size, size );
	Array<double> p( size, size );

	a.print();
	a.LU( &l, &u, &p );
	a.print();
	l.print();
	u.print();
	p.print();
	
	(a*p).print();
	(l*u).print();
	//size_t size = pow(2,26), nIter = 10;

	//Array<ComplexFloat> v = rand<ComplexFloat>( size );
	//Array<ComplexFloat> V( size );
	//
	//long double time = 0;
	//tic();
	//for( int i = 0; i < nIter; i++ )
	//	V = v.fft();
	//toc();
}