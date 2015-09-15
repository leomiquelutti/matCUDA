#include "matCUDA.h"

using namespace matCUDA;

int main()
{
	size_t size = 4;

	// creates ComplexDouble-type Array object 
	// with size random elements - vector
	Array<ComplexDouble> v1 = rand<ComplexDouble>( size );

	// creates ComplexDouble-type Array object
	// with size x size random elements - matrix
	Array<ComplexDouble> m1 = rand<ComplexDouble>( size, size );

	// multiplies m1 times v1 adn stores in v2
	Array<ComplexDouble> v2 = m1*v1;

	// creates m2 with dimensions of m1
	Array<ComplexDouble> m2( m1.getDim(0), m1.getDim(1) );

	// check if determinant of m1 is different from 0. if so, inverts m1 and stores in m2
	if( m1.determinant() != ComplexDouble(0,0) )
		m2 = m1.invert();

	int a;
	Array<ComplexDouble> m = m1*m2;
	m.print();

	Array<ComplexDouble> b = eye<ComplexDouble>( size );
	b.print();

	if( m == b )
		a = 2;
	else
		a = 3;
	
	// check is m1 times m2 equals identity matrix
	bool equalIdentity = m == eye<ComplexDouble>( size );
	m.print();

	m.invert().print();

	m.print();

	return 0;
}