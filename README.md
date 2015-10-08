# matCUDA
A C++ library that aims to facilitate the development of C++ codes that involves generalized vectors and matrix operations with GPU-based computing using CUDA. A very brief overview of its funcionality is shown below.

```
#include "matCUDA.h"
using namespace matCUDA;

void example()
{
	size_t size = 1024;

	// creates Array object (vector) of type std::complex<double> with size random elements
	Array<ComplexDouble> v1 = rand<ComplexDouble>( size );

	// creates Array object (matrix) of type std::complex<double> with size*size random elements
	Array<ComplexDouble> m1 = rand<ComplexDouble>( size, size );

	// multiplies m1 times v1 adn stores in v2
	Array<ComplexDouble> v2 = m1*v1;

	// check if determinant of m1 is different from 0. if so, inverts m1 and stores in m2
	if( m1.determinant() != ComplexDouble(0,0) )
		Array<ComplexDouble> m2 = m1.invert();

	// check is m1 times m2 equals identity matrix
	bool equalIdentity = m1*m2 == eye<ComplexDouble>( size );
}
```

An instance of the Array class represents a vector or matrix (or even arrays of higher degrees), and needed memory is automatically allocated at its creation. Several overloaded operators (as *, +, -, etc) are responsible for the operations between Arrays, as addition, multiplication, etc. Also, there are some algebraic functions implemented, as inversion, LU and QR decomposition, transpose, etc, (almost) every operation implemented to take place on the GPU.
