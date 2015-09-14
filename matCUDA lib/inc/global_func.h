

<<<<<<< HEAD
	template <typename TElement>
	Array<TElement> dpss( index_t N, double NW, index_t degree );

	Array<ComplexFloat> fft( Array<float> *in );
	Array<ComplexDouble> fft( Array<double> *in );

	template <typename TElement>
	Array<TElement> read_file_vector( std::string s );

	template <typename TElement>
	Array<TElement> read_file_matrix( std::string s );

	template <typename TElement>
	Array<TElement> rand(index_t u);

	template <typename TElement>
	Array<TElement> rand(index_t u1, index_t u2);

	// measure time and print
	void tic();
	void toc();
	long double toc( long double in );

	Array<ComplexFloat> Array_S2C( Array<float> ); // TODO
	Array<ComplexDouble> Array_D2Z( Array<double> ); // TODO
	//Array<TElement> transpose( Array<TElement> ); // TODO
	//Array<TElement> conj( Array<TElement> ); // TODO
	//Array<TElement> hermitian( Array<TElement> ); // TODO
	//Array<TElement> invert( Array<TElement> ); // TODO
	//TElement determinant( Array<TElement> ); // TODO
}
=======
>>>>>>> origin/master
