#ifndef FFT_h
#define FFT_h

#include <cmath>
#include <complex>

extern void fft(int *x_in, 
	std::complex<double> *x_out,
	int N);
void fft_rec(std::complex<double> *x, int N);

#endif