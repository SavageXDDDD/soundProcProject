#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include "AudioFile.h"
#include <iostream>
#include <vector>
#include <complex>
#include <corecrt_math_defines.h>
#include "RtAudio.h"
#include <chrono>
#include "rtaudio_c.h"
#define _USE_MATH_DEFINES
using namespace std;
#define REAL 0
#define IMAG 1


std::vector<std::complex<double> > fast_fourier_transform(std::vector<std::complex<double> > x, bool inverse = false) {
    std::vector<std::complex<double> > w(x.size(), 0.0);  // w[k] = e^{-i2\pi k/N}
    // Precalculate w[k] for faster FFT computation, do it in a 'binary-search' way to provide decent numeric accuracy
    w[0] = 1.0;
    for (int pow_2 = 1; pow_2 < (int)x.size(); pow_2 *= 2) {
        w[pow_2] = std::polar(1.0, 2 * M_PI * pow_2 / x.size() * (inverse ? 1 : -1));
    }
    for (int i = 3, last = 2; i < (int)x.size(); i++) {
        // This way of computing w[k] guarantees that each w[k] is computed in at most log_2 N multiplications
        if (w[i] == 0.0) {
            w[i] = w[last] * w[i - last];
        }
        else {
            last = i;
        }
    }

    for (int block_size = x.size(); block_size > 1; block_size /= 2) {
        // Do the rearrangement for 'recursive' call block by block for each level'
        std::vector<std::complex<double> > new_x(x.size());

        for (int start = 0; start < (int)x.size(); start += block_size) {
            for (int i = 0; i < block_size; i++) {
                new_x[start + block_size / 2 * (i % 2) + i / 2] = x[start + i];
            }
        }
        x = new_x;
    }

    for (int block_size = 2; block_size <= (int)x.size(); block_size *= 2) {
        // Now compute the FFT 'recursively' level by level bottom-up
        std::vector<std::complex<double> > new_x(x.size());
        int w_base_i = x.size() / block_size;  // w[w_base_i] is the e^{-i2\pi / N} value for the N of this level

        for (int start = 0; start < (int)x.size(); start += block_size) {
            for (int i = 0; i < block_size / 2; i++) {
                new_x[start + i] = x[start + i] + w[w_base_i * i] * x[start + block_size / 2 + i];
                new_x[start + block_size / 2 + i] = x[start + i] - w[w_base_i * i] * x[start + block_size / 2 + i];
            }
        }
        x = new_x;
    }
    return x;
}

vector<double> hann(int N) {
    vector<double> h;
    for (int n = 0; n < N; n++) {
        h.push_back(pow(sin(M_PI * n / N), 2));
    }
    return h;
}

vector<double> hamming(int N) {
    vector<double> h;
    double max = 0;
    for (int n = 0; n < N; n++) {
        h.push_back(0.54+0.46*cos((2 * M_PI * n)/(N-1)));
        if (0.54 + 0.46 * cos((2 * M_PI * n) / (N - 1)) > max) max = 0.54 + 0.46 * cos((2 * M_PI * n) / (N - 1));
    }
    for (int n = 0; n < N; n++) {
        h.at(n) = h.at(n) / max;
    }
    return h;
}

vector<double> sinWin(int N) {
    vector<double> h;
    vector<double> hannWin = hann(N);
    vector<double> hamWin = hamming(N);
    for (int n = 0; n < N; n++) {
        h.push_back(hannWin.at(n)/hamWin.at(n));
    }
    return h;
}

vector<vector<double>> analyse(vector<double> X, int frameSize, float overlap) {
    vector<vector<double>> y{};
    if (X.size() == 0) {
        cout << "empty vector passed, empty vector returned" << endl;
        return y;
    }
    if (overlap >= 1 || overlap <0) {
        cout << "bad overlap value, must be between 0 and 0.(9), empty vector returned" << endl;
        return y;
    }
    if (frameSize <= 0) {
        cout << "bad frame size, must be greater than 0, empty vector returned" << endl;
        return y;
    }
    int nFrames = 1 + (X.size() - frameSize) / (frameSize * (1 - overlap));
    vector<double> hannWindow = hann(frameSize);
    //cout << "nFrames=" << nFrames << endl;
    y.resize(nFrames);
    int j1 = 0;
    int iStart = 0; int iEnd = frameSize;
    for (int i = 0; i < nFrames; i++) {
        int k = 0;
        for (int j = iStart; j < iEnd; j++) {
            y.at(i).push_back(X.at(j)*hannWindow.at(k));
            k++;
        }
        iStart += frameSize * (1 - overlap);
        iEnd += frameSize * (1 - overlap);
        //cout << i << endl;
    }
    return y;
}

vector<double> synthesise(vector<vector<double>> X, int frameSize, float overlap) {
    vector<double> y{};
    if (X.size() == 0) {
        cout << "empty vector passed, empty vector returned" << endl;
        return y;
    }
    if (overlap >= 1 || overlap < 0) {
        cout << "bad overlap value, must be between 0 and 0.(9), empty vector returned" << endl;
        return y;
    }
    if (frameSize <= 0) {
        cout << "bad frame size, must be greater than 0, empty vector returned" << endl;
        return y;
    }
    int iStart = frameSize*(1-overlap)-1; int iEnd = frameSize;
    cout << y.size() << endl;
    for (int i = 0; i < frameSize; i++) {
        //cout << i << endl;
        //cout << X.size() << " || " << X.at(0).size() << endl;
        y.push_back(X.at(0).at(i));
    }
    cout << y.size() << endl;
    for (int i = 1; i < X.size(); i++) {
        for (int j = iStart; j < iEnd; j++) {
            y.at(j) += X.at(i).at(j%frameSize);
        }
        for (int j = 0; j < frameSize * (1 - overlap) -1; j++) {
            y.push_back(X.at(i).at(j));
        }
        iStart += frameSize * (1 - overlap);
        iEnd += frameSize * (1 - overlap);
        //cout << i << endl;
        //cout << y.size() << " || " << iStart << " || " << iEnd << endl;
    }
    //cout << y.size();
    return y;
}

vector<double> umodav(vector<double> noise, vector<double> signal, int frameSize, float overlap) {
    vector<double> y{};
    if (noise.size() == 0) {
        cout << "empty noise vector passed, empty vector returned" << endl;
        return y;
    }
    if (signal.size() == 0) {
        cout << "empty signal vector passed, empty vector returned" << endl;
        return y;
    }
    if (noise.size() != signal.size()) {
        cout << "noise must be same length as signal, empty vector returned" << endl;
        return y;
    }
    if (overlap >= 1 || overlap < 0) {
        cout << "bad overlap value, must be between 0 and 0.(9), empty vector returned" << endl;
        return y;
    }
    if (frameSize <= 0) {
        cout << "bad frame size, must be greater than 0, empty vector returned" << endl;
        return y;
    }
    vector<vector<double>> splitNoise = analyse(noise, frameSize, overlap);
    vector<vector<double>> splitSignal = analyse(signal, frameSize, overlap);
    vector<vector<complex<double>>> splitNoiseFFT{};
    vector<vector<complex<double>>> splitSignalFFT{};
    vector<complex<double>> buf(splitNoise.at(0).size());
    vector<vector<double>> Ps(splitNoise.size());
    double V;
    cout << splitNoise.size() <<endl;
    cout << Ps.size() << endl;
    for (int i = 0; i < splitNoise.size(); i++) {
        if (splitNoise.at(i).size() % 2 != 0)
            splitNoise.at(i).resize(pow(2, ((int)(log2(splitNoise.at(i).size()))) + 1));
        buf.clear();
        buf.resize(splitNoise.at(0).size());
        transform(splitNoise.at(i).begin(), splitNoise.at(i).end(), buf.begin(), [](double da) {
        return std::complex<double>(da, 0); });
        splitNoiseFFT.push_back(fast_fourier_transform(buf));
        buf.clear();
        buf.resize(splitNoise.at(0).size());
        transform(splitSignal.at(i).begin(), splitSignal.at(i).end(), buf.begin(), [](double da) {
            return std::complex<double>(da, 0); });
        splitSignalFFT.push_back(fast_fourier_transform(buf));
    };
    for (int i = 0; i < splitNoiseFFT.size(); i++) {
        for (int j = 0; j < splitNoiseFFT.at(i).size(); j++) {
            //cout << i << " || " << j << endl;
            V = pow(abs(splitSignalFFT.at(i).at(j)), 2) - pow(abs(splitNoiseFFT.at(i).at(j)), 2) ;
            V > 0 ? Ps.at(i).push_back(V) : Ps.at(i).push_back(0.0);
        }
    }
    vector<double> sinthWin = sinWin(splitNoise.size());
    vector<complex<double>> IFFTbuf{};
    vector<vector<double>> absIFFT(splitNoise.size());
    for (int i = 0; i < splitNoise.size(); i++) {
        buf.clear();
        buf.resize(splitNoise.at(0).size());
        transform(Ps.at(i).begin(), Ps.at(i).end(), buf.begin(), [](double da) {
            return std::complex<double>(da, 0); });
        for (int j = 0; j < Ps.at(0).size(); j++) {
            buf.at(j) = sqrt(abs(buf.at(j))) * exp(1i*arg(splitSignalFFT.at(i).at(j)));
        }
        IFFTbuf = fast_fourier_transform(buf, true);
        //cout << IFFTbuf.size() << endl;
        for (int j = 0; j < splitNoise.at(i).size(); j++) {
            absIFFT.at(i).push_back(real(IFFTbuf.at(j) * sinthWin.at(j)));
        }
    };
    y = synthesise(absIFFT, frameSize, overlap);
    return y;
}

int main() {
    auto begin = std::chrono::high_resolution_clock::now();
    string filePath = "C:\\test\\signal_noise101.wav";
    string noisePath = "C:\\test\\noise101.wav";
    AudioFile<double> audioFile;
    AudioFile<double> noiseFile;
    vector<vector<double>> y { {},{} };
    vector<double> in = {};
    vector<double> noise = {};
    //ofstream file("C:\\Users\\Lenovo\\source\\repos\\soundProcProject\\FFT.txt");
    try{
        audioFile.load(filePath);
        noiseFile.load(noisePath);
        audioFile.printSummary();
    }
    catch (exception e) {
        cout << "failed to read audio file "<< filePath << endl;
        return 696969;
    }
    for (int i = 0; i < noiseFile.getNumSamplesPerChannel(); i++)
    {
        in.push_back(audioFile.samples[0][i]);
        noise.push_back(noiseFile.samples[0][i]);
    }
    
    vector<double> z1 = umodav(noise, in, 257, 0.5);
    cout << z1.size() << endl;
    //cout << z.size() << endl;
    //audioFile.save("C:\\test\\морковка1.wav");
    for (int i = 0; i < z1.size(); i++)
    {
        audioFile.samples[0][i] = z1.at(i);
        //audioFile.samples[1][i] = z1.at(i);
    }
    audioFile.save("C:\\test\\rst.wav");
    auto end = std::chrono::high_resolution_clock::now();
    auto eslaped = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    cout << "microseconds " << eslaped.count() << endl;
    /*
    std::vector<std::complex<double> > x(in.size());
    std::transform(in.begin(), in.end(), in.begin(), x.begin(), [](double da, double db) {
        return std::complex<double>(da, 0); });
    cout << x.size() << endl;
    if((int)log2(x.size()) != log2(x.size())) x.resize(pow(2, ((int)(log2(x.size())))+1));
    cout << x.size() << endl;
    int N = in.size();
    std::vector<std::complex<double> > result = fast_fourier_transform(x);
    for (int i = 0; i < 10; i++) {
        cout << "fft(" << i << ") = " << result.at(i) << endl;
    }
    */
    return 0;
}
