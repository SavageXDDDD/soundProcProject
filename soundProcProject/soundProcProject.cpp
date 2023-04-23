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


vector<complex<double> > fast_fourier_transform(vector<complex<double> > x, bool inverse = false) {
    vector<complex<double> > w(x.size(), 0.0);  // w[k] = e^{-i2\pi k/N}
    // Precalculate w[k] for faster FFT computation, do it in a 'binary-search' way to provide decent numeric accuracy
    w[0] = 1.0;
    for (int pow_2 = 1; pow_2 < (int)x.size(); pow_2 *= 2) {
        w[pow_2] = polar(1.0, 2 * M_PI * pow_2 / x.size() * (inverse ? 1 : -1));
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
        vector<complex<double> > new_x(x.size());

        for (int start = 0; start < (int)x.size(); start += block_size) {
            for (int i = 0; i < block_size; i++) {
                new_x[start + block_size / 2 * (i % 2) + i / 2] = x[start + i];
            }
        }
        x = new_x;
    }

    for (int block_size = 2; block_size <= (int)x.size(); block_size *= 2) {
        // Now compute the FFT 'recursively' level by level bottom-up
        vector<complex<double> > new_x(x.size());
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
    for (int n = 1; n < N+1; n++) {
        h.push_back(pow(sin(M_PI * n / N), 2));
    }
    return h;
}

vector<double> hamming(int N) {
    vector<double> h;
    for (int n = 1; n < N+1; n++) {
        h.push_back(0.54-0.46*cos((2 * M_PI * n)/(N-1)));
    }
    return h;
}

vector<double> sinWin(int N) {
    vector<double> h;
    vector<double> hannWin = hann(N);
    vector<double> hamWin = hamming(N);
    for (int n = 0; n < N; n++) {
        h.push_back(hannWin.at(n) / hamWin.at(n));
    }
    return h;
}

vector<vector<complex<double>>> analyse(vector<double> X, int frameSize, float overlap) {
    vector<vector<double>> y{};
    int nFrames = 1 + floor((X.size() - frameSize) / (frameSize * (1 - overlap)));
    vector<vector<complex<double>>> result{};
    vector<vector<complex<double>>> result1{};
    if (X.size() == 0) {
        cout << "empty vector passed, empty vector returned" << endl;
        return result;
    }
    if (overlap >= 1 || overlap <0) {
        cout << "bad overlap value, must be between 0 and 0.(9), empty vector returned" << endl;
        return result;
    }
    if (frameSize <= 0) {
        cout << "bad frame size, must be greater than 0, empty vector returned" << endl;
        return result;
    }
    vector<double> hammingWindow = hamming(frameSize);
    y.resize(nFrames);
    int iStart = 0; int iEnd = frameSize;
    for (int i = 0; i < nFrames; i++) {
        int k = 0;
        for (int j = iStart; j < iEnd; j++) {
            y.at(i).push_back(X.at(j)*hammingWindow.at(k));
            k++;
        }
        iStart += frameSize * (1 - overlap);
        iEnd += frameSize * (1 - overlap);
    }
    vector<complex<double>> buf;
    vector<complex<double>> buf1;
    for (int i = 0; i < nFrames; i++) {
        buf.clear();
        buf.resize(frameSize);
        transform(y.at(i).begin(), y.at(i).end(), buf.begin(), [](double da) {
            return complex<double>(da, 0.0); });
        buf1 = fast_fourier_transform(buf);
        result.push_back(buf1);
    }
    result1 = result;
    return result1;
}

vector<double> synthesise(vector<vector<complex<double>>> X, int frameSize, float overlap) {
    vector<double> y;
    vector<double> outBuf(frameSize);
    vector<double> result;
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
    vector<complex<double>> buf;
    vector<complex<double>> buf1;
    vector<double> Ws = sinWin(frameSize);
    int hsize = floor(frameSize * (1 - overlap));
    vector<vector<complex<double>>> ifftFrames;
    //ifftFrames.reserve(X.size());
    y.resize((X.size()+1) * overlap * frameSize);
    for (int i = 0; i < X.size(); i++) {
        buf.clear(); 
        buf = X.at(i);
        buf.resize(hsize+1);
        for (int j = hsize-1; j > 0; j--) buf.push_back(conj(buf.at(j)));
        buf1 = fast_fourier_transform(buf, true);
        for (int j = 0; j < frameSize; j++) buf1.at(j) *= Ws.at(j)/frameSize;
        ifftFrames.push_back(buf1);
    }
    //for (int i = 0; i < ifftFrames.at(0).size(); i++) cout << ifftFrames.at(0).at(i) << endl;
    //fill(buf.begin(), buf.end(), 0.0);
    int iStart = 0;
    for (int i = 0; i < X.size(); i++) {
        for (int j = 0; j < ifftFrames.at(i).size(); j++) {
            outBuf.at(j) += real(ifftFrames.at(i).at(j));
            y.at(iStart + j) = outBuf.at(j);
        }
        for (int j = 0; j < floor(outBuf.size() / 2); j++) {
            outBuf.at(j) = outBuf.at(j + floor(buf.size() / 2));
        }
        iStart += hsize;
        fill(outBuf.begin() + hsize, outBuf.end(), 0.0);
    }
    //for (int i = 54200; i < 54260; i++) cout << y.at(i) << endl;
    result = y;
    return result;
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
    vector<vector<complex<double>>> splitNoiseFFT = analyse(noise, frameSize, overlap);
    vector<vector<complex<double>>> splitSignalFFT = analyse(signal, frameSize, overlap);
    vector<complex<double>> buf(splitNoiseFFT.at(0).size());
    vector<vector<double>> Ps(splitNoiseFFT.size());
    int hsize = floor(frameSize * (1 - overlap));
    double Pd;
    double Px;
    double V;
    double alpha = 50.0;
    double beta = 0.0;
    double gama = 0.05;
    
    for (int i = 0; i < splitNoiseFFT.size(); i++) {
        Pd = 0.0;
        Px = 0.0;
        /*
        for (int j = 0; j < splitNoiseFFT.at(i).size(); j++) {
            Pd += pow(abs(splitNoiseFFT.at(i).at(j)), 2);
            Px += pow(abs(splitSignalFFT.at(i).at(j)), 2);
        }
        Pd = Pd / splitNoiseFFT.at(i).size();
        */
        for (int j = 0; j < splitNoiseFFT.at(i).size(); j++) {
            V = pow(abs(splitSignalFFT.at(i).at(j)), 2) - pow(abs(splitNoiseFFT.at(i).at(j)),2) * alpha;
            V > beta * pow(abs(splitNoiseFFT.at(i).at(j)), 2) ? Ps.at(i).push_back(V) : Ps.at(i).push_back(beta * pow(abs(splitNoiseFFT.at(i).at(j)), 2));
        }
    }
    vector<vector<complex<double>>> IFFTbuf(splitNoiseFFT.size());
    for (int i = 0; i < splitNoiseFFT.size(); i++) {
        buf.clear();
        for (int j = 0; j < hsize+1; j++) {
            buf.push_back(sqrt(Ps.at(i).at(j)) * exp(1i * arg(splitSignalFFT.at(i).at(j))));
        }
        for (int j = hsize-1; j > 0; j--) {
            buf.push_back(conj(buf.at(j)));
        }
        IFFTbuf.push_back(buf);
    };
    y = synthesise(IFFTbuf, frameSize, overlap);
    return y;
}

int main() {
    auto begin = chrono::high_resolution_clock::now();
    string filePath = "C:\\test\\signal_noise101.wav";
    string noisePath = "C:\\test\\noise101.wav";
    AudioFile<double> audioFile;
    AudioFile<double> noiseFile;
    vector<vector<double>> y { {},{} };
    vector<double> in = {};
    vector<double> noise = {};
    try{
        audioFile.load(filePath);
        noiseFile.load(noisePath);
        audioFile.printSummary();
    }
    catch (exception e) {
        cout << "failed to read audio file "<< filePath << endl;
        return 696969;
    }
    for (int i = 0; i < audioFile.getNumSamplesPerChannel(); i++)
    {
        in.push_back(audioFile.samples[0][i]);
        noise.push_back(noiseFile.samples[0][i]);
    }
    
    //vector<double> z1 = umodav(noise, in, 512, 0.5);
    vector<vector<complex<double>>> a = analyse(in, 512, 0.5);
    vector<double> z1 = synthesise(a, 512, 0.5);
    if (audioFile.getNumSamplesPerChannel() < z1.size()) {
        for (int i = 0; i < audioFile.getNumSamplesPerChannel(); i++)
        {
            audioFile.samples[0][i] = z1.at(i);
            //audioFile.samples[1][i] = z1.at(i);
        }
    }
    else {
        for (int i = 0; i < z1.size(); i++)
        {
            audioFile.samples[0][i] = z1.at(i);
            //audioFile.samples[1][i] = z1.at(i);
        }
    }
    audioFile.save("C:\\test\\rst.wav");
    auto end = chrono::high_resolution_clock::now();
    auto eslaped = chrono::duration_cast<chrono::microseconds>(end - begin);
    cout << "microseconds " << eslaped.count() << endl;
    return 0;
}
