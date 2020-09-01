#include "bb_api.h"
#pragma comment(lib, "bb_api.lib")

#include <iostream>

void sweepExample()
{
    int handle = -1;
    bbStatus openStatus = bbOpenDevice(&handle);
    if(openStatus != bbNoError) {
        std::cout << bbGetErrorString(openStatus);
        return;
    }

    // Configure a sweep from 850MHz to 950MHz with an 
    //  RBW and VBW of 10kHz and an expected input of -20dBm
    bbConfigureAcquisition(handle, BB_MIN_AND_MAX, BB_LOG_SCALE);
    bbConfigureCenterSpan(handle, 900.0e6, 100.0e6);
    bbConfigureLevel(handle, -20.0, BB_AUTO_ATTEN);
    bbConfigureGain(handle, BB_AUTO_GAIN);
    bbConfigureSweepCoupling(handle, 10.0e3, 10.0e3, 0.001, BB_RBW_SHAPE_FLATTOP, BB_NO_SPUR_REJECT);
    bbConfigureProcUnits(handle, BB_POWER);

    // Configuration complete, initialize the device
    if(bbInitiate(handle, BB_SWEEPING, 0) != bbNoError) {
        // Handle error
    }

    // Get sweep characteristics and allocate memory for sweep
    unsigned int sweepSize;
    double binSize, startFreq;
    bbQueryTraceInfo(handle, &sweepSize, &binSize, &startFreq);

    float *min = new float[sweepSize];
    float *max = new float[sweepSize];

    // Get one or many sweeps with these configurations
    bbFetchTrace_32f(handle, sweepSize, min, max);

    delete [] min;
    delete [] max;

    // Finished/close device
    bbAbort(handle);
    bbCloseDevice(handle);
}