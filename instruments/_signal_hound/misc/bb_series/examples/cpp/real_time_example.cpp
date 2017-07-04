#include "bb_api.h"
#pragma comment(lib, "bb_api.lib")

#include <iostream>

void realTimeExample()
{
    int handle = -1;
    bbStatus openStatus = bbOpenDevice(&handle);
    if(openStatus != bbNoError) {
        std::cout << bbGetErrorString(openStatus);
        return;
    }

    // Configure a 27MHz real-time stream at a 2.44GHz center
    bbConfigureAcquisition(handle, BB_MIN_AND_MAX, BB_LOG_SCALE);
    bbConfigureCenterSpan(handle, 2.44e9, 20.0e6);
    bbConfigureLevel(handle, -20.0, BB_AUTO_ATTEN);
    bbConfigureGain(handle, BB_AUTO_GAIN);
    // 9.8kHz RBW, for real-time must specify a Nuttall BW value,
    // See API manual for possible RBW values
    bbConfigureSweepCoupling(handle, 9863.28125, 9863.28125, 0.001, 
        BB_RBW_SHAPE_NUTTALL, BB_NO_SPUR_REJECT);
    // Configure a frame rate of 30fps and 100dB scale
    bbConfigureRealTime(handle, 100.0, 30);

    // Initialize the device for real-time mode
    if(bbInitiate(handle, BB_REAL_TIME, 0) != bbNoError) {
        // Handle error
    }

    // Get sweep characteristics and allocate memory for sweep and
    // real-time frame.
    unsigned int sweepSize;
    double binSize, startFreq;
    bbQueryTraceInfo(handle, &sweepSize, &binSize, &startFreq);
    int frameWidth, frameHeight;
    bbQueryRealTimeInfo(handle, &frameWidth, &frameHeight);

    float *sweep = new float[sweepSize];
    float *frame = new float[frameWidth * frameHeight];

    // Retrieve roughly 1 second worth of real-time persistence frames and sweeps.
    int frameCount = 0;
    while(frameCount++ < 30) {
        bbFetchRealTimeFrame(handle, sweep, frame);
    }

    delete [] sweep;
    delete [] frame;

    // Finished/close device
    bbAbort(handle);
    bbCloseDevice(handle);
}