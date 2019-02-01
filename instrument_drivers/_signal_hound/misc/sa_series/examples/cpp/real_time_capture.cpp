#include "sa_api.h"
#pragma comment(lib, "sa_api.lib")

#include <iostream>

void realTimeSweep()
{
    int handle = -1;
    saStatus openStatus = saOpenDevice(&handle);
    if(openStatus != saNoError) {
        // Handle unable to open/find device error here
        std::cout << saGetErrorString(openStatus) << std::endl;
        return;
    }

    // Configure real-time analysis to be centered on a local
    // FM broadcast frequency, with a 1kHz RBW.
    // Set a frame rate of 30fps, and 100dB height on persistence frames.
    saConfigCenterSpan(handle, 97.1e6, 200.0e3);
    saConfigAcquisition(handle, SA_MIN_MAX, SA_LOG_SCALE);
    saConfigLevel(handle, -10.0);
    saConfigSweepCoupling(handle, 1.0e3, 1.0e3, true);
    saConfigRealTime(handle, 100.0, 30);

    // Initialize the device with the configuration just set
    saStatus initiateStatus = saInitiate(handle, SA_REAL_TIME, 0);
    if(initiateStatus != saNoError) {
        // Unable to initialize
        std::cout << saGetErrorString(initiateStatus) << std::endl;
        return;
    }

    // Get sweep and frame characteristics
    int sweepLen;
    double startFreq, binSize;
    saQuerySweepInfo(handle, &sweepLen, &startFreq, &binSize);
    int frameWidth, frameHeight;
    saQueryRealTimeFrameInfo(handle, &frameWidth, &frameHeight);

    // Allocate memory for the sweep and frame
    float *max = new float[sweepLen];
    float *frame = new float[frameWidth * frameHeight];

    // Get 30 frames and sweeps, representing 1 second of real-time analysis
    int frames = 0;
    while(frames < 30) {
        saStatus sweepStatus = saGetRealTimeFrame(handle, max, frame);
        frames++;

        // Update your application
    }

    saAbort(handle);
    delete [] max;
    delete [] frame;
    
    saCloseDevice(handle);
}