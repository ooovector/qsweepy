#include "sa_api.h"
#pragma comment(lib, "sa_api.lib")

#include <iostream>

// This example is the minimal code needed to configure the device
// to perform a single sweep. The device will block for several seconds
// while opening the device, and the call to saGetSweep will block

void simpleSweep1()
{
    int handle = -1;
    saStatus openStatus = saOpenDevice(&handle);
    if(openStatus != saNoError) {
        // Handle unable to open/find device error here
        std::cout << saGetErrorString(openStatus) << std::endl;
        return;
    }

    // Configure the device to sweep a 1MHz span centered on 900MHz
    // Min/Max detector, with RBW/VBW equal to 1kHz
    saConfigCenterSpan(handle, 900.0e6, 1.0e6);
    saConfigAcquisition(handle, SA_MIN_MAX, SA_LOG_SCALE);
    saConfigLevel(handle, -10.0);
    saConfigSweepCoupling(handle, 1.0e3, 1.0e3, true);

    // Initialize the device with the configuration just set
    saStatus initiateStatus = saInitiate(handle, SA_SWEEPING, 0);
    if(initiateStatus != saNoError) {
        // Handle unable to initialize
        std::cout << saGetErrorString(initiateStatus) << std::endl;
        return;
    }

    // Get sweep characteristics
    int sweepLen;
    double startFreq, binSize;
    saQuerySweepInfo(handle, &sweepLen, &startFreq, &binSize);

    // Allocate memory for the sweep
    float *min = new float[sweepLen];
    float *max = new float[sweepLen];

    // Get 1 or more sweeps with this configuration
    // This function can be called several times and will return a 
    //  sweep measured directly after the function is called.
    // The function blocks until the sweep is returned.
    saStatus sweepStatus = saGetSweep_32f(handle, min, max);

    delete [] min;
    delete [] max;

    saAbort(handle);
    saCloseDevice(handle);	
}