#include "sa_api.h"
#pragma comment(lib, "sa_api.lib")

#include <iostream>

// This example expands on simple_sweep_1
// Calling saGetSweep is fine for the majority of sweep configurations,
// but for spans larger than 100MHz the device can take several seconds or
// more to sweep, all the way up to 30 seconds for a full span sweep. The API
// provides a way to retrieve the sweep as it is processed and present the 
// partially updated sweep to you. Below is an example of how you might perform 
// this. This type of functionality might be desired if you do not want
// your application to block for the full duration of the sweep.

void simpleSweep2()
{
    int handle = -1;
    saStatus openStatus = saOpenDevice(&handle);
    if(openStatus != saNoError) {
        // Handle unable to open/find device error here
        std::cout << saGetErrorString(openStatus) << std::endl;
        return;
    }

    // Configure a 2GHz span sweep centered at 2GHz at 100kHz RBW
    saConfigCenterSpan(handle, 2.0e9, 2.0e6);
    saConfigAcquisition(handle, SA_MIN_MAX, SA_LOG_SCALE);
    saConfigLevel(handle, -10.0);
    saConfigSweepCoupling(handle, 100.0e3, 100.0e3, true);

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

    // Get one full sweep through the partial sweep function
    int startIndex = 0, stopIndex = 0;
    while(stopIndex < sweepLen) {
        saStatus sweepStatus = saGetPartialSweep_32f(handle, 
            min, max, &startIndex, &stopIndex);

        // Update your program here if desired
    }

    // Multiple other sweeps can be acquired here

    delete [] min;
    delete [] max;

    saAbort(handle);
    saCloseDevice(handle);	
}