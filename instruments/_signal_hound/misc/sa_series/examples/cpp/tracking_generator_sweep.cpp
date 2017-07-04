#include "sa_api.h"
#pragma comment(lib, "sa_api.lib")

#include <iostream>

// This example demonstrates how to use the API to perform a single TG sweep.
// See the manual for a full description of each step of the process in the
// Scalar Network Analysis section.

void trackingGeneratorSweep()
{
    int handle = -1;
    saStatus openStatus = saOpenDevice(&handle);
    if(openStatus != saNoError) {
        // Handle unable to open/find device error here
        std::cout << saGetErrorString(openStatus) << std::endl;
        return;
    }

    if(saAttachTg(handle) != saNoError) {
        // Unable to find tracking generator
        return;
    }

    // Sweep some device at 900MHz center with 1MHz span
    saConfigCenterSpan(handle, 900.0e6, 1.0e6);
    saConfigAcquisition(handle, SA_MIN_MAX, SA_LOG_SCALE);
    saConfigLevel(handle, -10.0);
    saConfigSweepCoupling(handle, 1.0e3, 1.0e3, true);

    // Additional configuration routine
    // Configure a 100 point sweep
    // The size of the sweep is a suggestion to the API, it will attempt to
    // get near the requested size.
    // Optimized for high dynamic range and passive devices
    saConfigTgSweep(handle, 100, true, true);

    // Initialize the device with the configuration just set
    if(saInitiate(handle, SA_TG_SWEEP, 0) != saNoError) {
        // Handle unable to initialize
        return;
    }

    // Get sweep characteristics
    int sweepLen;
    double startFreq, binSize;
    saQuerySweepInfo(handle, &sweepLen, &startFreq, &binSize);

    // Allocate memory for the sweep
    float *min = new float[sweepLen];
    float *max = new float[sweepLen];

    // Create test set-up without DUT present
    // Get one sweep
    saGetSweep_32f(handle, min, max);
    // Store baseline
    saStoreTgThru(handle, TG_THRU_0DB);

    // Should pause here, and insert DUT into test set-up
    saGetSweep_32f(handle, min, max);

    // From here, you can sweep several times without needing to restore the thru, 
    // once you change your setup, you should reconfigure the device and 
    // store the thru again without the DUT inline.

    delete [] min;
    delete [] max;

    saAbort(handle);
    saCloseDevice(handle);
}