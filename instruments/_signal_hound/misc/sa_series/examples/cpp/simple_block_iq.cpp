#include "sa_api.h"
#pragma comment(lib, "sa_api.lib")

#include <iostream>

// This examples demonstrates configuring the device and retrieving one
// block of IQ data of arbitrary length.

void iqBlockCapture()
{
    int handle = -1;
    saStatus openStatus = saOpenDevice(&handle);
    if(openStatus != saNoError) {
        // Handle unable to open/find device error here
        std::cout << saGetErrorString(openStatus) << std::endl;
        return;
    }

    // Set center freq, span is ignored
    saConfigCenterSpan(handle, 869.0e6, 1.0e3);
    // Set expected input level
    saConfigLevel(handle, -10.0);
    // Configure sample rate and bandwidth
    // Sample rate of 486111.11 / 4 and bandwidth of 100kHz
    saConfigIQ(handle, 4, 100.0e3);

    // Initialize the API for IQ streaming. The device will begin streaming
    // until saAbort(), saInitiate(), or saCloseDevice() is called.
    saInitiate(handle, SA_IQ, 0);

    // Verify the sample rate and bandwidth of the IQ stream
    double bandwidth, sampleRate;
    saQueryStreamInfo(handle, 0, &bandwidth, &sampleRate);

    // Specify how many IQ samples to collect
    const int BLOCK_SIZE = 16384;

    saIQPacket pkt;
    pkt.iqData = new float[BLOCK_SIZE * 2];
    pkt.iqCount = BLOCK_SIZE;
    // By setting purge to true, you indicate to the API, that you wish
    // the acquisition of the IQ data block to occur after you call
    // the saGetIQData() function. See the iqStream example for an example
    // of setting this value to false.
    pkt.purge = true; 

    // Retrieve IQ data
    saStatus iqStatus = saGetIQData(handle, &pkt);

    // At this point, BLOCK_SIZE IQ data samples have been retrieved and
    //  stored in the array created at pkt.iqData
    // saGetIQData can continue to be called to retrieve more IQ samples

    // Clean up
    delete [] pkt.iqData;

    saAbort(handle);
    saCloseDevice(handle);
}