#include "bb_api.h"
#pragma comment(lib, "bb_api.lib")

#include <iostream>

/*
This example demonstrates how to configure, initialize, and retrieve data
from a BB60 in the IQ streaming mode.
*/

void iqExample()
{
    int handle = -1;
    bbStatus status = bbOpenDevice(&handle);
    if(status != bbNoError) {
        std::cout << bbGetErrorString(status);
        return;
    }

    // Set center frequency, span is ignored for IQ streaming
    bbConfigureCenterSpan(handle, 2400.0e6, 1.0e3);
    // Set reference level and auto gain/atten
    bbConfigureLevel(handle, -20.0, BB_AUTO_ATTEN);
    bbConfigureGain(handle, BB_AUTO_GAIN);
    // Set a sample rate of 40.0e6 / 2 = 20.0e6 MS/s and bandwidth of 15 MHz
    bbConfigureIQ(handle, 2, 15.0e6);

    // Initialize the device for IQ streaming
    status = bbInitiate(handle, BB_STREAMING, BB_STREAM_IQ);
    if(status != bbNoError) {
        // Handle error
    }

    // Verify bandwidth and sample rate
    double bandwidth;
    int sampleRate;
    bbQueryStreamInfo(handle, 0, &bandwidth, &sampleRate);

    // Allocate memory for BLOCK_SIZE complex values
    const int BLOCK_SIZE = 262144;
    float *buffer = new float[BLOCK_SIZE * 2];

    // Set up block capture
    bbIQPacket pkt;
    pkt.iqData = buffer;
    pkt.iqCount = BLOCK_SIZE;
    pkt.triggers = 0;
    pkt.triggerCount = 0;
    // Setting purge to true tells the API to return IQ data that has been
    // acquired only after calling the bbGetIQ() function. Setting purge to false
    // will return data contiguous to the data returned from the last call to
    // bbGetIQ().
    pkt.purge = true;

    // Perform capture
    status = bbGetIQ(handle, &pkt);

    // At this point, BLOCK_SIZE IQ data samples have been retrieved and
    //  stored in the buffer array.
    // bbGetIQ can continue to be called to retrieve more IQ samples.

    delete [] buffer;

    bbAbort(handle);
    bbCloseDevice(handle);
}
