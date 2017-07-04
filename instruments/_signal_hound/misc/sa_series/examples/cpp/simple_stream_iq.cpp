#include "sa_api.h"
#pragma comment(lib, "sa_api.lib")

#include <iostream>

// This example demonstrates configuring the device to stream continuous
// IQ data to your application.

void iqStreaming()
{
    int handle = -1;
    saStatus openStatus = saOpenDevice(&handle);
    if(openStatus != saNoError) {
        // Handle unable to open/find device error here
        std::cout << saGetErrorString(openStatus) << std::endl;
        return;
    }

    // Set center freq, span is ignored
    saConfigCenterSpan(handle, 97.1e3, 1.0e3);
    // Set expected input level
    saConfigLevel(handle, -10.0);
    // Configure sample rate and bandwidth
    // Sample rate of 486111.11 / 1 and bandwidth of 250kHz
    saConfigIQ(handle, 1, 250.0e3);

    saInitiate(handle, SA_IQ, 0);

    // Verify the sample rate and bandwidth of the IQ stream
    double bandwidth, sampleRate;
    saQueryStreamInfo(handle, 0, &bandwidth, &sampleRate);

    // How many IQ samples to collect per call
    const int BUF_SIZE = 4096;

    saIQPacket pkt;
    pkt.iqData = new float[BUF_SIZE * 2];
    pkt.iqCount = BUF_SIZE;
    // Setting purge to false ensures each call to getIQPacket()
    // returns contiguous IQ data to the last time the function was called.
    // This also means IQ data must be queried at the rate of the
    // device sample rate. In this case, the sample rate is 486.111k,
    // so the saGetIQData function must be called
    // 486111 / 4096 = ~118 times per second.
    pkt.purge = false; 

    // Retreive about 1 second worth of contiguous IQ data or
    // 120 * 4096 IQ data values.
    int pktCount = 0;
    while(pktCount++ < 120) {
        // Get next contiguous block of IQ data
        saStatus iqStatus = saGetIQData(handle, &pkt);
        std::cout << pkt.sec << " " << pkt.milli << std::endl;
        // Store/process data before getting another buffer
        // Check any errors or status updates 
    }

    // Clean up
    delete [] pkt.iqData;

    saAbort(handle);
    saCloseDevice(handle);
}