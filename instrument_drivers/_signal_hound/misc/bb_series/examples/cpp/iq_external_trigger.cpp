#include "bb_api.h"
#pragma comment(lib, "bb_api.lib")

#include <iostream>

/*
This example demonstrates configuring the device to accept and report
external triggers on the BNC port of the BB60. If enabled while IQ streaming,
the API can report the IQ sample on which an external trigger event was recorded.
This enables correlating a digital logic event to a location in the RF data stream.
*/

void iqWithExternalTrigger()
{
    int handle;
    bbStatus status = bbOpenDevice(&handle);
    // Check if device opened properly
    if(status != bbNoError) {
        std::cout << bbGetErrorString(status);
        return;
    }

    // Set center frequency, span is ignored for IQ streaming
    bbConfigureCenterSpan(handle, 869.0e6, 1.0e3);
    // Set reference level and auto gain/atten
    bbConfigureLevel(handle, -20.0, BB_AUTO_ATTEN);
    bbConfigureGain(handle, BB_AUTO_GAIN);
    // Set a sample rate of 40.0e6 / 2 = 20.0e6 MS/s and bandwidth of 15 MHz
    bbConfigureIQ(handle, 2, 15.0e6);
    // Configure BNC port 2 for input rising edge trigger detection
    bbConfigureIO(handle, 0, BB_PORT2_IN_TRIGGER_RISING_EDGE);

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

    int triggers[71];

    // Set up block capture
    bbIQPacket pkt;
    pkt.iqData = buffer;
    pkt.iqCount = BLOCK_SIZE;
    // Specify a pointer to the trigger buffer and size of buffer.
    pkt.triggers = triggers;
    pkt.triggerCount = 70;
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