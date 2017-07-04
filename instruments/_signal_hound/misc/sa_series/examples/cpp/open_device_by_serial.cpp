#include "sa_api.h"
#pragma comment(lib, "sa_api.lib")

#include <iostream>

// Selectively open a device by specifying the serial number
// of the target device

void openDeviceBySerial()
{
    int handle = -1;
    int serials[8] = {0}, deviceCount;

    saGetSerialNumberList(serials, &deviceCount);

    if(deviceCount <= 0) {
        return;
    }

    // Find desired serial number
    int openIndex = 0;

    saStatus openStatus = saOpenDeviceBySerialNumber(&handle, serials[openIndex]);
    if(openStatus != saNoError) {
        std::cout << saGetErrorString(openStatus) << std::endl;
        return;
    }

    // Device is open here

    // When finished
    saCloseDevice(handle);
}