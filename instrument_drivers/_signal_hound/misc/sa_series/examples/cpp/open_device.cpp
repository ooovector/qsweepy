#include "sa_api.h"
#pragma comment(lib, "sa_api.lib")

#include <iostream>

// Open any SA44/SA124 and display all available diagnostic information

void openDevice()
{
    std::cout << "API Version: " << saGetAPIVersion() << std::endl;

    int handle = -1;
    saStatus openStatus = saOpenDevice(&handle);
    if(openStatus != saNoError) {
        std::cout << saGetErrorString(openStatus) << std::endl;
        return;
    }

    std::cout << "Device Found!\n";

    saDeviceType deviceType;
    saGetDeviceType(handle, &deviceType);

    std::cout << "Device Type: ";
    if(deviceType == saDeviceTypeSA44) {
        std::cout << "SA44\n";
    } else if(deviceType == saDeviceTypeSA44B) {
        std::cout << "SA44B\n";
    } else if(deviceType == saDeviceTypeSA124A) {
        std::cout << "SA124A\n";
    } else if(deviceType == saDeviceTypeSA124B) {
        std::cout << "SA124B\n";
    }

    int serialNumber = 0;
    saGetSerialNumber(handle, &serialNumber);
    std::cout << "Serial Number: " << serialNumber << std::endl;

    float temperature = 0.0;
    saQueryTemperature(handle, &temperature);
    std::cout << "Internal Temp: " << temperature << " C" << std::endl;

    float voltage = 0.0;
    saQueryDiagnostics(handle, &voltage);
    std::cout << "Voltage: " << voltage << " V" << std::endl;

    saCloseDevice(handle);
}