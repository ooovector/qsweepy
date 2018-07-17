using System;
using System.Runtime.InteropServices;

// The C# API class is class of static members which is a one-to-one
//  mapping of the C API. This makes it very easy to look up functions
//  in the API manual. 
// This class exposes the necessary functionality for sweeping the device
//  and performing I/Q streaming. 

// Place the 32 or 64 bit bb_api.dll file in the same directory as
//  your c# executable. The dll must match your project CPU type, e.g.
//  If you are building a 32-bit application, use the 32-bit dll.

// Status Codes
// Errors are negative and suffixed with 'Err'
// Errors stop the flow of execution, warnings do not
enum bbStatus
{
    // Configuration Errors
    bbInvalidModeErr = -112,
    bbReferenceLevelErr = -111,
    bbInvalidVideoUnitsErr = -110,
    bbInvalidWindowErr = -109,
    bbInvalidBandwidthTypeErr = -108,
    bbInvalidSweepTimeErr = -107,
    bbBandwidthErr = -106,
    bbInvalidGainErr = -105,
    bbAttenuationErr = -104,
    bbFrequencyRangeErr = -103,
    bbInvalidSpanErr = -102,
    bbInvalidScaleErr = -101,
    bbInvalidDetectorErr = -100,

    // General Errors
    bbUSBTimeoutErr = -15,
    bbDeviceConnectionErr = -14,
    bbPacketFramingErr = -13,
    bbGPSErr = -12,
    bbGainNotSetErr = -11,
    bbDeviceNotIdleErr = -10,
    bbDeviceInvalidErr = -9,
    bbBufferTooSmallErr = -8,
    bbNullPtrErr = -7,
    bbAllocationLimitErr = -6,
    bbDeviceAlreadyStreamingErr = -5,
    bbInvalidParameterErr = -4,
    bbDeviceNotConfiguredErr = -3,
    bbDeviceNotStreamingErr = -2,
    bbDeviceNotOpenErr = -1,

    // No Error
    bbNoError = 0,

    // Warnings/Messages
    bbAdjustedParameter = 1,
    bbADCOverflow = 2,
    bbNoTriggerFound = 3,
    bbClampedToUpperLimit = 4,
    bbClampedToLowerLimit = 5,
    bbUncalibratedDevice = 6
};

class bb_api
{
    // bbGetDeviceType : type
    public static int BB_DEVICE_BB60A = 0x1;
    public static int BB_DEVICE_BB60C = 0x2;
    // bbConfigureLevel : atten
    public static double BB_AUTO_ATTEN = -1.0;
    // bbConfigureGain : gain
    public static int BB_AUTO_GAIN = -1;
    // bbConfigAcquisition : detector
    public static uint BB_MIN_AND_MAX = 0x0;
    public static uint BB_AVERAGE = 0x1;
    public static uint BB_MIN_ONLY = 0x2;
    public static uint BB_MAX_ONLY = 0x3;
    // bbConfigAcquisition : scale
    public static uint BB_LOG_SCALE = 0x0;
    public static uint BB_LIN_SCALE = 0x1;
    public static uint BB_LOG_FULL_SCALE = 0x2;
    public static uint BB_LIN_FULL_SCALE = 0x3;
    // bbConfigureSweepCoupling : rbwType
    public static uint BB_NATIVE_RBW = 0x0;
    public static uint BB_NON_NATIVE_RBW = 0x1;
    // bbConfigureSweepCoupling : rejection
    public static uint BB_NO_SPUR_REJECT = 0x0;
    public static uint BB_SPUR_REJECT = 0x1;
    // bbConfigureWindow : window
    public static uint BB_NUTALL = 0x0;
    public static uint BB_BLACKMAN = 0x1;
    public static uint BB_HAMMING = 0x2;
    public static uint BB_FLAT_TOP = 0x3;
    // bbConfigureProcUnits : units
    public static uint BB_LOG = 0x0;
    public static uint BB_VOLTAGE = 0x1;
    public static uint BB_POWER = 0x2;
    public static uint BB_SAMPLE = 0x3;
    // bbConfigureIQ : downsampleFactor
    public static int BB_MIN_DECIMATION = 1; // 2 ^ 0
    public static int BB_MAX_DECIMATION = 128; // 2 ^ 7
    // bbInitiate : mode
    public static uint BB_SWEEPING = 0x0;
    public static uint BB_REAL_TIME = 0x1;
    public static uint BB_STREAMING = 0x4;
    public static uint BB_AUDIO_DEMOD = 0x7;
    // bbInitiate : flag
    public static uint BB_STREAM_IQ = 0x0;
    public static uint BB_STREAM_IF = 0x1;
    // bbConfigureIO : port1
    public static uint BB_PORT1_AC_COUPLED = 0x00;
    public static uint BB_PORT1_DC_COUPLED = 0x04;
    public static uint BB_PORT1_INT_REF_OUT = 0x00;
    public static uint BB_PORT1_EXT_REF_IN = 0x08;
    public static uint BB_PORT1_OUT_AC_LOAD = 0x10;
    public static uint BB_PORT1_OUT_LOGIC_LOW = 0x14;
    public static uint BB_PORT1_OUT_LOGIC_HIGH = 0x1C;
    // bbConfigureIO : port2
    public static uint BB_PORT2_OUT_LOGIC_LOW = 0x00;
    public static uint BB_PORT2_OUT_LOGIC_HIGH = 0x20;
    public static uint BB_PORT2_IN_TRIGGER_RISING_EDGE = 0x40;
    public static uint BB_PORT2_IN_TRIGGER_FALLING_EDGE = 0x60;

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbOpenDevice(ref int device);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbCloseDevice(int device);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureAcquisition(int device,
        uint detector, uint scale);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureCenterSpan(int device,
        double center, double span);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureLevel(int device,
        double ref_level, double atten);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureGain(int device, int gain);
    
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureSweepCoupling(int device,
        double rbw, double vbw, double sweepTime, uint rbw_type, uint rejection);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureWindow(int device, uint window);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureProcUnits(int device, uint units);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureIO(int device,
        uint port1, uint port2);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureDemod(int device, 
        int mod_type, double freq, float if_bandwidth, float low_pass_freq, 
        float high_pass_freq, float fm_deemphasis);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureIQ(int device,
        int downsampleFactor, double bandwidth);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbInitiate(int device,
        uint mode, uint flag);

    [DllImport("bb_api", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbFetchTrace_32f(int device,
        int arraySize, float[] min, float[] max);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbFetchTrace(int device,
        int array_size, double[] min, double[] max);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbFetchAudio(int device, ref float audio);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbFetchRaw(int device,
        float[] buffer,  int[] triggers);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbQueryTraceInfo(int device,
        ref uint trace_len, ref double bin_size, ref double start);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbQueryStreamInfo(int device, 
        ref int return_len, ref double bandwidth, ref int samples_per_sec);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbQueryTimestamp(int device,
        ref uint seconds, ref uint nanoseconds);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbAbort(int device);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbPreset(int device);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbSelfCal(int device);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbSyncCPUtoGPS(int com_port, int baud_rate);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbGetDeviceType(int device, ref int type);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbGetSerialNumber(int device, ref uint serial_number);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbGetFirmwareVersion(int device, ref int version);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbGetDeviceDiagnostics(int device,
        ref float temperature, ref float usbVoltage, ref float usbCurrent);

    public static string bbGetDeviceName(int device)
    {
        int device_type = -1;
        bbGetDeviceType(device, ref device_type);
        if (device_type == BB_DEVICE_BB60A)
            return "BB60A";
        if (device_type == BB_DEVICE_BB60C)
            return "BB60C";

        return "Unknown device";
    }

    public static string bbGetSerialString(int device)
    {
        uint serial_number = 0;
        if (bbGetSerialNumber(device, ref serial_number) == bbStatus.bbNoError)
            return serial_number.ToString();

        return "";
    }

    public static string bbGetFirmwareString(int device)
    {
        int firmware_version = 0;
        if (bbGetFirmwareVersion(device, ref firmware_version) == bbStatus.bbNoError)
            return firmware_version.ToString();

        return "";
    }

    public static string bbGetAPIString()
    {
        IntPtr str_ptr = bbGetAPIVersion();
        return System.Runtime.InteropServices.Marshal.PtrToStringAnsi(str_ptr);
    }

    public static string bbGetStatusString(bbStatus status)
    {
        IntPtr str_ptr = bbGetErrorString(status);
        return System.Runtime.InteropServices.Marshal.PtrToStringAnsi(str_ptr);
    }

    // Call get_string variants above instead
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr bbGetAPIVersion();
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr bbGetErrorString(bbStatus status);
}

