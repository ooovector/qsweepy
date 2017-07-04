using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

class Program
{
    static void Main(string[] args)
    {
        int id = -1;
        bbStatus status = bbStatus.bbNoError;

        Console.Write("Opening Device, Please Wait\n");
        status = bb_api.bbOpenDevice(ref id);
        if (status != bbStatus.bbNoError)
        {
            Console.Write("Error: Unable to open BB60\n");
            Console.Write(bb_api.bbGetStatusString(status) + "\n");
            prompt_user_input();
            return;
        }
        else
        {
            Console.Write("Device Found\n\n");
        }

        Console.Write("API Version: " + bb_api.bbGetAPIString() + "\n");
        Console.Write("Device Type: " + bb_api.bbGetDeviceName(id) + "\n");
        Console.Write("Serial Number: " + bb_api.bbGetSerialString(id) + "\n");
        Console.Write("Firmware Version: " + bb_api.bbGetFirmwareString(id) + "\n");
        Console.Write("\n");

        float temp = 0.0F, voltage = 0.0F, current = 0.0F;
        bb_api.bbGetDeviceDiagnostics(id, ref temp, ref voltage, ref current);
        Console.Write("Device Diagnostics\n" +
            "Temperature: " + temp.ToString() + " C\n" +
            "USB Voltage: " + voltage.ToString() + " V\n" +
            "USB Current: " + current.ToString() + " mA\n");
        Console.Write("\n");

        Console.Write("Configuring Device For a Sweep\n");
        bb_api.bbConfigureAcquisition(id, bb_api.BB_MIN_AND_MAX, bb_api.BB_LOG_SCALE);
        bb_api.bbConfigureCenterSpan(id, 1.0e9, 20.0e6);
        bb_api.bbConfigureLevel(id, -20.0, bb_api.BB_AUTO_ATTEN);
        bb_api.bbConfigureGain(id, bb_api.BB_AUTO_GAIN);
        bb_api.bbConfigureSweepCoupling(id, 10.0e3, 10.0e3, 0.001,
            bb_api.BB_NON_NATIVE_RBW, bb_api.BB_NO_SPUR_REJECT);
        bb_api.bbConfigureProcUnits(id, bb_api.BB_LOG);

        status = bb_api.bbInitiate(id, bb_api.BB_SWEEPING, 0);
        if (status != bbStatus.bbNoError)
        {
            Console.Write("Error: Unable to initialize BB60\n");
            Console.Write(bb_api.bbGetStatusString(status) + "\n");
            prompt_user_input();
            return;
        }

        uint trace_len = 0;
        double bin_size = 0.0;
        double start_freq = 0.0;
        status = bb_api.bbQueryTraceInfo(id, ref trace_len, ref bin_size, ref start_freq);

        float[] sweep_max, sweep_min;
        sweep_max = new float[trace_len];
        sweep_min = new float[trace_len];

        bb_api.bbFetchTrace_32f(id, unchecked((int)trace_len), sweep_min, sweep_max);
        Console.Write("Sweep Retrieved\n\n");

        Console.Write("Configuring the device to stream I/Q data\n");
        bb_api.bbConfigureCenterSpan(id, 1.0e9, 20.0e6);
        bb_api.bbConfigureLevel(id, -20.0, bb_api.BB_AUTO_ATTEN);
        bb_api.bbConfigureGain(id, bb_api.BB_AUTO_GAIN);
        bb_api.bbConfigureIQ(id, bb_api.BB_MIN_DECIMATION, 20.0e6);
        
        status = bb_api.bbInitiate(id, bb_api.BB_STREAMING, bb_api.BB_STREAM_IQ);
        if (status != bbStatus.bbNoError)
        {
            Console.Write("Error: Unable to initialize BB60 for streaming\n");
            Console.Write(bb_api.bbGetStatusString(status) + "\n");
            prompt_user_input();
            return;
        }

        int return_len = 0;
        int samples_per_sec = 0;
        double bandwidth = 0.0;
        bb_api.bbQueryStreamInfo(id, ref return_len, ref bandwidth, ref samples_per_sec);
        Console.Write("Initialized Stream for \n");
        Console.Write("Samples per second: " + (samples_per_sec/1.0e6).ToString() + " MS/s\n");
        Console.Write("Bandwidth: " + (bandwidth/1.0e6).ToString() + " MHz\n");
        Console.Write("Samples per function call: " + return_len.ToString() + "\n");

        // Alternating I/Q samples
        // return_len is the number of I/Q pairs, so.. allocate twice as many floats
        float[] iq_samples = new float[return_len * 2];
        int[] triggers = new int[80];

        bb_api.bbFetchRaw(id, iq_samples, triggers);
        Console.Write("Retrieved one I/Q packet\n\n");

        Console.Write("Closing Device\n");
        bb_api.bbCloseDevice(id);

        prompt_user_input();
    }

    private static void prompt_user_input()
    {
        Console.Write("Press enter to end the program\n");
        Console.Read();
    }
}
