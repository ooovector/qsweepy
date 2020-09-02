Note for API versions 3.0.5 and later: You will need to place both the bb_api.dll file and ftd2xx.dll files in your development directory or you will receive "DLL not found" errors.

The bb_api.cs source file contains the bb_api class, which acts as an interface to our BB60C/A API. The methods provided in the bb_api class have one-to-one mappings to the C functions provided by our C++ header file. This makes it easy to port an existing project to/from C++, and also makes it easy to look up the functions in the API manual.

The example.cs source file contains a method which exercises the bb_api class. Import both files into an empty C# console application to get a jump-start on your project. 

This interface class exposes the necessary functionality to sweep the BB60 device and to perform continuous I/Q streaming.

Please take a look at the API manual to learn more about the API and its functions. When you are building a C# application, you must be sure to place the proper 32/64-bit bb_api.dll file into your build directory. e.g. If your C# CPU build type is 32-bit, put the 32-bit bb_api.dll file in your specified build directory. If the DLL bit size does not match your project you will receive this error...

"An unhandled exception of type 'System.BadImageFormatException' occurred in csharp_bbapi.exe"
"Additional information: An attempt was made to load a program with an incorrect format. (Exception from HRESULT: 0x8007000B)"

If you have any further questions contact SignalHound at aj@signalhound.com.