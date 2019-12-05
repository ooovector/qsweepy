import ctypes
import logging
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

try:
	signal_hound_dll = ctypes.WinDLL (module_dir+"\\sa_api.dll")
except Exception as e:
	print (e)

# Limits
sa44_min_freq = 1.
#define SA44_MIN_FREQ (1.0)
sa124_min_freq = 100.e3
#define SA124_MIN_FREQ (100.0e3)
sa44_max_freq = 4.4e9
#define SA44_MAX_FREQ (4.4e9)
sa124_max_freq = 13.e9
#define SA124_MAX_FREQ (13.0e9)
min_span = 1.0
#define SA_MIN_SPAN (1.0)
max_ref = 20.
#define SA_MAX_REF (20) // dBm
max_atten = 3
#define SA_MAX_ATTEN (3)
max_gain = 2
#define SA_MAX_GAIN (2)
min_rbw = 0.1
#define SA_MIN_RBW (0.1)
max_rbw = 6.e6
#define SA_MAX_RBW (6.0e6)
min_rt_rbw = 100.
#define SA_MIN_RT_RBW (100.0)
max_rt_rbw = 10000.
#define SA_MAX_RT_RBW (10000.0)
min_iq_bandwidth = 100.
#define SA_MIN_IQ_BANDWIDTH (100.0)
max_iq_decimation = 128
#define SA_MAX_IQ_DECIMATION (128)

iq_sample_rate = 486111.111
#define SA_IQ_SAMPLE_RATE (486111.111)

# Modes
idle = -1
#define SA_IDLE      (-1)
sweeping = 0
#define SA_SWEEPING  (0x0)
real_time = 1
#define SA_REAL_TIME (0x1)
iq = 2
#define SA_IQ        (0x2)
audio = 3
#define SA_AUDIO     (0x3)
tg_sweep = 4
#define SA_TG_SWEEP  (0x4)

# RBW shapes
rbw_shape_flattop = 1
#define SA_RBW_SHAPE_FLATTOP (0x1)
rbw_shape_cispr = 2
#define SA_RBW_SHAPE_CISPR (0x2)

# Detectors
min_max = 0
#define SA_MIN_MAX (0x0)
average = 1
#define SA_AVERAGE (0x1)

# Scales
log_scale = 0
#define SA_LOG_SCALE      (0x0)
lin_scale = 1
#define SA_LIN_SCALE      (0x1)
log_full_scale = 2
#define SA_LOG_FULL_SCALE (0x2) // N/A
lin_full_scale = 3
#define SA_LIN_FULL_SCALE (0x3) // N/A

# Levels
sa_auto_atten = -1
#define SA_AUTO_ATTEN (-1)
sa_auto_gain = -1
#define SA_AUTO_GAIN  (-1)

# Video Processing Units
log_units = 0
#define SA_LOG_UNITS   (0x0)
volt_units = 1
#define SA_VOLT_UNITS  (0x1)
power_units = 2
#define SA_POWER_UNITS (0x2)
bypass = 3
#define SA_BYPASS      (0x3)

audio_am = 0
#define SA_AUDIO_AM  (0x0)
audio_fm = 1
#define SA_AUDIO_FM  (0x1)
audio_usb = 2
#define SA_AUDIO_USB (0x2)
audio_lsb = 3
#define SA_AUDIO_LSB (0x3)
audio_cw = 4
#define SA_AUDIO_CW  (0x4)

# TG Notify Flags
tg_thru_0dB = 0
#define TG_THRU_0DB  (0x1)
tg_thru_20dB = 0
#define TG_THRU_20DB  (0x2)

ref_unused = 0
#define SA_REF_UNUSED (0)
ref_internal_out = 1
#define SA_REF_INTERNAL_OUT (1)
ref_external_out = 2
#define SA_REF_EXTERNAL_IN (2)

get_serial_number_list_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int*8, ctypes.POINTER(ctypes.c_int))
get_serial_number_list = get_serial_number_list_proto(("saGetSerialNumberList", signal_hound_dll), ((2, 'serial_numbers'), (2, 'device_count')) )
#SA_API saStatus saGetSerialNumberList(int serialNumbers[8], int *deviceCount);

open_device_by_serial_number_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int)
open_device_by_serial_number = open_device_by_serial_number_proto(("saOpenDeviceBySerialNumber", signal_hound_dll), ((2, 'device'), (1, 'serial_number')) )
#SA_API saStatus saOpenDeviceBySerialNumber(int *device, int serialNumber);

open_device_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_int))
open_device = open_device_proto (("saOpenDevice", signal_hound_dll), ((2, 'device'),) )
#SA_API saStatus saOpenDevice(int *device);

close_device_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int)
close_device = close_device_proto (("saCloseDevice", signal_hound_dll), ((1, 'device'),) )
#SA_API saStatus saCloseDevice(int device);

preset = close_device_proto (("saPreset", signal_hound_dll), ((1, 'device'),) )
#SA_API saStatus saPreset(int device);

set_cal_file_path_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_char_p)
set_cal_file_path = set_cal_file_path_proto (("saSetCalFilePath", signal_hound_dll), ((1, 'path'),) )
#SA_API saStatus saSetCalFilePath(const char *path);

get_serial_number_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
get_serial_number = get_serial_number_proto (("saGetSerialNumber", signal_hound_dll), ((1, 'device'), (2, 'serial')))
#SA_API saStatus saGetSerialNumber(int device, int *serial);
get_device_type = get_serial_number_proto (("saGetDeviceType", signal_hound_dll), ((1, 'device'), (2, 'device_type')))
#SA_API saStatus saGetDeviceType(int device, saDeviceType *device_type);

get_firmware_string_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_char*16)
get_firmware_string = get_firmware_string_proto (("saGetFirmwareString", signal_hound_dll), ((1, 'device'), (2, 'firmware_string')))
#SA_API saStatus saGetFirmwareString(int device, char firmwareString[16]);

config_acquisition_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
config_acquisition = config_acquisition_proto (("saConfigAcquisition", signal_hound_dll), ((1, 'device'), (1, 'detector'), (1, 'scale')))
#SA_API saStatus saConfigAcquisition(int device, int detector, int scale);

config_center_span_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double)
config_center_span = config_center_span_proto (("saConfigCenterSpan", signal_hound_dll), ((1, 'device'), (1, 'center'), (1, 'span')))
#SA_API saStatus saConfigCenterSpan(int device, double center, double span);

config_level_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_double)
config_level = config_level_proto (("saConfigLevel", signal_hound_dll), ((1, 'device'), (1, 'ref')))
#SA_API saStatus saConfigLevel(int device, double ref);

config_gain_atten_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool)
config_gain_atten = config_gain_atten_proto(("saConfigGainAtten", signal_hound_dll), ((1, 'device'), (1, 'atten'), (1, 'gain'), (1, 'preamp')))
#SA_API saStatus saConfigGainAtten(int device, int atten, int gain, bool preAmp);

config_sweep_coupling_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_bool)
config_sweep_coupling = config_sweep_coupling_proto(("saConfigSweepCoupling", signal_hound_dll), ((1, 'device'), (1, 'rbw'), (1, 'vbw'), (1, 'reject')))
#SA_API saStatus saConfigSweepCoupling(int device, double rbw, double vbw, bool reject);

config_rbw_shape_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)
config_rbw_shape = config_rbw_shape_proto(("saConfigRBWShape", signal_hound_dll), ((1, 'device'), (1, 'rbw_shape')))
#SA_API saStatus saConfigRBWShape(int device, int rbwShape);

config_proc_units_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)
config_proc_units = config_proc_units_proto(("saConfigProcUnits", signal_hound_dll), ((1, 'device'), (1, 'units')))
#SA_API saStatus saConfigProcUnits(int device, int units);

config_IQ_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double)
config_IQ = config_IQ_proto(("saConfigIQ", signal_hound_dll), ((1, 'device'), (1, 'decimation'), (1, 'bandwidth')))
#SA_API saStatus saConfigIQ(int device, int decimation, double bandwidth);

config_audio_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,\
										ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
config_audio = config_audio_proto(("saConfigAudio", signal_hound_dll), ((1, 'device'), (1, 'audioType'), (1, 'center_freq'), \
																		(1, 'bandwidth'), (1, 'audio_low_pass_freq'), \
																		(1, 'audio_high_pass_freq'), (1, 'fm_deemphasis')))
#SA_API saStatus saConfigAudio(int device, int audioType, double centerFreq,
#                              double bandwidth, double audioLowPassFreq,
#                              double audioHighPassFreq, double fmDeemphasis);

config_real_time_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int)
config_real_time = config_real_time_proto(("saConfigRealTime", signal_hound_dll), ((1, 'device'), (1, 'frame_scale'), (1, 'frame_rate')))
#SA_API saStatus saConfigRealTime(int device, double frameScale, int frameRate);

config_real_time_overlap_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_double)
config_real_time_overlap = config_real_time_overlap_proto(("saConfigRealTimeOverlap", signal_hound_dll), ((1, 'device'), (1, 'advance_rate')))
#SA_API saStatus saConfigRealTimeOverlap(int device, double advanceRate);

set_timebase_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)
set_timebase = set_timebase_proto(("saSetTimebase", signal_hound_dll), ((1, 'device'), (1, 'timebase')))
#SA_API saStatus saSetTimebase(int device, int timebase);

initiate_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
initiate = initiate_proto(("saInitiate", signal_hound_dll), ((1, 'device'), (1, 'mode'), (1, 'flag')))
#SA_API saStatus saInitiate(int device, int mode, int flag);

abort_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int)
abort = abort_proto(("saAbort", signal_hound_dll), ((1, 'device'),))
#SA_API saStatus saAbort(int device);

query_sweep_info_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), \
											ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))
query_sweep_info = query_sweep_info_proto(("saQuerySweepInfo", signal_hound_dll), ((1, 'device'), (2, 'sweep_length'), \
																					(2, 'start_freq'), (2, 'bin_size')))
#SA_API saStatus saQuerySweepInfo(int device, int *sweepLength, double *startFreq, double *binSize);

query_stream_info_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), \
											ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))
query_stream_info = query_stream_info_proto(("saQueryStreamInfo", signal_hound_dll), ((1, 'device'), (2, 'retrun_len'), \
																					(2, 'bandwidth'), (2, 'samples_per_second')))
#SA_API saStatus saQueryStreamInfo(int device, int *returnLen, double *bandwidth, double *samplesPerSecond);

query_real_time_frame_info_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
query_real_time_frame_info = query_real_time_frame_info_proto(("saQueryRealTimeFrameInfo", signal_hound_dll), \
									((1, 'device'), (2, 'device'), (2, 'device')))
#SA_API saStatus saQueryRealTimeFrameInfo(int device, int *frameWidth, int *frameHeight);

query_real_time_poi_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double))
query_real_time_poi = query_real_time_poi_proto(("saQueryRealTimePoi", signal_hound_dll), ((1, 'device'), (2, 'poi')))
#SA_API saStatus saQueryRealTimePoi(int device, double *poi);

get_sweep_32f_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))
get_sweep_32f = get_sweep_32f_proto(("saGetSweep_32f", signal_hound_dll), ((1, 'device'), (2, 'min'), (2, 'max')))
#SA_API saStatus saGetSweep_32f(int device, float *min, float *max);

get_sweep_64f_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))
get_sweep_64f = get_sweep_64f_proto(("saGetSweep_64f", signal_hound_dll), ((1, 'device'), (2, 'min'), (2, 'max')))
#SA_API saStatus saGetSweep_64f(int device, double *min, double *max);

get_partial_sweep_32f_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),\
												ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
get_partial_sweep_32f = get_partial_sweep_32f_proto(("saGetPartialSweep_32f", signal_hound_dll), \
									((1, 'device'), (1, 'min'), (1, 'max'), (2, 'start'), (2, 'stop')))
#SA_API saStatus saGetPartialSweep_32f(int device, float *min, float *max, int *start, int *stop);

get_partial_sweep_64f_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),\
												ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
get_partial_sweep_64f = get_partial_sweep_64f_proto(("saGetPartialSweep_64f", signal_hound_dll), \
									((1, 'device'), (1, 'min'), (1, 'max'), (2, 'start'), (2, 'stop')))
#SA_API saStatus saGetPartialSweep_64f(int device, double *min, double *max, int *start, int *stop);

get_real_time_frame_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))
get_real_time_frame = get_real_time_frame_proto(("saGetRealTimeFrame", signal_hound_dll), ((1, 'device'), (1, 'sweep'), (1, 'frame')))
#SA_API saStatus saGetRealTimeFrame(int device, float *sweep, float *frame);

get_iq_32f_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float))
get_iq_32f = get_iq_32f_proto(("saGetIQ_32f", signal_hound_dll), ((1, 'device'), (1, 'iq')))
#SA_API saStatus saGetIQ_32f(int device, float *iq);

get_iq_64f_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double))
get_iq_64f = get_iq_64f_proto(("saGetIQ_64f", signal_hound_dll), ((1, 'device'), (1, 'iq')))
#SA_API saStatus saGetIQ_64f(int device, double *iq);

class sa_iq_packet(ctypes.Structure):
	_fields_ = [('iq_data', ctypes.POINTER(ctypes.c_float)),
				('iq_count', ctypes.c_int),
				('purge', ctypes.c_int),
				('data_remaining', ctypes.c_int),
				('sample_loss', ctypes.c_int),
				('sec', ctypes.c_int),
				('milli', ctypes.c_int)]
#typedef struct saIQPacket {
#    float *iqData;
#    int iqCount;
#    int purge;
#    int dataRemaining;
#    int sampleLoss;
#    int sec;
#    int milli;
#} saIQPacket;

get_iq_data_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(sa_iq_packet))
get_iq_data = get_iq_data_proto(("saGetIQData", signal_hound_dll), ((1, 'device'),(1, 'pkt')))
#SA_API saStatus saGetIQData(int device, saIQPacket *pkt);

get_iq_data_unpacked_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, \
									ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), \
									ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
get_iq_data_unpacked = get_iq_data_unpacked_proto(("saGetIQDataUnpacked", signal_hound_dll), ((1, 'device'), (1, 'iq_data'), (1, 'iq_count') ,\
									(1, 'purge'), (2, 'data_remaining'), (2, 'sample_loss'), (2, 'sec'), (2, 'milli')))
#SA_API saStatus saGetIQDataUnpacked(int device, float *iqData, int iqCount, int purge,
#                                    int *dataRemaining, int *sampleLoss, int *sec, int *milli);

get_audio_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float))
get_audio = get_audio_proto(("saGetAudio", signal_hound_dll), ((1, 'device'), (1, 'audio')))
#SA_API saStatus saGetAudio(int device, float *audio);

query_temperature_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float))
query_temperature = query_temperature_proto(("saQueryTemperature", signal_hound_dll), ((1, 'device'), (2, 'temp')))
#SA_API saStatus saQueryTemperature(int device, float *temp);

query_diagnostics = query_temperature_proto(("saQueryDiagnostics", signal_hound_dll), ((1, 'device'), (2, 'voltage')))
#SA_API saStatus saQueryDiagnostics(int device, float *voltage);

attach_tg_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int)
attach_tg = attach_tg_proto(("saAttachTg", signal_hound_dll), ((1, 'device'),))
#SA_API saStatus saAttachTg(int device);

is_tg_attached_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_bool))
is_tg_attached = is_tg_attached_proto(("saIsTgAttached", signal_hound_dll), ((1, 'device'),(2, 'attached')))
#SA_API saStatus saIsTgAttached(int device, bool *attached);

config_tg_sweep_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool)
config_tg_sweep = config_tg_sweep_proto(("saConfigTgSweep", signal_hound_dll), ((1, 'device'), (1, 'sweep_size'), \
									(1, 'high_dynamic_range'), (1, 'passive_device')))
#SA_API saStatus saConfigTgSweep(int device, int sweepSize, bool highDynamicRange, bool passiveDevice);

store_tg_thru_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)
store_tg_thru = store_tg_thru_proto(("saStoreTgThru", signal_hound_dll), ((1, 'device'), (1, 'flag')))
#SA_API saStatus saStoreTgThru(int device, int flag);

set_tg_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double)
set_tg = set_tg_proto(("saSetTg", signal_hound_dll), ((1, 'device'), (1, 'frequency'), (1, 'amplitude')))
#SA_API saStatus saSetTg(int device, double frequency, double amplitude);

set_tg_reference_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)
set_tg_reference = set_tg_reference_proto(("saSetTgReference", signal_hound_dll), ((1, 'device'), (1, 'reference')))
#SA_API saStatus saSetTgReference(int device, int reference);

get_tg_freq_ampl_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))
get_tg_freq_ampl = get_tg_freq_ampl_proto(("saGetTgFreqAmpl", signal_hound_dll), ((1, 'device'), (2, 'frequency'), (2, 'amplitude')))
#SA_API saStatus saGetTgFreqAmpl(int device, double *frequency, double *amplitude);

config_if_output_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int)
config_if_output = config_if_output_proto(("saConfigIFOutput", signal_hound_dll), ((1, 'device'), (1, 'input_freq'), (1, 'output_freq'), \
									(1, 'input_atten'), (1, 'output_gain')))
#SA_API saStatus saConfigIFOutput(int device, double inputFreq, double outputFreq,
#                                 int inputAtten, int outputGain);

class self_test_results(ctypes.Structure):
	_fields_ = [('high_band_mixer', ctypes.c_bool),
				('low_band_mixer', ctypes.c_bool),
				('attenuator', ctypes.c_bool),
				('second_if', ctypes.c_bool),
				('preamplifier', ctypes.c_bool),
				('high_band_mixer_value', ctypes.c_double),
				('low_band_mixer_value', ctypes.c_double),
				('attenuator_value', ctypes.c_double),
				('second_if_value', ctypes.c_double),
				('preamplifier_value', ctypes.c_double)]
#typedef struct saSelfTestResults {
#    // Pass/Fail
#    bool highBandMixer, lowBandMixer;
#    bool attenuator, secondIF, preamplifier;
#    // Readings
#    double highBandMixerValue, lowBandMixerValue;
#    double attenuatorValue, secondIFValue, preamplifierValue;
#} saSelfTestResults;

self_test_proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.POINTER(self_test_results))
self_test = self_test_proto(("saSelfTest", signal_hound_dll), ((1, 'device'), (2, 'results')))
#SA_API saStatus saSelfTest(int device, saSelfTestResults *results);

get_api_version = ctypes.WINFUNCTYPE(ctypes.c_char_p)(("saGetAPIVersion", signal_hound_dll), ())
#SA_API const char* saGetAPIVersion();
get_product_id = ctypes.WINFUNCTYPE(ctypes.c_char_p)(("saGetProductID", signal_hound_dll), ())
#SA_API const char* saGetProductID();
get_error_string = ctypes.WINFUNCTYPE(ctypes.c_char_p, ctypes.c_int)(("saGetErrorString", signal_hound_dll), ((1, 'code'),))
#SA_API const char* saGetErrorString(saStatus code);
