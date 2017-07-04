import ctypes

labbrick_dll = ctypes.WinDLL ("vnx_fmsynth.dll")

set_test_mode_proto = ctypes.WINFUNCTYPE(None, ctypes.c_bool)
set_test_mode = set_test_mode_proto(("fnLMS_SetTestMode", labbrick_dll), ((1, 'test_mode'),) )

get_num_devices_proto = ctypes.WINFUNCTYPE (ctypes.c_int) # Return type
get_num_devices = get_num_devices_proto (("fnLMS_GetNumDevices", labbrick_dll), () )

get_model_name_unicode_proto = ctypes.WINFUNCTYPE (ctypes.c_int, ctypes.c_uint, ctypes.c_wchar_p)
get_model_name_unicode = get_model_name_unicode_proto(("fnLMS_GetModelNameW", labbrick_dll), ((1, 'deviceID'), (1, 'ModelName')))

init_device_proto = ctypes.WINFUNCTYPE (ctypes.c_int, ctypes.c_uint)
get_serial_number = init_device_proto(("fnLMS_GetSerialNumber", labbrick_dll), ((1, 'deviceID'),))
init_device = init_device_proto(("fnLMS_InitDevice", labbrick_dll), ((1, 'deviceID'),))
close_device = init_device_proto(("fnLMS_CloseDevice", labbrick_dll), ((1, 'deviceID'),))
save_settings = init_device_proto(("fnLMS_SaveSettings", labbrick_dll), ((1, 'deviceID'),))
get_frequency = init_device_proto (("fnLMS_GetFrequency", labbrick_dll), ((1, 'deviceID'),) )
get_start_frequency = init_device_proto (("fnLMS_GetStartFrequency", labbrick_dll), ((1, 'deviceID'),) )
get_end_frequency = init_device_proto (("fnLMS_GetEndFrequency", labbrick_dll), ((1, 'deviceID'),) )
get_sweep_time = init_device_proto (("fnLMS_GetSweepTime", labbrick_dll), ((1, 'deviceID'),) )
get_rf_on = init_device_proto (("fnLMS_GetRF_On", labbrick_dll), ((1, 'deviceID'),) )
get_use_internal_ref = init_device_proto (("fnLMS_GetUseInternalRef", labbrick_dll), ((1, 'deviceID'),) )
get_power_level = init_device_proto (("fnLMS_GetPowerLevel", labbrick_dll), ((1, 'deviceID'),) )
get_abs_power_level = init_device_proto (("fnLMS_GetAbsPowerLevel", labbrick_dll), ((1, 'deviceID'),) )
get_max_pwr = init_device_proto (("fnLMS_GetMaxPwr", labbrick_dll), ((1, 'deviceID'),) )
get_min_pwr = init_device_proto (("fnLMS_GetMinPwr", labbrick_dll), ((1, 'deviceID'),) )
get_max_freq = init_device_proto (("fnLMS_GetMaxFreq", labbrick_dll), ((1, 'deviceID'),) )
get_min_freq = init_device_proto (("fnLMS_GetMinFreq", labbrick_dll), ((1, 'deviceID'),) )
get_pulse_mode = init_device_proto (("fnLMS_GetPulseMode", labbrick_dll), ((1, 'deviceID'),) )
get_has_fast_pulse_mode = init_device_proto (("fnLMS_GetHasFastPulseMode", labbrick_dll), ((1, 'deviceID'),) )
get_use_internal_pulse_mode = init_device_proto (("fnLMS_GetUseInternalPulseMod", labbrick_dll), ((1, 'deviceID'),) )
get_device_status = init_device_proto (("fnLMS_GetDeviceStatus", labbrick_dll), ((1, 'deviceID'),) )

get_pulse_on_time_proto = ctypes.WINFUNCTYPE (ctypes.c_float, ctypes.c_uint) # Return type
get_pulse_on_time = get_pulse_on_time_proto(("fnLMS_GetPulseOnTime", labbrick_dll), ((1, 'deviceID'),))
get_pulse_off_time = get_pulse_on_time_proto(("fnLMS_GetPulseOffTime", labbrick_dll), ((1, 'deviceID'),))

set_frequency_proto = ctypes.WINFUNCTYPE (ctypes.c_int, ctypes.c_uint, ctypes.c_int)
set_frequency = set_frequency_proto(("fnLMS_SetFrequency", labbrick_dll), ((1, 'deviceID'), (1, 'frequency')))
set_start_frequency = set_frequency_proto(("fnLMS_SetStartFrequency", labbrick_dll), ((1, 'deviceID'), (1, 'start_frequency')))
set_end_frequency = set_frequency_proto(("fnLMS_SetEndFrequency", labbrick_dll), ((1, 'deviceID'), (1, 'end_frequency')))
set_sweep_time = set_frequency_proto(("fnLMS_SetSweepTime", labbrick_dll), ((1, 'deviceID'), (1, 'sweep_time')))
set_power_level = set_frequency_proto(("fnLMS_SetPowerLevel", labbrick_dll), ((1, 'deviceID'), (1, 'power_level')))
set_pulse_on_time = set_frequency_proto(("fnLMS_SetPulseOnTime", labbrick_dll), ((1, 'deviceID'), (1, 'pulse_on_time')))
set_pulse_off_time = set_frequency_proto(("fnLMS_SetPulseOffTime", labbrick_dll), ((1, 'deviceID'), (1, 'pulse_off_time')))

set_rf_on_proto = ctypes.WINFUNCTYPE (ctypes.c_int, ctypes.c_uint, ctypes.c_bool)
set_rf_on = set_rf_on_proto(("fnLMS_SetRFOn", labbrick_dll), ((1, 'deviceID'), (1, 'on')))
enable_internal_pulse_mod = set_rf_on_proto(("fnLMS_EnableInternalPulseMod", labbrick_dll), ((1, 'deviceID'), (1, 'on')))
set_use_external_pulse_mod = set_rf_on_proto(("fnLMS_SetUseExternalPulseMod", labbrick_dll), ((1, 'deviceID'), (1, 'external')))
set_use_internal_ref = set_rf_on_proto(("fnLMS_SetUseInternalRef", labbrick_dll), ((1, 'deviceID'), (1, 'internal')))
set_sweep_direction = set_rf_on_proto(("fnLMS_SetSweepDirection", labbrick_dll), ((1, 'deviceID'), (1, 'up')))
set_sweep_mode = set_rf_on_proto(("fnLMS_SetSweepMode", labbrick_dll), ((1, 'deviceID'), (1, 'mode')))
set_sweep_type = set_rf_on_proto(("fnLMS_SetSweepType", labbrick_dll), ((1, 'deviceID'), (1, 'swptype')))
start_sweep = set_rf_on_proto(("fnLMS_StartSweep", labbrick_dll), ((1, 'deviceID'), (1, 'go')))

set_fast_pulsed_output_proto = ctypes.WINFUNCTYPE (ctypes.c_int, ctypes.c_uint, ctypes.c_float, ctypes.c_float, ctypes.c_bool)
set_fast_pulsed_output = set_fast_pulsed_output_proto(("fnLMS_SetFastPulsedOutput", labbrick_dll), ((1, 'deviceID'), \
																									(1, 'pulse_on_time'), \
																									(1, 'pulse_rep_time'), \
																									(1, 'on')))