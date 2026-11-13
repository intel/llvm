// REQUIRES: level_zero_v2_adapter && level_zero_dev_kit && (arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31 || arch-intel_gpu_cri)
// REQUIRES: aspect-usm_device_allocations && aspect-ext_oneapi_ipc_memory
// REQUIRES-INTEL-DRIVER: lin: 37125
// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: UMFW-348

// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out

// Tests that USM device memory shared through the Level Zero IPC API can be
// accessed by native recording graphs in both the exporting and importing
// process.
//
// Relies on the driver's opaque IPC handle support (minimum driver version
// above), which lets the importing process pass the exporter's handle to
// zeMemOpenIpcHandle unchanged.

#define GRAPH_E2E_NATIVE_RECORDING
#define USE_ZE_IPC

#include "../../Inputs/ipc_memory.cpp"
