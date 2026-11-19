// REQUIRES: level_zero_v2_adapter && (arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31 || arch-intel_gpu_cri)
// REQUIRES: aspect-usm_device_allocations && aspect-ext_oneapi_ipc_memory
// UNSUPPORTED: level_zero && windows
// UNSUPPORTED-TRACKER: UMFW-348

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that IPC-shared USM device memory can be accessed by graphs in both
// the exporting and importing process, using native recording.

#define GRAPH_E2E_NATIVE_RECORDING

#include "../../Inputs/ipc_memory.cpp"
