// REQUIRES: aspect-usm_device_allocations && aspect-ext_oneapi_ipc_memory
// UNSUPPORTED: level_zero && windows
// UNSUPPORTED-TRACKER: UMFW-348

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that IPC-shared USM device memory can be accessed by graphs in both
// the exporting and importing process, using the command-buffer path.

#include "../Inputs/ipc_memory.cpp"
