// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: aspect-ext_oneapi_external_semaphore_import
// REQUIRES: windows

// UNSUPPORTED: gpu-intel-gen12
// UNSUPPORTED-INTENDED: Unknown issue with integrated GPU failing
//                       when importing memory

// RUN: %{build} %link-directx -o %t.out
// RUN: %{run-unfiltered-devices} %t.out

#define TEST_SEMAPHORE_IMPORT
#include "read_write_unsampled.cpp"
