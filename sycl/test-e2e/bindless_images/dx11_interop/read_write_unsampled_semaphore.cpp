// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: aspect-ext_oneapi_external_semaphore_import
// REQUIRES: windows

// UNSUPPORTED: gpu-intel-gen12
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/22148

// UNSUPPORTED: gpu-intel-dg2
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/22155

// RUN: %{build} %link-directx -o %t.out
// RUN: %{run-unfiltered-devices} %t.out

#define TEST_SEMAPHORE_IMPORT
#include "read_write_unsampled.cpp"
