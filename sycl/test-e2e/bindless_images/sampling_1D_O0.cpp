// REQUIRES: aspect-ext_oneapi_bindless_images
// XFAIL: level_zero

// RUN: %{build} %O0 -o %t.out
// RUN: %{run-unfiltered-devices} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/17212

// Uncomment to print additional test information
// #define VERBOSE_PRINT

#include "sampling_1D.hpp"

int main() { return test(); }
