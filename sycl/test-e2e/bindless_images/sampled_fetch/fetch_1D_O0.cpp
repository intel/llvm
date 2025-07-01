// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_bindless_sampled_image_fetch_1d
// UNSUPPORTED: target-amd
// UNSUPPORTED-INTENDED: Sampled fetch not currently supported on AMD

// RUN: %{build} %O0 -o %t.out
// RUN: %{run} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

#include "fetch_1D.hpp"

int main() { return test(); }
