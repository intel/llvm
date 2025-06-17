// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_bindless_sampled_image_fetch_1d
// XFAIL: level_zero && windows
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/18919

// RUN: %{build} %O0 -o %t.out
// RUN: %{run} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

#include "fetch_1D.hpp"

int main() { return test(); }
