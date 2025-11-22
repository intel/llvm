// REQUIRES: aspect-usm_device_allocations, ocloc, arch-intel_gpu_bmg_g21

// -- Test for compiling and loading a kernel bundle with a SYCLBIN containing
//    the use of optional kernel features using only AOT target for BMG G21.

// RUN: %clangxx --offload-new-driver -fsyclbin=executable -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g21" %S/Inputs/optional_kernel_features.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_EXECUTABLE_STATE

#include "Inputs/optional_kernel_features.hpp"
