// REQUIRES: aspect-usm_device_allocations

// UNSUPPORTED: cuda, hip
// UNSUPPORTED-INTENDED: CUDA and HIP targets produce only native device
// binaries and can therefore not produce object-state SYCLBIN files.

// -- Test for compiling and loading a kernel bundle with a SYCLBIN containing
//    the use of optional kernel features.

// RUN: %clangxx --offload-new-driver -fsyclbin=object %{sycl_target_opts} %S/Inputs/optional_kernel_features.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_OBJECT_STATE

#include "Inputs/optional_kernel_features.hpp"
