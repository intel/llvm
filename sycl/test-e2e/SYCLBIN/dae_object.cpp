// REQUIRES: aspect-usm_device_allocations

// UNSUPPORTED: cuda, hip
// UNSUPPORTED-INTENDED: CUDA and HIP targets produce only native device
// binaries and can therefore not produce object-state SYCLBIN files.

// -- Test for using a kernel from a SYCLBIN with a dead argument.

// RUN: %clangxx --offload-new-driver -fsyclbin=object %{sycl_target_opts} %S/Inputs/dae_kernel.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_OBJECT_STATE

#include "Inputs/dae.hpp"
