// REQUIRES: aspect-usm_device_allocations

// UNSUPPORTED: cuda, hip
// UNSUPPORTED-INTENDED: CUDA and HIP targets produce only native device
// binaries and can therefore not produce input-state SYCLBIN files.

// -- Test for using device globals in SYCLBIN.

// UNSUPPORTED: opencl && gpu
// UNSUPPORTED-TRACKER: GSD-4287

// RUN: %clangxx --offload-new-driver -fsyclbin=input %{sycl_target_opts} %S/Inputs/dg_kernel.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_INPUT_STATE

#include "Inputs/dg.hpp"
