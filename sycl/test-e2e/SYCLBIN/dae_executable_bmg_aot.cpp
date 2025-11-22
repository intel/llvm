// REQUIRES: aspect-usm_device_allocations, ocloc, arch-intel_gpu_bmg_g21

// -- Test for using a kernel from a SYCLBIN with a dead argument using only AOT
// target for BMG G21.

// RUN: %clangxx --offload-new-driver -fsyclbin=executable -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g21" %S/Inputs/dae_kernel.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_EXECUTABLE_STATE

#include "Inputs/dae.hpp"
