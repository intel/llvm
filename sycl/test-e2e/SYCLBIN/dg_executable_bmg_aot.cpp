// REQUIRES: aspect-usm_device_allocations, ocloc, arch-intel_gpu_bmg_g21

// -- Test for using device globals in SYCLBIN using only AOT target for BMG
// G21.

// UNSUPPORTED: opencl && gpu
// UNSUPPORTED-TRACKER: GSD-4287

// UNSUPPORTED: cuda
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/19533

// RUN: %clangxx --offload-new-driver -fsyclbin=executable -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g21" %S/Inputs/dg_kernel.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_EXECUTABLE_STATE

#include "Inputs/dg.hpp"
