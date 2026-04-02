// REQUIRES: aspect-usm_shared_allocations

// -- Test for linking two SYCLBIN kernel_bundle with different optimization
// -- levels.

// ptxas currently fails to compile images with unresolved symbols. Disable for
// other targets than SPIR-V until this has been resolved. (CMPLRLLVM-68810)
// Note: %{sycl_target_opts} should be added to the SYCLBIN compilation lines
// once fixed.
// REQUIRES: target-spir

// RUN: %clangxx --offload-new-driver -fsyclbin=object -fsycl-allow-device-image-dependencies %if cl_options %{/Od%} %else %{-O0%} %S/Inputs/exporting_function.cpp -o %t.export.syclbin
// RUN: %clangxx --offload-new-driver -fsyclbin=object -fsycl-allow-device-image-dependencies -O1 %S/Inputs/importing_kernel.cpp -o %t.import.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.export.syclbin %t.import.syclbin

#define SYCLBIN_OBJECT_STATE

#include "Inputs/link.hpp"
