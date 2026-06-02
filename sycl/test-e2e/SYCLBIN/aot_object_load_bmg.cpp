// REQUIRES: aspect-usm_device_allocations, ocloc, arch-intel_gpu_bmg_g21

// -- Regression test for CMPLRLLVM-75983: loading an AOT-only SYCLBIN in
// -- bundle_state::object must produce a non-empty kernel_bundle when the
// -- AOT image still carries unresolved imported symbols. The producer
// -- emits a kernel that imports a SYCL_EXTERNAL function (TestFunc)
// -- defined in a separate translation unit, so the resulting native image
// -- is in object state and was previously dropped by
// -- SYCLBINBinaries::getBestCompatibleImages.

// RUN: %clangxx --offload-new-driver -fsyclbin=object \
// RUN:   -fsycl-allow-device-image-dependencies \
// RUN:   -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen "-device bmg-g21" \
// RUN:   %S/Inputs/aot_object_with_imports.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_OBJECT_STATE

#include "Inputs/aot_object_load.hpp"
