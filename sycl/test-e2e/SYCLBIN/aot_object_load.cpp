// REQUIRES: ocloc

// -- Regression test for CMPLRLLVM-75983: loading an AOT-only SYCLBIN in
// -- bundle_state::object must produce a non-empty kernel_bundle when the
// -- AOT image still carries unresolved imported symbols. The producer
// -- emits a kernel that imports a SYCL_EXTERNAL function (TestFunc)
// -- defined in a separate translation unit, so the resulting native image
// -- is in object state and was previously dropped by
// -- SYCLBINBinaries::getBestCompatibleImages.
//
// -- The test targets spir64_gen via ocloc with -device * (%gpu_aot_target_opts)
// -- so it is not locked to a single Intel GPU architecture. spir64_x86_64
// -- is intentionally not exercised here: the Intel OpenCL CPU runtime's
// -- clBuildProgram does not accept unresolved imported symbols (no
// -- equivalent of IGC's -library-compilation), and the -cmd=compile path
// -- emits SPIR-V rather than native code, which would not exercise the
// -- native-AOT-object selector path under test. A separate test should be
// -- added once the CPU AOT pipeline grows the equivalent capability.
// -- The host-side test is load-only (no kernel launch), so it does not
// -- need the runtime that would actually execute the produced binary.

// RUN: %clangxx --offload-new-driver -fsyclbin=object \
// RUN:   -fsycl-allow-device-image-dependencies \
// RUN:   -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts \
// RUN:   %S/Inputs/aot_object_with_imports.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_OBJECT_STATE

#include "Inputs/aot_object_load.hpp"
