// REQUIRES: ocloc

// UNSUPPORTED: cuda, hip, cpu
// UNSUPPORTED-INTENDED: CUDA and HIP targets produce only native device
// binaries and can therefore not produce object-state SYCLBIN files. The
// CPU device cannot consume the spir64_gen AOT image produced by this
// test; the CPU AOT pipeline does not currently support the
// unresolved-import / object-state path exercised here, so opencl:cpu is
// excluded.

// -- Regression test for CMPLRLLVM-75983: loading an AOT-only SYCLBIN in
// -- bundle_state::object must produce a non-empty kernel_bundle when the
// -- AOT image still carries unresolved imported symbols. The producer
// -- emits a kernel that imports a SYCL_EXTERNAL function (TestFunc)
// -- defined in a separate translation unit, so the resulting native image
// -- is in object state and was previously dropped by
// -- SYCLBINBinaries::getBestCompatibleImages.
//
// -- spir64_x86_64 is intentionally not exercised here: the Intel OpenCL CPU
// -- runtime's clBuildProgram does not accept unresolved imported symbols (no
// -- equivalent of IGC's -library-compilation), and the -cmd=compile path
// -- emits SPIR-V rather than native code, which would not exercise the
// -- native-AOT-object selector path under test. A separate test should be
// -- added once the CPU AOT pipeline grows the equivalent capability.
// -- The host-side test is load-only (no kernel launch), so it does not
// -- need the runtime that would actually execute the produced binary.

// RUN: %clangxx --offload-new-driver -fsyclbin=object \
// RUN:   -fsycl-allow-device-image-dependencies \
// RUN:   -fsycl-targets=%{intel_gpu_aot_targets} \
// RUN:   %S/Inputs/aot_object_with_imports.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_OBJECT_STATE

#include "Inputs/aot_object_load.hpp"
