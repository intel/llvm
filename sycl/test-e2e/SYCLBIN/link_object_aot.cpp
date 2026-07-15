// REQUIRES: ocloc
// REQUIRES: level_zero
// REQUIRES: aspect-usm_shared_allocations

// UNSUPPORTED: cuda, hip, cpu, opencl
// UNSUPPORTED-INTENDED: CUDA and HIP targets produce only native device
// binaries and can therefore not produce object-state SYCLBIN files. The
// CPU device cannot consume the spir64_gen AOT image produced by this test.
// Native-AOT cross-image link is currently only plumbed for Level Zero (via
// the -library-compilation compile flag / zeModuleDynamicLink); the OpenCL
// CPU/GPU AOT cross-image link uses a different mechanism (-create-library)
// that is not yet implemented, so all OpenCL devices are excluded.

// -- Link two AOT-only object-state SYCLBINs. Unlike link_object.cpp (JIT
// -- spir64) this compiles for a native AOT target (spir64_gen), so the
// -- object-state bundles carry native device code images with unresolved
// -- imported symbols rather than SPIR-V. sycl::link must then route these
// -- through the native-AOT partition in kernel_bundle_impl (they cannot go
// -- through urProgramLinkExp, which requires SPIR-V): each AOT image is
// -- built independently with ALLOW_UNRESOLVED_SYMBOLS via
// -- ProgramManager::build and the cross-image references are resolved by
// -- dynamicLink(). Regression coverage for #22196 (CMPLRLLVM-75983).
//
// -- The exporting bundle provides TestFunc; the importing bundle defines
// -- TestKernel1 which imports it. Linking both and launching TestKernel1
// -- exercises the full AOT object-state link-and-run path.

// RUN: %clangxx --offload-new-driver -fsyclbin=object \
// RUN:   -fsycl-targets=%{intel_gpu_aot_targets} \
// RUN:   %S/Inputs/exporting_function.cpp -o %t.export.syclbin
// RUN: %clangxx --offload-new-driver -fsyclbin=object \
// RUN:   -fsycl-targets=%{intel_gpu_aot_targets} \
// RUN:   %S/Inputs/importing_kernel.cpp -o %t.import.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.export.syclbin %t.import.syclbin

#define SYCLBIN_OBJECT_STATE

#include "Inputs/link.hpp"
