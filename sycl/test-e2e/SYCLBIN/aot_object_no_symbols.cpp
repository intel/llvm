// REQUIRES: ocloc
// REQUIRES: level_zero
// REQUIRES: aspect-usm_shared_allocations

// UNSUPPORTED: cuda, hip, cpu, opencl
// UNSUPPORTED-INTENDED: CUDA and HIP targets produce only native device
// binaries and can therefore not produce object-state SYCLBIN files. The CPU
// device cannot consume the spir64_gen AOT image produced by this test.
// Bringing a native-AOT object image to executable state goes through
// ProgramManager::build with ALLOW_UNRESOLVED_SYMBOLS followed by
// dynamicLink(), which is currently only plumbed for Level Zero, so all
// OpenCL devices are excluded.

// -- Load and run an AOT-only object-state SYCLBIN whose image carries
// -- neither imported nor exported symbols.
//
// -- Unlike aot_object_load.cpp (kernel with unresolved imports) and
// -- link_object_aot.cpp (importer/exporter pair), the kernel here is fully
// -- self-contained. ProgramManager::getBinImageState therefore classifies
// -- the native image as bundle_state::executable even though the SYCLBIN
// -- records object state, so SYCLBINBinaries::getBestCompatibleImages must
// -- accept it on the strength of the requested state alone: an object-state
// -- SYCLBIN's native image is its object-state content regardless of how the
// -- image itself classifies. Before that fix the selector's
// -- intrinsic-state equality gate dropped the image, the bundle loaded
// -- empty, and the following sycl::link had nothing to link.
//
// -- kernel_bundle_impl's ReconcileState then presents the image as object so
// -- it is not mistaken for already-linked, and sycl::link routes it through
// -- the native-AOT partition (build with ALLOW_UNRESOLVED_SYMBOLS +
// -- dynamicLink) to reach executable state.
//
// -- The -device * wildcard (%gpu_aot_target_opts) keeps this off a single
// -- Intel GPU architecture. Complements the hardware-independent unit test
// -- SYCLBINSelector.NativeWithoutSymbolsSurfacedForObject, which covers the
// -- selector in isolation but neither the producer nor the link-and-run path.

// RUN: %clangxx --offload-new-driver -fsyclbin=object \
// RUN:   -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts \
// RUN:   %S/Inputs/aot_object_no_symbols.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_OBJECT_STATE

#include "Inputs/common.hpp"

#include <sycl/usm.hpp>

#include <iostream>

static constexpr size_t NUM = 10;

int main(int argc, char *argv[]) {
  assert(argc == 2);

  sycl::queue Q;

  int Failed = 0;

  // Stage 1: invalid-state load checks. A non-zero result here means the
  // SYCLBIN's declared state is not object (a producer-side or test-setup
  // issue), not the selector behavior under test below.
  if (int F = CommonLoadCheck(Q.get_context(), argv[1])) {
    std::cout << "Stage 1 (CommonLoadCheck): failed with " << F
              << " unexpected successes.\n";
    Failed += F;
  }

  // Stage 2: the selector. The native image classifies as executable because
  // it has no imported symbols, but the object-state request must still
  // surface it.
  auto KBObj = syclexp::get_kernel_bundle<sycl::bundle_state::object>(
      Q.get_context(), {Q.get_device()}, std::string{argv[1]});
  if (KBObj.empty()) {
    std::cout << "Stage 2 (selector): object kernel_bundle is unexpectedly "
                 "empty - getBestCompatibleImages skipped the native image.\n";
    // Nothing to link or launch, so the remaining stages cannot run.
    return Failed + 1;
  }

  // Stage 3: object -> executable. For a native AOT image this is a build
  // with unresolved symbols allowed followed by dynamicLink, not
  // urProgramLinkExp.
  auto KBExe = syclexp::link({KBObj});
  if (!KBExe.ext_oneapi_has_kernel("TestKernelNoSyms")) {
    std::cout << "Stage 3 (link): linked bundle does not contain "
                 "TestKernelNoSyms.\n";
    return Failed + 1;
  }

  // Stage 4: launch the kernel and check the results.
  sycl::kernel TestKernelNoSyms =
      KBExe.ext_oneapi_get_kernel("TestKernelNoSyms");

  int *Ptr = sycl::malloc_shared<int>(NUM, Q);
  Q.fill(Ptr, int{0}, NUM).wait_and_throw();

  Q.submit([&](sycl::handler &CGH) {
     CGH.set_args(Ptr, int{NUM});
     CGH.single_task(TestKernelNoSyms);
   }).wait_and_throw();

  for (size_t I = 0; I < NUM; ++I) {
    if (Ptr[I] != static_cast<int>(I)) {
      std::cout << "Stage 4 (launch): Result: " << Ptr[I] << " expected " << I
                << "\n";
      ++Failed;
    }
  }

  sycl::free(Ptr, Q);
  return Failed;
}
