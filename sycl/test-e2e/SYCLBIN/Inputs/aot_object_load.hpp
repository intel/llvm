#include "common.hpp"

// Load-time regression test for CMPLRLLVM-75983: an AOT-only SYCLBIN
// produced with -fsyclbin=object whose contents have unresolved imported
// symbols must surface a non-empty kernel_bundle when loaded as
// bundle_state::object. Before the SYCLBINBinaries::getBestCompatibleImages
// fix, the selector skipped native images for any non-executable request
// and the bundle came back empty, which broke any subsequent sycl::link.

int main(int argc, char *argv[]) {
  assert(argc == 2);

  sycl::queue Q;

  int Failed = 0;

  // Stage 1: invalid-state load checks. CommonLoadCheck attempts to load
  // the SYCLBIN in every state other than the expected one and fails if
  // any of those calls did not throw. A non-zero result from this stage
  // means the SYCLBIN's declared state does not match SYCLBIN_*_STATE
  // (typically a producer-side or test-setup issue), not the selector
  // bug under test below.
  if (int F = CommonLoadCheck(Q.get_context(), argv[1])) {
    std::cout << "Stage 1 (CommonLoadCheck): failed with " << F
              << " unexpected successes.\n";
    Failed += F;
  }

  // Stage 2: actual regression check. Load the SYCLBIN in object state and
  // verify the bundle is non-empty. Before the
  // SYCLBINBinaries::getBestCompatibleImages fix, the selector skipped
  // native images for any non-executable request, so an AOT-only object
  // SYCLBIN came back as an empty kernel_bundle.
  auto KBObj = syclexp::get_kernel_bundle<sycl::bundle_state::object>(
      Q.get_context(), std::string{argv[1]});
  if (KBObj.empty()) {
    std::cout << "Stage 2 (selector): AOT object kernel_bundle is "
                 "unexpectedly empty - getBestCompatibleImages skipped the "
                 "native image.\n";
    ++Failed;
  }

  return Failed;
}
