// Test checks that compile options are transferred to backend in the New
// Offloading Model. We use -ftarget-register-alloc-mode and
// -ftarget-compile-fast clang options for the testing here.

// REQUIRES: arch-intel_gpu_pvc

// RUN: %{build} --offload-new-driver -ftarget-register-alloc-mode=pvc:auto -o %t_with.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t_with.out 2>&1 | FileCheck --check-prefix=CHECK-OPT %s

// CHECK-OPT: <--- urProgramBuildExp(
// CHECK-SAME-OPT: -ze-intel-enable-auto-large-GRF-mode

// RUN: %{build} --offload-new-driver -Wno-error=unused-command-line-argument -ftarget-compile-fast -o %t_with.out

// RUN: env SYCL_UR_TRACE=2 %{run} %t_with.out 2>&1 | FileCheck --check-prefix=CHECK-INTEL-GPU-WITH %s

// CHECK-INTEL-GPU-WITH: <--- urProgramBuild
// CHECK-INTEL-GPU-WITH-SAME: -igc_opts 'PartitionUnit=1,SubroutineThreshold=50000'

#include <sycl/detail/core.hpp>

int main() {
  sycl::buffer<size_t, 1> Buffer(4);

  sycl::queue Queue;

  sycl::range<1> NumOfWorkItems{Buffer.size()};

  Queue.submit([&](sycl::handler &cgh) {
    sycl::accessor Accessor{Buffer, cgh, sycl::write_only};
    cgh.parallel_for<class FillBuffer>(NumOfWorkItems, [=](sycl::id<1> WIid) {
      Accessor[WIid] = WIid.get(0);
    });
  });

  sycl::host_accessor HostAccessor{Buffer, sycl::read_only};

  bool MismatchFound = false;
  for (size_t I = 0; I < Buffer.size(); ++I) {
    if (HostAccessor[I] != I) {
      std::cout << "The result is incorrect for element: " << I
                << " , expected: " << I << " , got: " << HostAccessor[I]
                << std::endl;
      MismatchFound = true;
    }
  }

  return MismatchFound;
}
