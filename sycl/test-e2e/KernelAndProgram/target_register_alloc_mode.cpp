// REQUIRES: gpu-intel-pvc

// RUN: %{build} -ftarget-register-alloc-mode=pvc:auto -o %t_with.out
// RUN: %{build} -o %t_without.out
// RUN: %{build} -ftarget-register-alloc-mode=pvc:default -o %t_default.out

// RUN: env SYCL_PI_TRACE=-1 %{run} %t_with.out 2>&1 | FileCheck --check-prefix=CHECK-OPT %s
// RUN: env SYCL_PI_TRACE=-1 %{run} %t_without.out 2>&1 | FileCheck %if system-windows %{ --implicit-check-not=-ze-intel-enable-auto-large-GRF-mode %} %else %{ --check-prefix=CHECK-OPT %} %s
// RUN: env SYCL_PI_TRACE=-1 %{run} %t_default.out 2>&1 | FileCheck --implicit-check-not=-ze-intel-enable-auto-large-GRF-mode %s

// CHECK-OPT: ---> piProgramBuild(
// CHECK-OPT: -ze-intel-enable-auto-large-GRF-mode

#include <sycl/sycl.hpp>

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
