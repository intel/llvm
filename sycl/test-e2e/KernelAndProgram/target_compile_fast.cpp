// RUN: %{build} -ftarget-compile-fast -o %t_with.out
// RUN: %{build} -o %t_without.out

// RUN: env SYCL_PI_TRACE=-1 %{run} %t_with.out 2>&1 | FileCheck %if !gpu || hip || cuda %{ --check-prefix=CHECK-WITHOUT %} %else %{ --check-prefix=CHECK-INTEL-GPU-WITH %} %s
// RUN: env SYCL_PI_TRACE=-1 %{run} %t_without.out 2>&1 | FileCheck --implicit-check-not=-igc_opts %s

// CHECK-INTEL-GPU-WITH: ---> piProgramBuild(
// CHECK-INTEL-GPU-WITH: -igc_opts 'PartitionUnit=1,SubroutineThreshold=50000'

// CHECK-WITHOUT: ---> piProgramBuild(
// CHECK-WITHOUT-NOT: -igc_opts
// CHECK-WITHOUT: ) ---> pi_result : PI_SUCCESS

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
