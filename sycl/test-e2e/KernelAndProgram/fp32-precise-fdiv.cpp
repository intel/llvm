// RUN: %{build} -Wno-error=unused-command-line-argument -foffload-fp32-prec-div -foffload-fp32-prec-sqrt -o %t_with.out
// RUN: %{build} -Wno-error=unused-command-line-argument -foffload-fp32-prec-div -o %t_with_div.out
// RUN: %{build} -Wno-error=unused-command-line-argument -foffload-fp32-prec-sqrt -o %t_with_sqrt.out
// RUN: %{build} -o %t_without.out

// RUN: env SYCL_UR_TRACE=2 %{run} %t_with.out 2>&1 | FileCheck %if hip || cuda %{ --check-prefix=CHECK-WITHOUT %} %else %{ --check-prefix=CHECK-WITH %} %s
// RUN: env SYCL_UR_TRACE=2 %{run} %t_with_div.out 2>&1 | FileCheck %if hip || cuda %{ --check-prefix=CHECK-WITHOUT %} %else %{ --check-prefix=CHECK-WITH %} %s
// RUN: env SYCL_UR_TRACE=2 %{run} %t_with_sqrt.out 2>&1 | FileCheck %if hip || cuda %{ --check-prefix=CHECK-WITHOUT %} %else %{ --check-prefix=CHECK-WITH %} %s
// RUN: env SYCL_UR_TRACE=2 %{run} %t_without.out 2>&1 | FileCheck --implicit-check-not=fp32-correctly-rounded-divide-sqrt %s

// CHECK-WITH: <--- urProgramBuild
// CHECK-WITH-SAME: fp32-correctly-rounded-divide-sqrt

// CHECK-WITHOUT-NOT: <--- urProgramBuild{{.*}}fp32-correctly-rounded-divide-sqrt{{.*}} -> UR_RESULT_SUCCESS
// CHECK-WITHOUT: <--- urProgramBuild{{.*}} -> UR_RESULT_SUCCESS

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
