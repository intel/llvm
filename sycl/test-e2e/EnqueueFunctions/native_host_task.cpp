// Only L0V2 supports urEnqueueHostTaskExp.
// REQUIRES: level_zero_v2_adapter

// UNSUPPORTED: windows && gpu-intel-gen12
// UNSUPPORTED-INTENDED: UR_DEVICE_INFO_ENQUEUE_HOST_TASK_SUPPORT_EXP is not
// supported on win&gen12.

// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s

// CHECK: UR_DEVICE_INFO_ENQUEUE_HOST_TASK_SUPPORT_EXP
// CHECK: ---> urEnqueueHostTaskExp
// CHECK: <--- urEnqueueHostTaskExp

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/properties/all_properties.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue q;

  syclex::host_task(q, [=] {});
  q.wait();
}
