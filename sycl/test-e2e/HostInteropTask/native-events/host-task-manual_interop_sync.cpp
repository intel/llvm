// REQUIRES: cuda
//
// RUN: %{build} -o %t.out -lcuda
// RUN: %{run} %t.out

#include "host-task-native-events-cuda.hpp"
#include <cuda.h>
#include <sycl/sycl.hpp>

void test_empty_host_task_with_manual_interop_sync_property() {
  sycl::queue{}.submit([&](sycl::handler &cgh) {
    cgh.host_task([&](sycl::interop_handle ih) {},
                  {sycl::ext::codeplay::experimental::property::host_task::
                       manual_interop_sync{}});
  });
}

int main() { test_empty_host_task_with_manual_interop_sync_property(); }
