// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "native-events.hpp"

void test_empty_host_task_with_manual_interop_sync_property() {
  sycl::queue{}.submit([&](sycl::handler &cgh) {
    cgh.host_task([&](sycl::interop_handle ih) {}, {manual_interop_sync{}});
  });
}

int main() { test_empty_host_task_with_manual_interop_sync_property(); }
