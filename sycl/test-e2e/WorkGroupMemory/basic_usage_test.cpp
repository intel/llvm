// Non-pointer types version of the basic usage test.

// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/17339

// XFAIL: spirv-backend
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/18230

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/queue.hpp>

sycl::queue q;

#include "./basic_usage_common.hpp"

int main() {
  test<float>();
  test<int>();
  test<char>();
  test<uint16_t>();
  if (q.get_device().has(sycl::aspect::fp16))
    test<sycl::half>();
  return 0;
}
