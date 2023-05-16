// UNSUPPORTED: hip
// UNSUPPORTED: ze_debug-1,ze_debug4

// RUN: %{build} -fno-sycl-early-optimizations -fsycl-device-code-split=per_kernel -o %t_per_kernel.out
// RUN: %{build} -fno-sycl-early-optimizations -fsycl-device-code-split=per_source -o %t_per_source.out
// RUN: %{build} -fno-sycl-early-optimizations -fsycl-device-code-split=off -o %t_off.out
// RUN: %{build} -fno-sycl-early-optimizations -fsycl-device-code-split=auto -o %t_auto.out
// RUN: %{run} %t_per_kernel.out | FileCheck %s

// RUN: %{run} %t_per_source.out | FileCheck %s

// RUN: %{run} %t_auto.out | FileCheck %s

// RUN: %{run} %t_off.out | FileCheck %s

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

// This function is used by two different kernels.
// We want to ensure that it does not lead so multiple symbol collision
// when building an executable via sycl::compile and sycl::link.
int foo(int a) { return a + 1; }

template <int ID> class kernel_name {};

int main() {
  try {
    queue q;
    auto input = get_kernel_bundle<bundle_state::input>(
        q.get_context(), {q.get_device()},
        {get_kernel_id<kernel_name<0>>(), get_kernel_id<kernel_name<1>>()});

    auto compiled = sycl::compile(input);
    kernel_bundle<bundle_state::executable> linked = sycl::link(compiled);

    buffer<int> b(range{1});
    q.submit([&](handler &cgh) {
      cgh.use_kernel_bundle(linked);
      auto acc = b.get_access<access_mode::read_write>(cgh);
      cgh.single_task<kernel_name<0>>([=]() { acc[0] = foo(acc[0]); });
    });

    q.submit([&](handler &cgh) {
      auto acc = b.get_access<access_mode::read_write>(cgh);
      cgh.single_task<kernel_name<1>>([=]() { acc[0] = foo(acc[0]); });
    });
  } catch (exception &e) {
    std::cout << "Exception: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "OK";
  return 0;
}

//CHECK: OK