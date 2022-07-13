// UNSUPPORTED: cuda || hip
// UNSUPPORTED: ze_debug-1,ze_debug4

// RUN: %clangxx -fsycl -fno-sycl-early-optimizations -fsycl-device-code-split=per_kernel %s -o %t_per_kernel.out
// RUN: %clangxx -fsycl -fno-sycl-early-optimizations -fsycl-device-code-split=per_source %s -o %t_per_source.out
// RUN: %clangxx -fsycl -fno-sycl-early-optimizations -fsycl-device-code-split=off %s -o %t_off.out
// RUN: %clangxx -fsycl -fno-sycl-early-optimizations -fsycl-device-code-split=auto %s -o %t_auto.out
// RUN: %GPU_RUN_PLACEHOLDER %t_per_kernel.out %GPU_CHECK_PLACEHOLDER
// RUN: %CPU_RUN_PLACEHOLDER %t_per_kernel.out %CPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t_per_kernel.out %ACC_CHECK_PLACEHOLDER

// RUN: %GPU_RUN_PLACEHOLDER %t_per_source.out %GPU_CHECK_PLACEHOLDER
// RUN: %CPU_RUN_PLACEHOLDER %t_per_source.out %CPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t_per_source.out %ACC_CHECK_PLACEHOLDER

// RUN: %GPU_RUN_PLACEHOLDER %t_auto.out %GPU_CHECK_PLACEHOLDER
// RUN: %CPU_RUN_PLACEHOLDER %t_auto.out %CPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t_auto.out %ACC_CHECK_PLACEHOLDER

// RUN: %GPU_RUN_PLACEHOLDER %t_off.out %GPU_CHECK_PLACEHOLDER
// RUN: %CPU_RUN_PLACEHOLDER %t_off.out %CPU_CHECK_PLACEHOLDER
// RUN: %ACC_RUN_PLACEHOLDER %t_off.out %ACC_CHECK_PLACEHOLDER

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