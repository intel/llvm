// REQUIRES: native_cpu_ock

// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -mllvm -inline-threshold=500 -mllvm -sycl-native-cpu-no-vecz -mllvm -sycl-native-dump-device-ir %s | FileCheck %s

// Check that local types structure is created and placed on the stack
// We also check that the attribute mux-orig-fn is created as this is needed to
// find the original function after this pass is run

// CHECK: %localVarTypes = type { ptr addrspace(1) }
// CHECK: define void @_ZTS4TestILi1ELi4EiE.NativeCPUKernel{{.*}} #[[ATTR:[0-9]*]]
// CHECK: alloca %localVarTypes
// CHECK: attributes #[[ATTR]] = {{.*}} "mux-orig-fn"="_ZTS4TestILi1ELi4EiE"

#include <sycl/sycl.hpp>

template <int dims, int size, typename T = int> struct Test;

int main() {
  sycl::queue queue;

  constexpr int dims = 1;
  constexpr int size = 4;

  std::array<int, size> data;

  const auto range = sycl::range<dims>(size);
  const auto range_wg = sycl::range<dims>(1);
  {
    sycl::buffer<int, dims> buf(data.data(), range);

    queue.submit([&](sycl::handler &cgh) {
      auto acc = sycl::accessor(buf, cgh, sycl::write_only);
      cgh.parallel_for_work_group<Test<dims, size>>(
          range, range_wg, [=](auto group) { acc[group.get_group_id()] = 42; });
    });
    queue.wait_and_throw();
  }
  return 0;
}
