// RUN: %clangxx -D__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__  \
// RUN:    -fsycl-device-only -S -Xclang -emit-llvm %s -o - |    \
// RUN:    FileCheck %s --check-prefix CHECK-IR

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace ext::oneapi::experimental;
using namespace ext::intel::experimental;

using annotated_ptr_load = annotated_ptr<
    float,
    decltype(properties(
        alignment<8>,
        read_hint<cache_control<cache_mode::cached, cache_level::L1>,
                  cache_control<cache_mode::uncached, cache_level::L2,
                                cache_level::L3>,
                  cache_control<cache_mode::invalidate, cache_level::L4>>))>;

using annotated_ptr_store = annotated_ptr<
    float,
    decltype(properties(
        write_hint<cache_control<cache_mode::write_through, cache_level::L1>,
                   cache_control<cache_mode::write_back, cache_level::L2,
                                 cache_level::L3>>))>;

void cache_control_read_func() {
  queue q;
  constexpr int N = 10;
  float *ArrayA = malloc_shared<float>(N, q);
  q.submit([&](handler &cgh) {
    cgh.parallel_for<>(range<1>(N), [=](item<1> item) {
      auto item_id = item.get_linear_id();
      annotated_ptr_load src{&ArrayA[item_id]};
      *src = 55.0f;
    });
  });
}

void cache_control_write_func() {
  queue q;
  constexpr int N = 10;
  float *ArrayA = malloc_shared<float>(N, q);
  q.submit([&](handler &cgh) {
    cgh.parallel_for<>(range<1>(N), [=](item<1> item) {
      auto item_id = item.get_linear_id();
      annotated_ptr_store dst{&ArrayA[item_id]};
      *dst = 55.0f;
    });
  });
}

// CHECK-IR: spir_kernel{{.*}}cache_control_read_func
// CHECK-IR: {{.*}}getelementptr inbounds float{{.*}}!spirv.Decorations [[RDECOR:.*]]
// CHECK-IR: ret void

// CHECK-IR: spir_kernel{{.*}}cache_control_write_func
// CHECK-IR: {{.*}}getelementptr inbounds float{{.*}}!spirv.Decorations [[WDECOR:.*]]
// CHECK-IR: ret void

// CHECK-IR: [[RDECOR]] = !{[[RDECOR1:.*]], [[RDECOR2:.*]], [[RDECOR3:.*]], [[RDECOR4:.*]]}
// CHECK-IR: [[RDECOR1]] = !{i32 6442, i32 1, i32 0}
// CHECK-IR: [[RDECOR2]] = !{i32 6442, i32 2, i32 0}
// CHECK-IR: [[RDECOR3]] = !{i32 6442, i32 0, i32 1}
// CHECK-IR: [[RDECOR4]] = !{i32 6442, i32 3, i32 3}

// CHECK-IR: [[WDECOR]] = !{[[WDECOR1:.*]], [[WDECOR2:.*]], [[WDECOR3:.*]]}
// CHECK-IR: [[WDECOR1]] = !{i32 6443, i32 0, i32 1}
// CHECK-IR: [[WDECOR2]] = !{i32 6443, i32 1, i32 2}
// CHECK-IR: [[WDECOR3]] = !{i32 6443, i32 2, i32 2}
