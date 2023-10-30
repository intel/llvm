// RUN: %clangxx -D__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__  \
// RUN:    -fsycl-device-only -S -Xclang -emit-llvm %s -o - |    \
// RUN:    FileCheck %s --check-prefix CHECK-IR

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace ext::oneapi::experimental;
using namespace ext::intel::experimental;

using load_hint = annotated_ptr<
    float, decltype(properties(
               alignment<8>,
               read_hint<cache_control<cache_mode::cached, cache_level::L1>,
                         cache_control<cache_mode::uncached, cache_level::L2,
                                       cache_level::L3>>))>;

using load_assertion = annotated_ptr<
    int,
    decltype(properties(
        alignment<8>,
        read_assertion<cache_control<cache_mode::constant, cache_level::L1>,
                       cache_control<cache_mode::invalidate, cache_level::L2,
                                     cache_level::L3>>))>;

using store_hint = annotated_ptr<
    float,
    decltype(properties(
        write_hint<cache_control<cache_mode::write_through, cache_level::L1>,
                   cache_control<cache_mode::write_back, cache_level::L2,
                                 cache_level::L3>,
                   cache_control<cache_mode::streaming, cache_level::L4>>))>;

void cache_control_read_hint_func() {
  queue q;
  constexpr int N = 10;
  float *ArrayA = malloc_shared<float>(N, q);
  q.submit([&](handler &cgh) {
    cgh.parallel_for<>(range<1>(N), [=](item<1> item) {
      auto item_id = item.get_linear_id();
      load_hint src{&ArrayA[item_id]};
      *src = 55.0f;
    });
  });
}

void cache_control_read_assertion_func() {
  queue q;
  constexpr int N = 10;
  int *ArrayA = malloc_shared<int>(N, q);
  q.submit([&](handler &cgh) {
    cgh.parallel_for<>(range<1>(N), [=](item<1> item) {
      auto item_id = item.get_linear_id();
      load_assertion src{&ArrayA[item_id]};
      *src = 66;
    });
  });
}

void cache_control_write_hint_func() {
  queue q;
  constexpr int N = 10;
  float *ArrayA = malloc_shared<float>(N, q);
  q.submit([&](handler &cgh) {
    cgh.parallel_for<>(range<1>(N), [=](item<1> item) {
      auto item_id = item.get_linear_id();
      store_hint dst{&ArrayA[item_id]};
      *dst = 77.0f;
    });
  });
}

// CHECK-IR: spir_kernel{{.*}}cache_control_read_hint_func
// CHECK-IR: {{.*}}getelementptr inbounds float{{.*}}!spirv.Decorations [[RHINT:.*]]
// CHECK-IR: ret void

// CHECK-IR: spir_kernel{{.*}}cache_control_read_assertion_func
// CHECK-IR: {{.*}}getelementptr inbounds i32{{.*}}!spirv.Decorations [[RASSERT:.*]]
// CHECK-IR: ret void

// CHECK-IR: spir_kernel{{.*}}cache_control_write_hint_func
// CHECK-IR: {{.*}}getelementptr inbounds float{{.*}}!spirv.Decorations [[WHINT:.*]]
// CHECK-IR: ret void

// CHECK-IR: [[RHINT]] = !{[[RHINT1:.*]], [[RHINT2:.*]], [[RHINT3:.*]]}
// CHECK-IR: [[RHINT1]] = !{i32 6442, i32 1, i32 0}
// CHECK-IR: [[RHINT2]] = !{i32 6442, i32 2, i32 0}
// CHECK-IR: [[RHINT3]] = !{i32 6442, i32 0, i32 1}

// CHECK-IR: [[RASSERT]] = !{[[RASSERT1:.*]], [[RASSERT2:.*]], [[RASSERT3:.*]]}
// CHECK-IR: [[RASSERT1]] = !{i32 6442, i32 1, i32 3}
// CHECK-IR: [[RASSERT2]] = !{i32 6442, i32 2, i32 3}
// CHECK-IR: [[RASSERT3]] = !{i32 6442, i32 0, i32 4}

// CHECK-IR: [[WHINT]] = !{[[WHINT1:.*]], [[WHINT2:.*]], [[WHINT3:.*]], [[WHINT4:.*]]}
// CHECK-IR: [[WHINT1]] = !{i32 6443, i32 3, i32 3}
// CHECK-IR: [[WHINT2]] = !{i32 6443, i32 0, i32 1}
// CHECK-IR: [[WHINT3]] = !{i32 6443, i32 1, i32 2}
// CHECK-IR: [[WHINT4]] = !{i32 6443, i32 2, i32 2}
