// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm %s -o - |    \
// RUN:    FileCheck %s --check-prefix CHECK-IR

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace ext::oneapi::experimental;
using namespace ext::intel::experimental;

using load_hint = annotated_ptr<
    float, decltype(properties(
               read_hint<cache_control<cache_mode::cached, cache_level::L1>,
                         cache_control<cache_mode::uncached, cache_level::L2,
                                       cache_level::L3>>))>;
using load_assertion = annotated_ptr<
    int,
    decltype(properties(
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
using load_store_hint = annotated_ptr<
    float,
    decltype(properties(
        read_hint<cache_control<cache_mode::cached, cache_level::L3>>,
        read_assertion<cache_control<cache_mode::constant, cache_level::L4>>,
        write_hint<
            cache_control<cache_mode::write_through, cache_level::L4>>))>;

void cache_control_read_hint_func() {
  queue q;
  constexpr int N = 10;
  float *ArrayA = malloc_shared<float>(N, q);
  q.submit([&](handler &cgh) {
    cgh.single_task<>([=]() {
      load_hint src{&ArrayA[0]};
      *src = 55.0f;
    });
  });
}

void cache_control_read_assertion_func() {
  queue q;
  constexpr int N = 10;
  int *ArrayA = malloc_shared<int>(N, q);
  q.submit([&](handler &cgh) {
    cgh.single_task<>([=]() {
      load_assertion src{&ArrayA[0]};
      *src = 66;
    });
  });
}

void cache_control_write_hint_func() {
  queue q;
  constexpr int N = 10;
  float *ArrayA = malloc_shared<float>(N, q);
  q.submit([&](handler &cgh) {
    cgh.single_task<>([=]() {
      store_hint dst{&ArrayA[0]};
      *dst = 77.0f;
    });
  });
}

void cache_control_read_write_func() {
  queue q;
  constexpr int N = 10;
  float *ArrayA = malloc_shared<float>(N, q);
  q.submit([&](handler &cgh) {
    cgh.single_task<>([=]() {
      load_store_hint dst{&ArrayA[0]};
      *dst = 77.0f;
    });
  });
}

// CHECK-IR: spir_kernel{{.*}}cache_control_read_hint_func
// CHECK-IR: {{.*}}addrspacecast ptr addrspace(1){{.*}}!spirv.Decorations [[RHINT:.*]]
// CHECK-IR: ret void

// CHECK-IR: spir_kernel{{.*}}cache_control_read_assertion_func
// CHECK-IR: {{.*}}addrspacecast ptr addrspace(1){{.*}}!spirv.Decorations [[RASSERT:.*]]
// CHECK-IR: ret void

// CHECK-IR: spir_kernel{{.*}}cache_control_write_hint_func
// CHECK-IR: {{.*}}addrspacecast ptr addrspace(1){{.*}}!spirv.Decorations [[WHINT:.*]]
// CHECK-IR: ret void

// CHECK-IR: spir_kernel{{.*}}cache_control_read_write_func
// CHECK-IR: {{.*}}addrspacecast ptr addrspace(1){{.*}}!spirv.Decorations [[RWHINT:.*]]
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

// CHECK-IR: [[RWHINT]] = !{[[RWHINT1:.*]], [[RWHINT2:.*]], [[RWHINT3:.*]]}
// CHECK-IR: [[RWHINT1]] = !{i32 6442, i32 2, i32 1}
// CHECK-IR: [[RWHINT2]] = !{i32 6442, i32 3, i32 4}
// CHECK-IR: [[RWHINT3]] = !{i32 6443, i32 3, i32 1}
