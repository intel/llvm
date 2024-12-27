// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -o - %s | FileCheck %s

// This test checks if the sycl_registered_kernels module flag and
// associated entries are generated for registered kernel names.

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void foo() {
}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff_4() {
}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void iota(int, int *) {
}

template <typename T>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void tempfoo(T pt);

template <typename T>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void tempfoo2(T pt) {
  T t;
}

template void tempfoo2<int>(int);

template <>
void tempfoo2(float f);

template <>
void tempfoo2(short) { }

template <int N>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void tempfoo3() {
  (void)N;
}

namespace N {
[[__sycl_detail__::__registered_kernels__(
  {"foo", foo},
  {"iota", (void(*)(int, int *))iota},
  {"decl temp", tempfoo<int>},
  {"inst temp", tempfoo2<int>},
  {"decl spec", tempfoo2<float>},
  {"def spec", tempfoo2<short>},
  {"foo3", ff_4},
  {"nontype", tempfoo3<5>}
)]];
}

// Check for the presence of sycl-device module flag in device
// compilations and its absence in host compilations.
// CHECK: !{{[0-9]+}} = !{i32 5, !"sycl_registered_kernels", ![[LIST:[0-9]+]]}
// CHECK: ![[LIST]] = !{![[ENT1:[0-9]+]], ![[ENT2:[0-9]+]], ![[ENT3:[0-9]+]], ![[ENT4:[0-9]+]],  ![[ENT5:[0-9]+]], ![[ENT6:[0-9]+]], ![[ENT7:[0-9]+]], ![[ENT8:[0-9]+]]}
// CHECK: ![[ENT1]] = !{!"foo", !"{{.*}}sycl_kernel{{.*}}foo{{.*}}"}
// CHECK: ![[ENT2]] = !{!"foo3", !"{{.*}}sycl_kernel{{.*}}ff_4{{.*}}"}
// CHECK: ![[ENT3]] = !{!"iota", !"{{.*}}sycl_kernel{{.*}}iota{{.*}}"}
// CHECK: ![[ENT4]] = !{!"inst temp", !"{{.*}}sycl_kernel{{.*}}tempfoo2{{.*}}"}
// CHECK: ![[ENT5]] = !{!"def spec", !"{{.*}}sycl_kernel{{.*}}tempfoo2{{.*}}"}
// CHECK: ![[ENT6]] = !{!"decl temp", !"{{.*}}sycl_kernel{{.*}}tempfoo{{.*}}"}
// CHECK: ![[ENT7]] = !{!"decl spec", !"{{.*}}sycl_kernel{{.*}}tempfoo2{{.*}}"}
// CHECK: ![[ENT8]] = !{!"nontype", !"{{.*}}sycl_kernel{{.*}}tempfoo3{{.*}}"}
