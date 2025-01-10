// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -triple spir64 -o - %s | FileCheck %s

// This test checks if the sycl_registered_kernels named metadata and
// associated entries are generated for registered kernel names.

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void foo() {
}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void bar();

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
  {"nontype", tempfoo3<5>},
  {"decl non-temp", bar}
)]];
}

// Check that the functions registered in the __registered_kernels__ list
// are defined or declared as appropriate, that the kernels are generated
// for these functions and are called from the generated kernels.

// Check that the definitions for the functions foo, ff_4, iota,
// tempfoo2 explicitly instantiated with int, and tempfoo2 specialized
// with float are generated.
// CHECK: define {{.*}} void @[[FOO:[_A-Za-z0-9]+]]()
// CHECK: define {{.*}} void @[[FF_4:[_A-Za-z0-9]+]]()
// CHECK: define {{.*}} void @[[IOTA:[_A-Za-z0-9]+]](i32 {{.*}} %0, ptr addrspace(4) {{.*}} %1)
// CHECK: define {{.*}} void @[[TEMPFOO2INT:[_A-Za-z0-9]+]](i32 {{.*}} %pt)
// CHECK: define {{.*}} void @[[TEMPFOO2SHORT:[_A-Za-z0-9]+]](i16 {{.*}} %0)

// Check generation of the SYCL kernel for foo, and the call to foo.
// CHECK: define {{.*}} void @_Z17__sycl_kernel_foov()
// CHECK:   call {{.*}} void @[[FOO]]()

// Check generation of the SYCL kernel for ff_4, and the call to ff_4.
// CHECK: define {{.*}} void @_Z18__sycl_kernel_ff_4v()
// CHECK:   call {{.*}} void @[[FF_4]]()

// Check generation of the SYCL kernel for iota, and the call to iota.
// CHECK: define {{.*}} void @_Z18__sycl_kernel_iotaiPi(i32 {{.*}} %__arg_, ptr addrspace(1) {{.*}} %__arg_1)
// CHECK:   call {{.*}} void @[[IOTA]](i32 {{.*}} %0, ptr addrspace(4) {{.*}} %2)

// Check generation of the SYCL kernel for tempfoo2<int>, and the call to
// tempfoo2<int>.
// CHECK: define {{.*}} void @_Z22__sycl_kernel_tempfoo2IiEvT_(i32 {{.*}} %__arg_pt)
// CHECK:   call {{.*}} void @[[TEMPFOO2INT]](i32 {{.*}} %0)

// Check generation of the SYCL kernel for tempfoo2<short>, and the call to
// tempfoo2<short>.
// CHECK: define {{.*}} void @_Z22__sycl_kernel_tempfoo2IsEvT_(i16 {{.*}} %__arg_)
// CHECK:   call {{.*}} void @[[TEMPFOO2SHORT]](i16 {{.*}} %0)

// Check generation of the SYCL kernel for tempfoo<int>, the call to
// tempfoo<int>, and its declaration.
// CHECK: define {{.*}} void @_Z21__sycl_kernel_tempfooIiEvT_(i32 {{.*}} %__arg_pt)
// CHECK:   call {{.*}} void @[[TEMPFOOINT:[_A-Za-z0-9]+]](i32 {{.*}} %0)
// CHECK: declare {{.*}} void @[[TEMPFOOINT]](i32 {{.*}})

// Check generation of the SYCL kernel for tempfoo2<float>, the call to
// tempfoo2<float>, and its declaration.
// CHECK: define {{.*}} void @_Z22__sycl_kernel_tempfoo2IfEvT_(float {{.*}} %__arg_f)
// CHECK:   call {{.*}} void @[[TEMPFOO2FLOAT:[_A-Za-z0-9]+]](float {{.*}} %0)
// CHECK: declare {{.*}} void @[[TEMPFOO2FLOAT]](float {{.*}})

// Check generation of the SYCL kernel for tempfoo3<5>, the call to
// tempfoo3<5>, and its definition.
// CHECK: define {{.*}} void @_Z22__sycl_kernel_tempfoo3ILi5EEvv()
// CHECK:   call {{.*}} void @[[TEMPFOO35:[_A-Za-z0-9]+]]()
// CHECK: define {{.*}} void @[[TEMPFOO35]]()

// Check generation of the SYCL kernel for bar, the call to
// bar, and its declaration.
// CHECK: define {{.*}} void @_Z17__sycl_kernel_barv()
// CHECK:   call {{.*}} void @[[BAR:[_A-Za-z0-9]+]]()
// CHECK: declare {{.*}} void @[[BAR]]()

// Check for the presence of sycl_registered_kernels named metadata.
// CHECK: !sycl_registered_kernels = !{![[LIST:[0-9]+]]}
// CHECK: ![[LIST]] = !{![[ENT1:[0-9]+]], ![[ENT2:[0-9]+]], ![[ENT3:[0-9]+]], ![[ENT4:[0-9]+]],  ![[ENT5:[0-9]+]], ![[ENT6:[0-9]+]], ![[ENT7:[0-9]+]], ![[ENT8:[0-9]+]], ![[ENT9:[0-9]+]]}
// CHECK: ![[ENT1]] = !{!"foo", !"{{.*}}sycl_kernel{{.*}}foo{{.*}}"}
// CHECK: ![[ENT2]] = !{!"foo3", !"{{.*}}sycl_kernel{{.*}}ff_4{{.*}}"}
// CHECK: ![[ENT3]] = !{!"iota", !"{{.*}}sycl_kernel{{.*}}iota{{.*}}"}
// CHECK: ![[ENT4]] = !{!"inst temp", !"{{.*}}sycl_kernel{{.*}}tempfoo2{{.*}}"}
// CHECK: ![[ENT5]] = !{!"def spec", !"{{.*}}sycl_kernel{{.*}}tempfoo2{{.*}}"}
// CHECK: ![[ENT6]] = !{!"decl temp", !"{{.*}}sycl_kernel{{.*}}tempfoo{{.*}}"}
// CHECK: ![[ENT7]] = !{!"decl spec", !"{{.*}}sycl_kernel{{.*}}tempfoo2{{.*}}"}
// CHECK: ![[ENT8]] = !{!"nontype", !"{{.*}}sycl_kernel{{.*}}tempfoo3{{.*}}"}
// CHECK: ![[ENT9]] = !{!"decl non-temp", !"{{.*}}sycl_kernel{{.*}}bar{{.*}}"}
