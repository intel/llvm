// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device \
// RUN:   -triple spir64-unknown-unknown -sycl-std=2020 -emit-llvm %s -o - \
// RUN:   | FileCheck %s

// A free function kernel that is a function template instantiation has vague
// linkage and may be instantiated with the same arguments in more than one
// translation unit. The generated __sycl_kernel_ entry point must therefore
// have vague (weak_odr) linkage so the copies merge at device link time
// instead of colliding with "symbol multiply defined". A non-template free
// function kernel keeps strong external linkage.

namespace ns {

template <typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]
void templated_kernel(T *p) { *p = *p + 1; }

template void templated_kernel<int>(int *);

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]
void plain_kernel(int *p) { *p = *p + 1; }

} // namespace ns

// Template instantiation -> vague linkage entry point.
// CHECK: define weak_odr spir_kernel void @{{.*}}__sycl_kernel_ns{{.*}}templated_kernel

// Non-template -> strong external entry point.
// CHECK: define dso_local spir_kernel void @{{.*}}__sycl_kernel_ns{{.*}}plain_kernel
