// RUN: %clangxx -O0 -fsycl -fsycl-device-only -fsycl-targets=intel_gpu_pvc -fsycl-fp64-conv-emu %s -c -S -emit-llvm -o- | FileCheck %s

// CHECK: define {{.*}} spir_kernel void @_ZTSZ4mainE19fake_kernel_compute(){{.*}} !sycl_used_aspects ![[ASPECTFP64:[0-9]+]]
// CHECK-NOT: define {{.*}} spir_kernel void @_ZTSZ4mainE19fake_kernel_convert(){{.*}} !sycl_used_aspects
// CHECK-DAG: ![[ASPECTFP64]] = !{i32 6}
//
// Tests if -fsycl-fp64-conv-emu option helps to correctly generate fp64 aspect.

#include "sycl.hpp"

template <typename t, typename Func>
__attribute__((sycl_kernel)) void kernel_compute(const Func &func) {
    double a[3] = {1,2,3};
    double b[3] = {1,2,3};
    int i = 1;
    func(a, b, i);
}

template <typename t, typename Func>
__attribute__((sycl_kernel)) void kernel_convert(const Func &func) {
    double a[3] = {1,2,3};
    double b[3] = {1,2,3};
    int i = 1;
    func(a, b, i);;
}

extern "C" {
// symbols so that linker find them and doesn't fail.
void __sycl_register_lib(void *) {}
void __sycl_unregister_lib(void *) {}
}

int main() {
    kernel_compute<class fake_kernel_compute>([](double *a, double *b, int i) { b[i] = a[i] + 1.0; });
    kernel_convert<class fake_kernel_convert>([](double *a, double *b, int i) { b[i] = (double)((float)(a[i])); });
}
