// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm \
// RUN:   -std=c++17 %s -o - | FileCheck %s

// Verify the linkage of SYCL kernel caller entry points (CMPLRLLVM-78181).
// A kernel entry point synthesized from a function template instantiation (or
// an inline free function) must be emitted with vague (weak_odr) linkage so
// that the same instantiation in multiple translation units merges instead of
// colliding at device link time. A non-template, non-inline kernel keeps
// strong external linkage.

template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
void kernel_single_task(KernelType kernelFunc) {
  kernelFunc();
}

struct TemplatedKernelName;
struct ExplicitKernelName;
struct NonTemplateKernelName;

struct Functor {
  void operator()() const {}
};

// Implicit instantiation of the entry-point template -> weak_odr.
void trigger_implicit() {
  kernel_single_task<TemplatedKernelName>(Functor{});
}

// Explicit instantiation of the entry-point template -> weak_odr.
template void kernel_single_task<ExplicitKernelName, Functor>(Functor);

// Non-template kernel entry point -> strong external.
[[clang::sycl_kernel_entry_point(NonTemplateKernelName)]]
void non_template_kernel(Functor kernelFunc) {
  kernelFunc();
}

// CHECK-DAG: define weak_odr {{.*}}spir_kernel void @{{.*}}TemplatedKernelName
// CHECK-DAG: define weak_odr {{.*}}spir_kernel void @{{.*}}ExplicitKernelName
// CHECK-DAG: define {{(dso_local )?}}spir_kernel void @{{.*}}NonTemplateKernelName
