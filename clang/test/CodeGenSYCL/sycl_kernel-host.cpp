// RUN: %clang_cc1 -fsycl-is-host -triple spir64 -disable-llvm-passes %s -emit-llvm -o -  | FileCheck %s
// Test that the kernel implementation routine marked with 'sycl_kernel'
// has the attribute 'sycl_kernel' in the generated LLVM IR and that the
// function object passed to the sycl kernel is marked 'alwaysinline'
// on the host.

// CHECK: define {{.*}}spir_func void @{{.*}}func{{.*}}() #[[NOSKA:[0-9]+]] {
// CHECK: define internal spir_func void @{{.*}}Kernel{{.*}}Foo{{.*}}({{.*}}) #[[SKA:[0-9]+]] {
// CHECK: call spir_func void @{{.*}}KernelImpl{{.*}}({{.*}}, i32 1, double 2.000000e+00)
// CHECK: define internal spir_func void @{{.*}}Kernel{{.*}}Bar{{.*}}({{.*}}) #[[SKA]] {
// CHECK: call spir_func void @{{.*}}KernelImpl{{.*}}({{.*}}, i32 1, double 2.000000e+00)
// CHECK: define internal spir_func void @{{.*}}KernelImpl{{.*}}({{.*}} %f, i32 %i, double %d) #[[SKA]] {
// CHECK: call spir_func void @{{.*}}func{{.*}}(%class
// CHECK: define internal spir_func void @{{.*}}func{{.*}}(%class.anon* {{[^,]*}} %this, i32 %i, double %d) #[[ALWAYSINLINE:[0-9]+]]
// CHECK: define linkonce_odr spir_func void @{{.*}}KernelImpl{{.*}}Functor{{.*}}({{.*}}, i32 %i, double %d) #[[SKA]] comdat {
// CHECK: call spir_func void @{{.*}}Functor{{.*}}(%struct
// CHECK: define linkonce_odr spir_func void @{{.*}}Functor{{.*}}(%struct.Functor* {{[^,]*}} %this, i32 %i, double %d) #[[ALWAYSINLINE]]

template <typename Func>
void __attribute__((sycl_kernel))
KernelImpl(Func f, int i, double d) {
  // CHECK-NOT: call void
  f(i, d);
}

template <typename Name, typename Func>
void __attribute__((sycl_kernel))
Kernel(Func f) {
  KernelImpl(f, 1, 2.0);
}

struct Functor {
  void operator()(int i, double d) { d = i + 2; };
} functionobj;

void func() {
  auto Lambda = [](int i, double d) { d += i; };
  Kernel<class Foo>(Lambda);
  Kernel<class Bar>(functionobj);
}

// CHECK-NOT: attributes #[[NOSKA]] = { {{.*}}"sycl_kernel"{{.*}} }
// CHECK: attributes #[[SKA]] = { {{.*}}"sycl_kernel"{{.*}} }
// CHECK: attributes #[[ALWAYSINLINE]] = { {{.*}}alwaysinline{{.*}} }
