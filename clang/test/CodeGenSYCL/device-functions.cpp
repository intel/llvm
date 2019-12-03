// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -S -emit-llvm %s -o - | FileCheck %s

template <typename T>
T bar(T arg);

void foo() {
  int a = 1 + 1 + bar(1);
}

template <typename T>
T bar(T arg) {
  return arg;
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

// Make sure that definitions for the types not used in SYCL kernels are not
// emitted
// CHECK-NOT: %struct.A
// CHECK-NOT: @a = {{.*}} %struct.A
struct A {
  int x = 10;
} a;

int main() {
  a.x = 8;
  kernel_single_task<class test_kernel>([]() { foo(); });
  return 0;
}

// baz is not called from the SYCL kernel, so it must not be emitted
// CHECK-NOT: define {{.*}} @{{.*}}baz
void baz() {}

// CHECK-LABEL: define spir_kernel void @{{.*}}test_kernel
// CHECK-LABEL: define internal spir_func void @"_ZZ4mainENK3$_0clEv"(%class.anon* %this)
// CHECK-LABEL: define spir_func void @{{.*}}foo
// CHECK-LABEL: define linkonce_odr spir_func i32 @{{.*}}bar
