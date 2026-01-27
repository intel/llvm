// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s -check-prefixes CHECK-DEVICE,CHECK-BOTH
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s -check-prefixes CHECK-HOST,CHECK-BOTH

// CHECK-BOTH: @_ZZ15attrs_on_staticvE15static_annotate = internal{{.*}}constant i32 30, align 4
// CHECK-BOTH:    [[ANN_annotate:@.str[.0-9]*]] = {{.*}}foobar

// CHECK-BOTH: @llvm.global.annotations
// CHECK-DEVICE-SAME: { ptr addrspace(1) @_ZZ15attrs_on_staticvE15static_annotate
// CHECK-HOST-SAME: { ptr @_ZZ15attrs_on_staticvE15static_annotate
// CHECK-BOTH-SAME: [[ANN_annotate]]{{.*}} i32 15
// CHECK-HOST-NOT: llvm.var.annotation
// CHECK-HOST-NOT: llvm.ptr.annotation

void attrs_on_static() {
  const static int static_annotate [[clang::annotate("foobar")]] = 30;
}

// CHECK-HOST-NOT: llvm.var.annotation
// CHECK-HOST-NOT: llvm.ptr.annotation

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    attrs_on_static();
  });
  return 0;
}
