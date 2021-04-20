// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -I%S \
// RUN:      -fcxx-exceptions -fexceptions -disable-llvm-passes \
// RUN:      -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-DEVICE
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -I%S \
// RUN:      -fcxx-exceptions -fexceptions -disable-llvm-passes \
// RUN:      -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-HOST-LIN
//
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -I%S \
// RUN:      -fcxx-exceptions -fexceptions -disable-llvm-passes \
// RUN:      -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-HOST-WIN

// The test checks that exception handling code is generated only for host and not for device.

void f1() {}
void f2() {}
void f3() {}

void foo_noexcept() noexcept {
  // CHECK-DEVICE: call spir_func void @_Z2f1v()
  // CHECK-HOST-LIN: call void @_Z2f1v()
  // CHECK-HOST-WIN: call void @"?f1@@YAXXZ"()
  f1();
}

void foo_throw() throw() {
  // CHECK-DEVICE: call spir_func void @_Z2f2v()
  // CHECK-HOST-LIN: call void @_Z2f2v()
  // CHECK-HOST-WIN: call void @"?f3@@YAXXZ"()
  f2();
}

struct A {
  // Non-trivial destructor to force generation of cleanup code
  ~A(){}
};

void foo_cleanup() {
  A a;
  // CHECK-DEVICE: call spir_func void @_Z2f3v()
  // CHECK-HOST: call void @_Z2f3v()
  f3();
  // CHECK-DEVICE: call spir_func void @_ZN1AD1Ev
  // Regular + exception cleanup
  // CHECK-HOST-LIN: call void @_ZN1AD1Ev
  // CHECK-HOST-LIN: call void @_ZN1AD2Ev
  // CHECK-HOST-WIN: call void @"??1A@@QEAA@XZ"(%struct.A* {{[^,]*}} %a)
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel>([=](){
    foo_noexcept();
    foo_throw();
    foo_cleanup();
  });
}
