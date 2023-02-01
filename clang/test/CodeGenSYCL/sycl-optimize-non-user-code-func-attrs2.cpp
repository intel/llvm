// Test checks that noinline and optnone function's attributes aren't attached
// to functions whose topmost namespace is not sycl.

// RUN: %clang_cc1 -fsycl-is-device -O0 -fsycl-optimize-non-user-code -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// Check that kernel do not contain noinline and optnone func attrs.
// CHECK: define {{.*}} @_ZTSZ4mainE6kernel() #[[KERNEL_ATTRS:[0-9]+]]

// Check that 'anonymous namespace'::sycl::* functions do not contain
// noinline and optnone func attrs since topmost namespace is anonymous
// instead of sycl.
namespace {
  namespace sycl {
    // CHECK: define {{.*}} @_ZN12_GLOBAL__N_14sycl4bar4Ev() #[[BAR4_ATTRS:[0-9]+]]
    void bar4() {}
  }
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &func) {
  func();
}

// #0 and #2 contain noinline and optnone func attrs.
// CHECK: attributes #[[KERNEL_ATTRS]] = {{.*}} noinline {{.*}} optnone
// CHECK: attributes #[[BAR4_ATTRS]] = {{.*}} noinline {{.*}} optnone

int main() {
  kernel_single_task<class kernel>([]() {
    sycl::bar4();
  });
}
