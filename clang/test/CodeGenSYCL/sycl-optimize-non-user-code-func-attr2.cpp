// Test checks that noinline and optnone function's attributes aren't attached
// to functions whose topmost namespace is not sycl.

// RUN: %clang_cc1 -fsycl-is-device -O0 -fsycl-optimize-non-user-code -triple spir64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// Check that kernel do not contain noinline and optnone func attrs.
// CHECK: @_ZTSZ4mainE6kernel() #0

// Check that 'anonymous namespace'::sycl::* functions do not contain
// noinline and optnone func attrs since topmost namespace is anonymous
// instead of sycl.
namespace {
  namespace sycl {
    // CHECK: @_ZN12_GLOBAL__N_14sycl4bar4Ev() #1
    void bar4() {}
  }
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &func) {
  func();
}

// #0 and #1 contain noinline and optnone func attrs.
// CHECK: attributes #0 = {{.*}} noinline {{.*}} optnone
// CHECK: attributes #1 = {{.*}} noinline {{.*}} optnone

int main() {
  kernel_single_task<class kernel>([]() {
    sycl::bar4();
  });
}
