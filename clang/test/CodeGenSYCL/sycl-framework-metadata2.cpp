// Test checks that !sycl-framework metadata is emitted only to functions
// whose topmost namespace is sycl.

// RUN: %clang_cc1 -fsycl-is-device -O0 -fsycl-optimize-non-user-code -triple spir64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// Check that 'anonymous namespace'::sycl::* functions are not marked with
// !sycl-framework metadata since topmost namespace is anonymous
// instead of sycl.
namespace {
  namespace sycl {
    // CHECK-NOT: @_ZN2V14sycl4bar4Ev() {{.*}} !sycl-framework
    void bar4() {}
  }
}

// Check that kernel is not marked with !sycl-framework metadata
// CHECK-NOT: @_ZTSZ4mainE6kernel() {{.*}} !sycl-framework
template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &func) {
  func();
}

int main() {
  kernel_single_task<class kernel>([]() {
    sycl::bar4();
  });
}