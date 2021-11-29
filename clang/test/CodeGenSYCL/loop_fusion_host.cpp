// RUN: %clang_cc1 -fsycl-is-host -triple -x86_64-unknown-linux-gnu -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

// This test uses SYCL host only mode without integration header, so
// forward declare used kernel name class, otherwise it will be diagnosed by
// the diagnostic implemented in https://github.com/intel/llvm/pull/4945.
// The error happens because in host mode it is assumed that all kernel names
// are forward declared at global or namespace scope because of integration
// header.
class kernel_name_1;

template <int SIZE>
class KernelFunctor5 {
public:
  [[intel::loop_fuse(SIZE)]] void operator()() const {}
};

template <int SIZE>
class KernelFunctor3 {
public:
  [[intel::loop_fuse_independent(SIZE)]] void operator()() const {}
};

[[intel::loop_fuse]] void func1() {}
[[intel::loop_fuse_independent]] void func2() {}

void foo() {

  KernelFunctor5<5> f5;
  kernel<class kernel_name_1>(f5);

  KernelFunctor5<3> f3;
  kernel<class kernel_name_1>(f5);
}
// CHECK-NOT: !loop_fuse
