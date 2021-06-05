// RUN: %clang_cc1 -fsycl-is-host -triple -x86_64-unknown-linux-gnu -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

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
