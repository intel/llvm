// RUN: %clang_cc1 -fsycl-is-host -triple -x86_64-unknown-linux-gnu -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// Tests for IR of Intel FPGA [[intel::fpga_pipeline]] function attribute on Host (no-op in IR-CodeGen for host-mode).

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

template <int SIZE>
class KernelFunctor5 {
public:
  [[intel::fpga_pipeline(SIZE)]] void operator()() const {}
};

[[intel::fpga_pipeline]] void func1() {}

void foo() {

  KernelFunctor5<5> f5;
  kernel<class kernel_name_1>(f5);
}

// CHECK-NOT: !disable_loop_pipelining
