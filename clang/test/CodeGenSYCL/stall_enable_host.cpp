// RUN: %clang_cc1 -fsycl-is-host -triple -x86_64-unknown-linux-gnu -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// Tests for IR of Intel FPGA [[intel::use_stall_enable_clusters]] function attribute on Host (no-op in IR-CodeGen for host-mode).

[[intel::use_stall_enable_clusters]] void test() {}

void test1() {
  auto lambda = []() [[intel::use_stall_enable_clusters]]{};
  lambda();
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

class KernelFunctor {
public:
  [[intel::use_stall_enable_clusters]] void operator()() const {}

};

void foo() {

  KernelFunctor f;
  kernel<class kernel_name_1>(f);
}

// CHECK-NOT: !stall_enable
