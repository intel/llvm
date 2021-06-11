// RUN: %clang_cc1 -fsycl-is-host -triple -x86_64-unknown-linux-gnu -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// Tests for IR of Intel FPGA [[intel::use_stall_enable_clusters]] function attribute on Host.

// Test attribute is presented on function.
[[intel::use_stall_enable_clusters]] void test() {}
// CHECK: define {{.*}}void @{{.*}}testv() #0 !stall_enable ![[NUM2:[0-9]+]]

// Test attribute is presented on lambda function.
void test1() {
  auto lambda = []() [[intel::use_stall_enable_clusters]]{};
  lambda();
  // CHECK: define {{.*}}void @{{.*}}test1vENKUlvE_clEv(%class.anon* nonnull align 1 dereferenceable(1) %this) #0 align 2 !stall_enable ![[NUM2]]
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

  // Test attribute is presented on function metadata.
  KernelFunctor f;
  kernel<class kernel_name_1>(f);
  // CHECK: define {{.*}}void @{{.*}}KernelFunctorclEv(%class.KernelFunctor* nonnull align 1 dereferenceable(1) %this) #2 comdat align 2 !stall_enable ![[NUM2]]
}

// CHECK: ![[NUM2]] = !{i32 1}
