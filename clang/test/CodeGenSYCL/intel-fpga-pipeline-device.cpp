// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// Tests for IR of Intel FPGA [[intel::fpga_pipeline]] function attribute on Device.
#include "sycl.hpp"

using namespace cl::sycl;
queue q;

class Foo {
public:
  [[intel::fpga_pipeline(1)]] void operator()() const {}
};

template <int SIZE>
class Functor {
public:
  [[intel::fpga_pipeline(SIZE)]] void operator()() const {}
};

[[intel::fpga_pipeline(1)]] void func() {}

int main() {
  q.submit([&](handler &h) {
    //CHECK: define dso_local spir_kernel void @{{.*}}kernel_name1() #0 !kernel_arg_buffer_location ![[NUM:[0-9]+]]  !disable_loop_pipelining ![[NUM0:[0-9]+]]
    Foo boo;
    h.single_task<class kernel_name1>(boo);

    // CHECK: define dso_local spir_kernel void @{{.*}}kernel_name2() #0 !kernel_arg_buffer_location ![[NUM]] !disable_loop_pipelining ![[NUM0]]
    h.single_task<class kernel_name2>(
        []() [[intel::fpga_pipeline]]{});

    // CHECK: define dso_local spir_kernel void @{{.*}}kernel_name3() #0 !kernel_arg_buffer_location ![[NUM]] !disable_loop_pipelining ![[NUM1:[0-9]+]]
    h.single_task<class kernel_name3>(
        []() [[intel::fpga_pipeline(0)]]{});

    // CHECK: define dso_local spir_kernel void @{{.*}}kernel_name4() #0 !kernel_arg_buffer_location ![[NUM]] !disable_loop_pipelining ![[NUM0]]
    Functor<1> f;
    h.single_task<class kernel_name4>(f);

    // CHECK: define dso_local spir_kernel void @{{.*}}test_kernel5() #0 !kernel_arg_buffer_location ![[NUM]]
    h.single_task<class test_kernel5>(
        []() { func(); });
  });
  return 0;
}

// CHECK: ![[NUM]] = !{}
// CHECK: ![[NUM0]] = !{i32 0}
// CHECK: ![[NUM1]] = !{i32 1}
