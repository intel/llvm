// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// Tests for IR of Intel FPGA [[intel::use_stall_enable_clusters]] function attribute on Device.
// The metadata to be attached to the functionDecl that the attribute is applied to.
// The attributes get propagated to the kernel metadata i.e. spir_kernel.

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

[[intel::use_stall_enable_clusters]] void test() {}

struct FuncObj {
  [[intel::use_stall_enable_clusters]] void operator()() const {}
};

void test1() {
  auto lambda = []() [[intel::use_stall_enable_clusters]]{};
  lambda();
}

class Foo {
public:
  [[intel::use_stall_enable_clusters]] void operator()() const {}
};

class Functor {
public:
  [[intel::use_stall_enable_clusters]] void operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    // CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel1() #0 !stall_enable ![[NUM4:[0-9]+]] !kernel_arg_buffer_location ![[NUM5:[0-9]+]]
    // CHECK: define {{.*}}spir_func void @{{.*}}FuncObjclEv(%struct.{{.*}}FuncObj addrspace(4)* align 1 dereferenceable_or_null(1) %this) #3 comdat align 2 !stall_enable ![[NUM4]]
    h.single_task<class test_kernel1>(
        FuncObj());

    // CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel2() #0 !stall_enable ![[NUM4]] !kernel_arg_buffer_location ![[NUM5]]
    // CHECK define {{.*}}spir_func void @{{.*}}FooclEv(%class._ZTS3Foo.Foo addrspace(4)* align 1 dereferenceable_or_null(1) %this) #3 comdat align 2 !stall_enable ![[NUM4]]
    Foo f;
    h.single_task<class test_kernel2>(f);

    // CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel3() #0 !kernel_arg_buffer_location ![[NUM5]]
    // CHECK: define {{.*}}spir_func void @_Z4testv() #3 !stall_enable ![[NUM4]]
    h.single_task<class test_kernel3>(
        []() { test(); });

    // CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel4() #0 !kernel_arg_buffer_location ![[NUM5]]
    // CHECK: define {{.*}}spir_func void @{{.*}}test1vENKUlvE_clEv(%class.{{.*}}test1{{.*}}.anon addrspace(4)* align 1 dereferenceable_or_null(1) %this) #4 align 2 !stall_enable ![[NUM4]]
    h.single_task<class test_kernel4>(
        []() { test1(); });

    // CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel5() #0 !stall_enable ![[NUM4]] !kernel_arg_buffer_location ![[NUM5]]
    h.single_task<class test_kernel5>(
        []() [[intel::use_stall_enable_clusters]]{});

    // CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel6() #0 !stall_enable ![[NUM4]] !kernel_arg_buffer_location ![[NUM5]]
    Functor f1;
    h.single_task<class test_kernel6>(f1);
  });
  return 0;
}

// CHECK: ![[NUM4]] = !{i32 1}
// CHECK: ![[NUM5]] = !{}
