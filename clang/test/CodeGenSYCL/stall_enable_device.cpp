// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// Tests for IR of Intel FPGA [[intel::use_stall_enable_clusters]] function attribute on Device.
// The metadata to be attached to the functionDecl that the attribute is applied to.
// The attributes get propagated to the kernel metadata i.e. spir_kernel if directly applied
// through functors/lambda function.

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
    // CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel1() {{.*}} !stall_enable ![[NUM4:[0-9]+]]
    // CHECK: define {{.*}}spir_func void @{{.*}}FuncObjclEv(%struct.{{.*}}FuncObj addrspace(4)* align 1 dereferenceable_or_null(1) %this) #3 comdat align 2 !stall_enable ![[NUM4]]
    h.single_task<class test_kernel1>(
        FuncObj());

    // CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel2() {{.*}} !stall_enable ![[NUM4]]
    // CHECK define {{.*}}spir_func void @{{.*}}FooclEv(%class._ZTS3Foo.Foo addrspace(4)* align 1 dereferenceable_or_null(1) %this) #3 comdat align 2 !stall_enable ![[NUM4]]
    Foo f;
    h.single_task<class test_kernel2>(f);

    // Test attribute is not propagated to the kernel metadata i.e. spir_kernel.
    // CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel3()
    // CHECK-NOT: !stall_enable
    // CHECK-SAME: {
    // CHECK: define {{.*}}spir_func void @{{.*}}test{{.*}} !stall_enable ![[NUM4]]
    h.single_task<class test_kernel3>(
        []() { test(); });

    // Test attribute is not propagated to the kernel metadata i.e. spir_kernel.
    // CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel4()
    // CHECK-NOT: !stall_enable
    // CHECK-SAME: {
    // CHECK: define {{.*}}spir_func void @{{.*}}test1{{.*}}(%class.{{.*}}test1{{.*}}.anon addrspace(4)* align 1 dereferenceable_or_null(1) %this) #4 align 2 !stall_enable ![[NUM4]]
    h.single_task<class test_kernel4>(
        []() { test1(); });

    // CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel5() {{.*}} !stall_enable ![[NUM4]]
    h.single_task<class test_kernel5>(
        []() [[intel::use_stall_enable_clusters]]{});

    // CHECK: define {{.*}}spir_kernel void @{{.*}}test_kernel6() {{.*}} !stall_enable ![[NUM4]]
    // CHECK: define {{.*}}spir_func void @{{.*}}Functor{{.*}}(%class._ZTS7Functor.Functor addrspace(4)* align 1 dereferenceable_or_null(1) %this) #3 comdat align 2 !stall_enable ![[NUM4]]
    Functor f1;
    h.single_task<class test_kernel6>(f1);
  });
  return 0;
}

// CHECK: ![[NUM4]] = !{i32 1}
