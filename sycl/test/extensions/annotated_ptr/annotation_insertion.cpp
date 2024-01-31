// RUN: %clangxx -fsycl-device-only -fsycl-targets=spir64_fpga -S -emit-llvm %s -o - | FileCheck %s

// Tests that `@llvm.ptr.annotation` is inserted when calling
// `annotated_ptr::get()`

#include "sycl/sycl.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <iostream>

// clang-format on

using namespace sycl;
using namespace ext::oneapi::experimental;
using namespace ext::intel::experimental;

// CHECK: @[[AnnStr:.*]] = private unnamed_addr addrspace(1) constant [19 x i8] c"{5921:\220\22}{44:\228\22}\00"

using ann_ptr_t1 =
    annotated_ptr<int, decltype(properties(buffer_location<0>, alignment<8>))>;

struct MyIP {
  ann_ptr_t1 a;

  MyIP(int *a_) : a(a_) {}

  void operator()() const {
    // CHECK: %ptr.addr = alloca ptr addrspace(4), align 8
    // CHECK: store ptr addrspace(4) %ptr, ptr %ptr.addr, align 8
    // CHECK: %[[LoadPtr:.*]] = load ptr addrspace(4), ptr %ptr.addr, align 8
    // CHECK: %[[AnnPtr:.*]] = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %[[LoadPtr]], ptr addrspace(1) @[[AnnStr]]
    // CHECK: ret ptr addrspace(4) %[[AnnPtr]]
    int *ptr = a.get(); // llvm.ptr.annotation is inserted
    *ptr = 15;
  }
};

void TestVectorAddWithAnnotatedMMHosts() {
  sycl::queue q;
  auto raw = malloc_shared<int>(5, q);
  q.submit([&](handler &h) { h.single_task(MyIP{raw}); }).wait();
  free(raw, q);
}

int main() {
  TestVectorAddWithAnnotatedMMHosts();
  return 0;
}
