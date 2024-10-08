// RUN: %clangxx -fsycl-device-only -fsycl-targets=spir64_fpga -S -emit-llvm %s -o - | FileCheck %s  --check-prefix CHECK-IR

// Tests that the address space of `annotated_ptr` kernel argument is refined
// when:
// 1. `usm_kind` property is specified in annotated_ptr type, and
// 2. both `__SYCL_DEVICE_ONLY__` and `__ENABLE_USM_ADDR_SPACE__` are turned on
// (i.e. equiv to flags "-fsycl-device-only -fsycl-targets=spir64_fpga")

#include "sycl/sycl.hpp"

#include <sycl/ext/intel/fpga_extensions.hpp>

#include <iostream>

using namespace sycl;
using namespace ext::oneapi::experimental;
using namespace ext::intel::experimental;

using annotated_ptr_t1 =
    annotated_ptr<int,
                  decltype(properties(buffer_location<0>, usm_kind_device))>;
using annotated_ptr_t2 =
    annotated_ptr<int, decltype(properties(buffer_location<1>, usm_kind_host))>;
using annotated_ptr_t3 =
    annotated_ptr<int,
                  decltype(properties(buffer_location<2>, usm_kind_shared))>;

struct MyIP {

  // CHECK-IR: spir_kernel void @_ZTS4MyIP(ptr addrspace(5) {{.*}} %_arg_a, ptr addrspace(6) {{.*}} %_arg_b, ptr addrspace(1) {{.*}} %_arg_c)
  annotated_ptr_t1 a;
  annotated_ptr_t2 b;
  annotated_ptr_t3 c;

  MyIP(int *a_, int *b_, int *c_) : a(a_), b(b_), c(c_) {}

  void operator()() const { *a = *b + *c; }
};

void TestVectorAddWithAnnotatedMMHosts() {
  queue q;
  auto p1 = malloc_device<int>(5, q);
  auto p2 = malloc_host<int>(5, q);
  auto p3 = malloc_shared<int>(5, q);

  q.submit([&](handler &h) { h.single_task(MyIP{p1, p2, p3}); }).wait();

  free(p1, q);
  free(p2, q);
  free(p3, q);
}

int main() {
  TestVectorAddWithAnnotatedMMHosts();
  return 0;
}
