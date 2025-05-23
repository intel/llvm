// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-IR
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

constexpr auto Props1 = sycl::ext::oneapi::experimental::properties{
    sycl::ext::oneapi::experimental::max_work_group_size<8>};
constexpr auto Props2 = sycl::ext::oneapi::experimental::properties{
    sycl::ext::oneapi::experimental::max_work_group_size<8, 4>};
constexpr auto Props3 = sycl::ext::oneapi::experimental::properties{
    sycl::ext::oneapi::experimental::max_work_group_size<8, 4, 2>};

struct TestKernel_Props1 {
  void operator()() const {}
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return Props1;
  }
};

struct TestKernel_Props2 {
  void operator()() const {}
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return Props2;
  }
};

struct TestKernel_Props3 {
  void operator()() const {}
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return Props3;
  }
};

int main() {
  sycl::queue Q;
  sycl::event Ev;

  // CHECK-IR: spir_kernel void @{{.*}}MaxWGSizeKernel0(){{.*}} #[[MaxWGSizeAttr0:[0-9]+]]
  // CHECK-IR-SAME: !max_work_group_size ![[MaxWGSizeMD0:[0-9]+]]
  Q.single_task<class MaxWGSizeKernel0>(TestKernel_Props1{});
  // CHECK-IR: spir_kernel void @{{.*}}MaxWGSizeKernel1(){{.*}} #[[MaxWGSizeAttr1:[0-9]+]]
  // CHECK-IR-SAME: !max_work_group_size ![[MaxWGSizeMD1:[0-9]+]]
  Q.submit([&](sycl::handler &h) {
    h.depends_on(Ev);
    h.single_task<class MaxWGSizeKernel1>(TestKernel_Props2{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MaxWGSizeKernel2(){{.*}} #[[MaxWGSizeAttr2:[0-9]+]]
  // CHECK-IR-SAME: !max_work_group_size ![[MaxWGSizeMD2:[0-9]+]]
  Q.submit([&](sycl::handler &h) {
    h.depends_on({Ev});
    h.single_task<class MaxWGSizeKernel2>(TestKernel_Props3{});
  });

  return 0;
}

// CHECK-IR: attributes #[[MaxWGSizeAttr0]] = { {{.*}}"sycl-max-work-group-size"="8"
// CHECK-IR: attributes #[[MaxWGSizeAttr1]] = { {{.*}}"sycl-max-work-group-size"="8,4"
// CHECK-IR: attributes #[[MaxWGSizeAttr2]] = { {{.*}}"sycl-max-work-group-size"="8,4,2"

// CHECK-IR: ![[MaxWGSizeMD0]] = !{i64 8, i64 1, i64 1}
// CHECK-IR: ![[MaxWGSizeMD1]] = !{i64 4, i64 8, i64 1}
// CHECK-IR: ![[MaxWGSizeMD2]] = !{i64 2, i64 4, i64 8}
