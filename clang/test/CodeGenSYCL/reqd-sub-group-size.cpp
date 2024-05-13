// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -disable-llvm-passes -sycl-std=2020 -triple spir64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

using namespace sycl;
queue q;

class Functor16 {
public:
  [[intel::reqd_sub_group_size(16)]] void operator()() const {}
};

class Functor8 {
public:
  [[intel::reqd_sub_group_size(8)]] void operator()() const {}
};

template <int SIZE>
class Functor2 {
public:
  [[intel::reqd_sub_group_size(SIZE)]] void operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    Functor16 f16;
    h.single_task<class kernel_name1>(f16);

    Functor8 f8;
    h.single_task<class kernel_name2>(f8);

    h.single_task<class kernel_name3>(
        []() [[intel::reqd_sub_group_size(4)]]{});

    Functor2<2> f2;
    h.single_task<class kernel_name4>(f2);
  });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1() #0 {{.*}} !intel_reqd_sub_group_size ![[SGSIZE16:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name2() #0 {{.*}} !intel_reqd_sub_group_size ![[SGSIZE8:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name3() #0 {{.*}} !intel_reqd_sub_group_size ![[SGSIZE4:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name4() #0 {{.*}} !intel_reqd_sub_group_size ![[SGSIZE2:[0-9]+]]
// CHECK: ![[SGSIZE16]] = !{i32 16}
// CHECK: ![[SGSIZE8]] = !{i32 8}
// CHECK: ![[SGSIZE4]] = !{i32 4}
// CHECK: ![[SGSIZE2]] = !{i32 2}
