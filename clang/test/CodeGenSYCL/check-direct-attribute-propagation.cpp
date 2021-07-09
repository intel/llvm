// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -sycl-std=2020 -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

class Foo {
public:
  [[intel::scheduler_target_fmax_mhz(1)]] void operator()() const {}
};

template <int SIZE>
class Functor {
public:
  [[intel::scheduler_target_fmax_mhz(SIZE)]] void operator()() const {}
};

[[intel::scheduler_target_fmax_mhz(5)]] void foo() {}

class Foo1 {
public:
  [[intel::num_simd_work_items(1)]] void operator()() const {}
};

template <int SIZE>
class Functor1 {
public:
  [[intel::num_simd_work_items(SIZE)]] void operator()() const {}
};

[[intel::num_simd_work_items(5)]] void foo1() {}

class Foo2 {
public:
  [[intel::no_global_work_offset(1)]] void operator()() const {}
};

template <int SIZE>
class Functor2 {
public:
  [[intel::no_global_work_offset(SIZE)]] void operator()() const {}
};

[[intel::no_global_work_offset(0)]] void foo2() {}

class Foo3 {
public:
  [[intel::max_global_work_dim(1)]] void operator()() const {}
};

template <int SIZE>
class Functor3 {
public:
  [[intel::max_global_work_dim(SIZE)]] void operator()() const {}
};

[[intel::max_global_work_dim(1)]] void foo3() {}


int main() {
  q.submit([&](handler &h) {
    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1() #0 {{.*}} !scheduler_target_fmax_mhz ![[NUM1:[0-9]+]]
    Foo boo;
    h.single_task<class kernel_name1>(boo);

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name2() #0 {{.*}} !scheduler_target_fmax_mhz ![[NUM42:[0-9]+]]
    h.single_task<class kernel_name2>(
        []() [[intel::scheduler_target_fmax_mhz(42)]]{});

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name3() #0 {{.*}} !scheduler_target_fmax_mhz ![[NUM2:[0-9]+]]
    Functor<2> f;
    h.single_task<class kernel_name3>(f);

    // Test attribute is not propagated.
    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name4()
    // CHECK-NOT: !scheduler_target_fmax_mhz
    h.single_task<class kernel_name4>(
        []() { foo(); });

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name5() #0 {{.*}} !num_simd_work_items ![[NUM1]]
    Foo1 boo1;
    h.single_task<class kernel_name5>(boo1);

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name6() #0 {{.*}} !num_simd_work_items ![[NUM42]]
    h.single_task<class kernel_name6>(
        []() [[intel::num_simd_work_items(42)]]{});

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name7() #0 {{.*}} !num_simd_work_items ![[NUM2]]
    Functor1<2> f1;
    h.single_task<class kernel_name7>(f1);

    // Test attribute is not propagated.
    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name8()
    // CHECK-NOT: !num_simd_work_items
    h.single_task<class kernel_name8>(
        []() { foo1(); });
    
    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name9() #0 {{.*}} !no_global_work_offset ![[NUM:[0-9]+]]
    Foo2 boo2;
    h.single_task<class kernel_name9>(boo2);

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name10() #0 {{.*}} ![[NUM0:[0-9]+]]
    h.single_task<class kernel_name10>(
        []() [[intel::no_global_work_offset(0)]]{});

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name11() #0 {{.*}} !no_global_work_offset ![[NUM]]
    Functor2<1> f2;
    h.single_task<class kernel_name11>(f2);

    // Test attribute is not propagated.
    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name12()
    // CHECK-NOT: !no_global_work_offset
    h.single_task<class kernel_name12>(
        []() { foo2(); });

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name13() #0 {{.*}} !max_global_work_dim ![[NUM1]]
    Foo3 boo3;
    h.single_task<class kernel_name13>(boo3);

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name14() #0 {{.*}} !max_global_work_dim ![[NUM1]]
    h.single_task<class kernel_name14>(
        []() [[intel::max_global_work_dim(1)]]{});

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name15() #0 {{.*}} !max_global_work_dim ![[NUM2]]
    Functor3<2> f3;
    h.single_task<class kernel_name15>(f3);

    // Test attribute is not propagated.
    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name16()
    // CHECK-NOT: !max_global_work_dim
    h.single_task<class kernel_name16>(
        []() { foo3(); });
  });
  return 0;
}

// CHECK: ![[NUM]] = !{}
// CHECK: ![[NUM1]] = !{i32 1}
// CHECK: ![[NUM42]] = !{i32 42}
// CHECK: ![[NUM2]] = !{i32 2}
// CHECK-NOT: ![[NUM0]]  = !{i32 0}
