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


class Foo4 {
public:
  [[intel::reqd_sub_group_size(16)]] void operator()() const {}
};

[[intel::reqd_sub_group_size(8)]] void foo4() {}

class Functor4 {
public:
  void operator()() const {
    foo();
  }
};

template <int SIZE>
class Functor5 {
public:
  [[intel::reqd_sub_group_size(SIZE)]] void operator()() const {}
};

class Foo5 {
public:
  [[sycl::reqd_work_group_size(32, 16, 16)]] void operator()() const {}
};

[[sycl::reqd_work_group_size(8, 1, 1)]] void foo5() {}

class Functor6 {
public:
  void operator()() const {
    foo5();
  }
};

template <int SIZE, int SIZE1, int SIZE2>
class Functor7 {
public:
  [[sycl::reqd_work_group_size(SIZE, SIZE1, SIZE2)]] void operator()() const {}
};

class Foo6 {
public:
  [[intel::max_work_group_size(32, 16, 16)]] void operator()() const {}
};

[[intel::max_work_group_size(8, 1, 1)]] void foo6() {}

class Functor8 {
public:
  void operator()() const {
    foo6();
  }
};

template <int SIZE, int SIZE1, int SIZE2>
class Functor9 {
public:
  [[intel::max_work_group_size(SIZE, SIZE1, SIZE2)]] void operator()() const {}
};

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

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name17() #0 {{.*}} !intel_reqd_sub_group_size ![[NUM16:[0-9]+]]
    Foo4 boo4;
    h.single_task<class kernel_name17>(boo4);

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name18() #0 {{.*}} !intel_reqd_sub_group_size ![[NUM1]]
    h.single_task<class kernel_name18>(
        []() [[intel::reqd_sub_group_size(1)]]{});

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name19() #0 {{.*}} !intel_reqd_sub_group_size ![[NUM2]]
    Functor5<2> f5;
    h.single_task<class kernel_name19>(f5);

    // Test attribute is not propagated.
    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name20()
    // CHECK-NOT: !reqd_sub_group_size
    Functor4 f4;
    h.single_task<class kernel_name20>(f4);

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name21() #0 {{.*}} !reqd_work_group_size ![[NUM32:[0-9]+]]
    Foo5 boo5;
    h.single_task<class kernel_name21>(boo5);

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name22() #0 {{.*}} !reqd_work_group_size ![[NUM88:[0-9]+]]
    h.single_task<class kernel_name22>(
        []() [[sycl::reqd_work_group_size(8, 8, 8)]]{});

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name23() #0 {{.*}} !reqd_work_group_size ![[NUM22:[0-9]+]]
    Functor7<2, 2, 2> f7;
    h.single_task<class kernel_name23>(f7);

    // Test attribute is not propagated.
    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name24()
    // CHECK-NOT: !reqd_work_group_size
    Functor6 f6;
    h.single_task<class kernel_name24>(f6);

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name25() #0 {{.*}} !max_work_group_size ![[NUM32]]
    Foo6 boo6;
    h.single_task<class kernel_name25>(boo6);

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name26() #0 {{.*}} !max_work_group_size ![[NUM88]]
    h.single_task<class kernel_name26>(
        []() [[intel::max_work_group_size(8, 8, 8)]]{});

    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name27() #0 {{.*}} !max_work_group_size ![[NUM22]]
    Functor9<2, 2, 2> f9;
    h.single_task<class kernel_name27>(f9);

    // Test attribute is not propagated.
    // CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name28()
    // CHECK-NOT: !max_work_group_size
    Functor8 f8;
    h.single_task<class kernel_name28>(f8);
  });
  return 0;
}

// CHECK: ![[NUM]] = !{}
// CHECK: ![[NUM1]] = !{i32 1}
// CHECK: ![[NUM42]] = !{i32 42}
// CHECK: ![[NUM2]] = !{i32 2}
// CHECK-NOT: ![[NUM0]]  = !{i32 0}
// CHECK: ![[NUM16]] = !{i32 16}
// CHECK: ![[NUM32]] = !{i32 16, i32 16, i32 32}
// CHECK: ![[NUM88]] = !{i32 8, i32 8, i32 8}
// CHECK: ![[NUM22]] = !{i32 2, i32 2, i32 2}
