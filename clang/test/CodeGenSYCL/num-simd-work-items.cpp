// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

class Foo {
public:
  [[intelfpga::num_simd_work_items(1)]] void operator()() const {}
};

template <int SIZE>
class Functor {
public:
  [[intelfpga::num_simd_work_items(SIZE)]] void operator()() const {}
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

void bar() {
  Foo boo;
  kernel<class kernel_name1>(boo);

  kernel<class kernel_name2>(
  []() [[intelfpga::num_simd_work_items(42)]] {});

  Functor<2> f;
  kernel<class kernel_name3>(f);
}

// CHECK: define spir_kernel void @{{.*}}kernel_name1() {{.*}} !num_simd_work_items ![[NUM1:[0-9]+]]
// CHECK: define spir_kernel void @{{.*}}kernel_name2() {{.*}} !num_simd_work_items ![[NUM42:[0-9]+]]
// CHECK: define spir_kernel void @{{.*}}kernel_name3() {{.*}} !num_simd_work_items ![[NUM2:[0-9]+]]
// CHECK: ![[NUM1]] = !{i32 1}
// CHECK: ![[NUM42]] = !{i32 42}
// CHECK: ![[NUM2]] = !{i32 2}
