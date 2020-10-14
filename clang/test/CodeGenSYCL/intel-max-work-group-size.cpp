// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

class Foo {
public:
  [[intel::max_work_group_size(1, 1, 1)]] void operator()() const {}
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

void bar() {
  Foo boo;
  kernel<class kernel_name1>(boo);

  kernel<class kernel_name2>(
      []() [[intel::max_work_group_size(8, 8, 8)]]{});
}

// CHECK: define spir_kernel void @{{.*}}kernel_name1() {{.*}} !max_work_group_size ![[NUM1:[0-9]+]]
// CHECK: define spir_kernel void @{{.*}}kernel_name2() {{.*}} !max_work_group_size ![[NUM8:[0-9]+]]
// CHECK: ![[NUM1]] = !{i32 1, i32 1, i32 1}
// CHECK: ![[NUM8]] = !{i32 8, i32 8, i32 8}
