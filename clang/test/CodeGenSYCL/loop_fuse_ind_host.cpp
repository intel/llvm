// RUN: %clang_cc1 -fsycl -fsycl-is-host -triple -x86_64-unknown-linux-gnu -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

template <int SIZE>
class KernelFunctor5 {
public:
  [[intel::loop_fuse_independent(SIZE)]] void operator()() const {}
};

[[intel::loop_fuse_independent]] void func1() {}
[[intel::loop_fuse_independent(0)]] void func2() {}
[[intel::loop_fuse_independent(1)]] void func3() {}
[[intel::loop_fuse_independent(10)]] void func4() {}

void foo() {

  KernelFunctor5<5> f5;
  kernel<class kernel_name_1>(f5);

  kernel<class kernel_name_2>(
      []() [[intel::loop_fuse_independent(10)]]{});

  kernel<class kernel_name_3>(
      []() { func4(); });
}

// CHECK: define void @{{.*}}func1{{.*}} !loop_fuse ![[LFI1:[0-9]+]]
// CHECK: define void @{{.*}}func2{{.*}} !loop_fuse ![[LFI0:[0-9]+]]
// CHECK: define void @{{.*}}func3{{.*}} !loop_fuse ![[LFI1]]
// CHECK: define void @{{.*}}func4{{.*}} !loop_fuse ![[LFI10:[0-9]+]]
// CHECK: define linkonce_odr void @{{.*}}KernelFunctor5{{.*}} !loop_fuse ![[LFI5:[0-9]+]]
// CHECK: define internal void @"{{.*}}foo{{.*}}"(%class.anon* %this){{.*}}!loop_fuse ![[LFI10]]
// CHECK: define internal void @"{{.*}}foo{{.*}}"(%class.anon.0* %this)
// CHECK-NOT: !loop_fuse
// CHECK-SAME: {

// CHECK: ![[LFI1]] = !{i32 1, i32 1}
// CHECK: ![[LFI0]] = !{i32 0, i32 1}
// CHECK: ![[LFI10]] = !{i32 10, i32 1}
// CHECK: ![[LFI5]] = !{i32 5, i32 1}
