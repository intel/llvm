// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

class Functor32x16x16 {
public:
  [[cl::reqd_work_group_size(32, 16, 16)]] void operator()() {}
};

[[cl::reqd_work_group_size(8, 1, 1)]] void f8x1x1() {}

class Functor {
public:
  void operator()() {
    f8x1x1();
  }
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar() {
  Functor32x16x16 f32x16x16;
  kernel<class kernel_name1>(f32x16x16);

  Functor f;
  kernel<class kernel_name2>(f);

  kernel<class kernel_name3>(
      []() [[cl::reqd_work_group_size(8, 8, 8)]]{});
}

// CHECK: define spir_kernel void @{{.*}}kernel_name1(%class.{{.*}}.Functor32x16x16* byval(%class.{{.*}}.Functor32x16x16) align 1 %_arg_kernelObject) {{.*}} !reqd_work_group_size ![[WGSIZE32:[0-9]+]]
// CHECK: define spir_kernel void @{{.*}}kernel_name2(%class.{{.*}}.Functor* byval(%class.{{.*}}.Functor) align 1 %_arg_kernelObject) {{.*}} !reqd_work_group_size ![[WGSIZE8:[0-9]+]]
// CHECK: define spir_kernel void @{{.*}}kernel_name3(%"class.{{.*}}.anon"* byval(%"class.{{.*}}.anon") align 1 %_arg_kernelObject) {{.*}} !reqd_work_group_size ![[WGSIZE88:[0-9]+]]
// CHECK: ![[WGSIZE32]] = !{i32 16, i32 16, i32 32}
// CHECK: ![[WGSIZE8]] = !{i32 1, i32 1, i32 8}
// CHECK: ![[WGSIZE88]] = !{i32 8, i32 8, i32 8}
