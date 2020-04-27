// RUN: %clang_cc1 -fsycl -fsycl-is-device -disable-llvm-passes -triple spir64-unknown-unknown-sycldevice -emit-llvm -o - %s | FileCheck %s

class Functor16 {
public:
  [[cl::intel_reqd_sub_group_size(16)]] void operator()() {}
};

[[cl::intel_reqd_sub_group_size(8)]] void foo() {}

class Functor {
public:
  void operator()() {
    foo();
  }
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar() {
  Functor16 f16;
  kernel<class kernel_name1>(f16);

  Functor f;
  kernel<class kernel_name2>(f);

  kernel<class kernel_name3>(
  []() [[cl::intel_reqd_sub_group_size(4)]] {});
}

// CHECK: define spir_kernel void @{{.*}}kernel_name1(%class.{{.*}}.Functor16* byval(%class.{{.*}}.Functor16) align 1 %_arg_kernelObject) {{.*}} !intel_reqd_sub_group_size ![[SGSIZE16:[0-9]+]]
// CHECK: define spir_kernel void @{{.*}}kernel_name2(%class.{{.*}}.Functor* byval(%class.{{.*}}.Functor) align 1 %_arg_kernelObject) {{.*}} !intel_reqd_sub_group_size ![[SGSIZE8:[0-9]+]]
// CHECK: define spir_kernel void @{{.*}}kernel_name3(%"class.{{.*}}.anon"* byval(%"class.{{.*}}.anon") align 1 %_arg_kernelObject) {{.*}} !intel_reqd_sub_group_size ![[SGSIZE4:[0-9]+]]
// CHECK: ![[SGSIZE16]] = !{i32 16}
// CHECK: ![[SGSIZE8]] = !{i32 8}
// CHECK: ![[SGSIZE4]] = !{i32 4}

