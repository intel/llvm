// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -sycl-std=2020 -emit-llvm %s -o - | FileCheck %s

struct NontriviallyCopyable {
  int i;
  NontriviallyCopyable(int _i) : i(_i) {}
  NontriviallyCopyable(const NontriviallyCopyable &x) : i(x.i) {}
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

__attribute__((sycl_device)) void device_func(NontriviallyCopyable ntc) {
  (void)ntc;
}

int main() {
  NontriviallyCopyable nontrivial{10};
  kernel<class kernel_name>(
    [=]() {
      (void)nontrivial;
    });
}

// SYCL kernels
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name(%struct._ZTS20NontriviallyCopyable.NontriviallyCopyable* byval(%struct._ZTS20NontriviallyCopyable.NontriviallyCopyable) {{.*}}) {{.*}}
// CHECK-NOT: define {{.*}}spir_func void @{{.*}}device_func({{.*}}byval(%struct._ZTS20NontriviallyCopyable.NontriviallyCopyable){{.*}}) {{.*}}
