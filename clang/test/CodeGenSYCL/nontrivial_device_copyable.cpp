// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -internal-isystem %S/Inputs -disable-llvm-passes -sycl-std=2020 -emit-llvm %s -o - | FileCheck %s

// Tests that SYCL kernel arguments with non-trivially copyable types are
// passed by-valued.

struct NontriviallyCopyable {
  int I;
  NontriviallyCopyable(int I) : I(I) {}
  NontriviallyCopyable(const NontriviallyCopyable &X) : I(X.I) {}
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &KernelFunc) {
  KernelFunc();
}

__attribute__((sycl_device)) void device_func(NontriviallyCopyable X) {
  (void)X;
}

int main() {
  NontriviallyCopyable NontriviallyCopyableObject{10};
  kernel<class kernel_name>(
    [=]() {
      (void)NontriviallyCopyableObject;
    });
}

// SYCL kernels
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name(%struct.NontriviallyCopyable* byval(%struct.NontriviallyCopyable)
// CHECK-NOT: define {{.*}}spir_func void @{{.*}}device_func({{.*}}byval(%struct.NontriviallyCopyable)
