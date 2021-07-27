// RUN: %clang_cc1 -fsycl-is-device %s -emit-llvm -sycl-std=2017 -triple spir64-unknown-unknown-sycldevice -o - | FileCheck %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  int *a;
  int *b;
  int *c;
  kernel<class kernel_restrict>(
      [ a, b, c ]() [[intel::kernel_args_restrict]] { c[0] = a[0] + b[0]; });
  // CHECK: define {{.*}}spir_kernel {{.*}}kernel_restrict(i32 addrspace(1)* noalias %{{.*}}, i32 addrspace(1)* noalias %{{.*}}, i32 addrspace(1)* noalias %{{.*}})

  int *d;
  int *e;
  int *f;

  kernel<class kernel_norestrict>(
      [d, e, f]() { f[0] = d[0] + e[0]; });
  // CHECK: define {{.*}}spir_kernel {{.*}}kernel_norestrict(i32 addrspace(1)* %{{.*}}, i32 addrspace(1)* %{{.*}}, i32 addrspace(1)* %{{.*}})

  int g = 42;
  kernel<class kernel_restrict_other_types>(
      [ a, b, c, g ]() [[intel::kernel_args_restrict]] { c[0] = a[0] + b[0] + g; });
  // CHECK: define {{.*}}spir_kernel {{.*}}kernel_restrict_other_types(i32 addrspace(1)* noalias %{{.*}}, i32 addrspace(1)* noalias %{{.*}}, i32 addrspace(1)* noalias %{{.*}}, i32 %{{.*}})
}
