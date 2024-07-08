// RUN: %clang_cc1 -fsycl-is-device %s -emit-llvm -triple spir64-unknown-unknown -o - | FileCheck %s

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
  // CHECK: define {{.*}}spir_kernel {{.*}}kernel_restrict(ptr addrspace(1) noalias noundef align 4 %{{.*}}, ptr addrspace(1) noalias noundef align 4 %{{.*}}, ptr addrspace(1) noalias noundef align 4 %{{.*}})

  int *d;
  int *e;
  int *f;

  kernel<class kernel_norestrict>(
      [d, e, f]() { f[0] = d[0] + e[0]; });
  // CHECK: define {{.*}}spir_kernel {{.*}}kernel_norestrict(ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}})

  int g = 42;
  kernel<class kernel_restrict_other_types>(
      [ a, b, c, g ]() [[intel::kernel_args_restrict]] { c[0] = a[0] + b[0] + g; });
  // CHECK: define {{.*}}spir_kernel {{.*}}kernel_restrict_other_types(ptr addrspace(1) noalias noundef align 4 %{{.*}}, ptr addrspace(1) noalias noundef align 4 %{{.*}}, ptr addrspace(1) noalias noundef align 4 %{{.*}}, i32 noundef %{{.*}})
}
