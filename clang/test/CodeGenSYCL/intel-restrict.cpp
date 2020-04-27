// RUN: %clang_cc1 -fsycl -fsycl-is-device %s -emit-llvm -triple spir64-unknown-unknown-sycldevice -o - | FileCheck %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  int *a;
  int *b;
  int *c;
  kernel<class kernel_restrict>(
      [a,b,c]() [[intel::kernel_args_restrict]] { c[0] = a[0] + b[0];});
// CHECK: define spir_kernel {{.*}}kernel_restrict(%"class.{{.*}}.anon"* byval(%"class.{{.*}}.anon") align 8 %_arg_kernelObject, i32 addrspace(1)* noalias %{{.*}}, i32 addrspace(1)* noalias %{{.*}}, i32 addrspace(1)* noalias %{{.*}})

  int *d;
  int *e;
  int *f;

  kernel<class kernel_norestrict>(
      [d,e,f]() { f[0] = d[0] + e[0];});
// CHECK: define spir_kernel {{.*}}kernel_norestrict(%"class.{{.*}}.anon.0"* byval(%"class.{{.*}}.anon.0") align 8 %_arg_kernelObject, i32 addrspace(1)* %{{.*}}, i32 addrspace(1)* %{{.*}}, i32 addrspace(1)* %{{.*}})

  int g = 42;
  kernel<class kernel_restrict_other_types>(
      [a,b,c,g]() [[intel::kernel_args_restrict]] { c[0] = a[0] + b[0] + g;});
// CHECK: define spir_kernel {{.*}}kernel_restrict_other_types(%"class.{{.*}}.anon.1"* byval(%"class.{{.*}}.anon.1") align 8 %_arg_kernelObject, i32 addrspace(1)* noalias %{{.*}}, i32 addrspace(1)* noalias %{{.*}}, i32 addrspace(1)* noalias %{{.*}}, i32 %{{.*}})
}
