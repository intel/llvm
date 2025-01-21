// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=SPIR
// RUN: %clang_cc1 -triple x86_64 -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=X86

// REQUIRES: x86-registered-target

__attribute__((sycl_device)) void foo(__attribute__((opencl_global)) int *);
__attribute__((sycl_device)) void foo(__attribute__((opencl_local)) int *);
__attribute__((sycl_device)) void foo(__attribute__((opencl_private)) int *);
__attribute__((sycl_device)) void foo(int *);

// SPIR: declare spir_func void @_Z3fooPU3AS1i(ptr addrspace(1) noundef)
// SPIR: declare spir_func void @_Z3fooPU3AS3i(ptr addrspace(3) noundef)
// SPIR: declare spir_func void @_Z3fooPU3AS0i(ptr noundef)
// SPIR: declare spir_func void @_Z3fooPi(ptr addrspace(4) noundef)

// X86: declare void @_Z3fooPU8SYglobali(ptr noundef)
// X86: declare void @_Z3fooPU7SYlocali(ptr noundef)
// X86: declare void @_Z3fooPU9SYprivatei(ptr noundef)
// X86: declare void @_Z3fooPi(ptr noundef)

__attribute__((sycl_device)) void test() {
  __attribute__((opencl_global)) int *glob;
  __attribute__((opencl_local)) int *loc;
  __attribute__((opencl_private)) int *priv;
  int *def;
  foo(glob);
  foo(loc);
  foo(priv);
  foo(def);
}
