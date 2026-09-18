// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=SPIR
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsycl-is-host -emit-llvm %s -o - | FileCheck %s --check-prefix=ITANIUM
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fsycl-is-host -emit-llvm %s -o - | FileCheck %s --check-prefix=MS

// REQUIRES: x86-registered-target

<<<<<<< HEAD
__attribute__((sycl_device)) void foo(__attribute__((opencl_global)) int *);
__attribute__((sycl_device)) void foo(__attribute__((opencl_local)) int *);
__attribute__((sycl_device)) void foo(__attribute__((opencl_private)) int *);
__attribute__((sycl_device)) void foo(int *);

// SPIR: declare spir_func void @_Z3fooPU3AS1i(ptr addrspace(1) noundef)
// SPIR: declare spir_func void @_Z3fooPU3AS3i(ptr addrspace(3) noundef)
// SPIR: declare spir_func void @_Z3fooPU3AS0i(ptr noundef)
// SPIR: declare spir_func void @_Z3fooPi(ptr addrspace(4) noundef)
=======
void foo(int [[clang::sycl_global]] *);
void foo(int [[clang::sycl_local]] *);
void foo(int [[clang::sycl_private]] *);
void foo(int [[clang::sycl_generic]] *);
void foo(int [[clang::sycl_constant]] *);
void foo(int *);

// SPIR: declare spir_func void @_Z3fooPU3AS1i(ptr addrspace(1) noundef) #1
// SPIR: declare spir_func void @_Z3fooPU3AS3i(ptr addrspace(3) noundef) #1
// SPIR: declare spir_func void @_Z3fooPU3AS0i(ptr noundef) #1
// SPIR: declare spir_func void @_Z3fooPU3AS4i(ptr addrspace(4) noundef) #1
// SPIR: declare spir_func void @_Z3fooPU3AS2i(ptr addrspace(2) noundef) #1
// SPIR: declare spir_func void @_Z3fooPi(ptr addrspace(4) noundef) #1
>>>>>>> c476f80e8f6d7cdaf462230611e3b07da5835da6

// ITANIUM: declare void @_Z3fooPU8SYglobali(ptr noundef)
// ITANIUM: declare void @_Z3fooPU7SYlocali(ptr noundef)
// ITANIUM: declare void @_Z3fooPU9SYprivatei(ptr noundef)
// ITANIUM: declare void @_Z3fooPU9SYgenerici(ptr noundef)
// ITANIUM: declare void @_Z3fooPU10SYconstanti(ptr noundef)
// ITANIUM: declare void @_Z3fooPi(ptr noundef)

// MS: declare dso_local void @"?foo@@YAXPEAU?$_ASSYglobal@$$CAH@__clang@@@Z"
// MS: declare dso_local void @"?foo@@YAXPEAU?$_ASSYlocal@$$CAH@__clang@@@Z"
// MS: declare dso_local void @"?foo@@YAXPEAU?$_ASSYprivate@$$CAH@__clang@@@Z"
// MS: declare dso_local void @"?foo@@YAXPEAU?$_ASSYgeneric@$$CAH@__clang@@@Z"
// MS: declare dso_local void @"?foo@@YAXPEAU?$_ASSYconstant@$$CAH@__clang@@@Z"
// MS: declare dso_local void @"?foo@@YAXPEAH@Z"

[[clang::sycl_external]] void test() {
  int [[clang::sycl_global]] *glob;
  int [[clang::sycl_local]] *loc;
  int [[clang::sycl_private]] *priv;
  int [[clang::sycl_generic]] *gen;
  int [[clang::sycl_constant]] *cnst;
  int *def;
  foo(glob);
  foo(loc);
  foo(priv);
  foo(gen);
  foo(cnst);
  foo(def);
}
