// RUN: DISABLE_INFER_AS=1 %clang_cc1 -triple spir64-unknown-linux-sycldevice  -std=c++11 -fsycl-is-device -disable-llvm-passes -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-LEGACY
// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice  -std=c++11 -fsycl-is-device -disable-llvm-passes -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NEW

void test() {
  static const int foo = 0x42;
  // CHECK-LEGACY: @_ZZ4testvE3foo = internal constant i32 66, align 4
  // CHECK-NEW:    @_ZZ4testvE3foo = internal addrspace(1) constant i32 66, align 4

  // CHECK: @[[STR:[.a-zA-Z0-9_]+]] = private unnamed_addr constant [14 x i8] c"Hello, world!\00", align 1
  int i = 0;
  int *pptr = &i;
  // CHECK-LEGACY: store i32* %i, i32** %pptr
  // CHECK-NEW: %[[GEN:[0-9]+]] = addrspacecast i32* %i to i32 addrspace(4)*
  // CHECK-NEW: store i32 addrspace(4)* %[[GEN]], i32 addrspace(4)** %pptr
  *pptr = foo;

  const char *str = "Hello, world!";
  // CHECK-LEGACY: store i8* getelementptr inbounds ([14 x i8], [14 x i8]* @[[STR]], i64 0, i64 0), i8** %{{.*}}, align 8
  // CHECK-NEW: store i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([14 x i8], [14 x i8]* @[[STR]], i64 0, i64 0) to i8 addrspace(4)*), i8 addrspace(4)** %{{.*}}, align 8

  i = str[0];
}


template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}


int main() {
  kernel_single_task<class fake_kernel>([]() { test(); });
  return 0;
}
