// RUN: DISABLE_INFER_AS=1 %clang_cc1 -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device  -disable-llvm-optzns -disable-llvm-passes -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-OLD
// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device  -disable-llvm-optzns -disable-llvm-passes -S -emit-llvm -x c++ %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-NEW

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

enum enum_type: int {
    A = 0,
    B = 1,
};

void test(enum_type val)
{
  kernel_single_task<class kernel_function>([=]() {
  //expected-warning@+1{{expression result unused}}
  val;
  });
}

int main() {

  // CHECK: define spir_kernel void @_ZTSZ4test9enum_typeE15kernel_function(i32 %_arg_)

  // CHECK: getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"*
  // CHECK-OLD: call spir_func void @"_ZZ4test9enum_typeENK3$_0clEv"(%"class.{{.*}}.anon"* %0)
  // CHECK-NEW: call spir_func void @"_ZZ4test9enum_typeENK3$_0clEv"(%"class.{{.*}}.anon" addrspace(4)* %4)


  test( enum_type::B );
  return 0;
}

// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows-msvc
