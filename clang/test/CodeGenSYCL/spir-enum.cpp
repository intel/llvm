// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

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
  // CHECK: call spir_func void @"_ZZ4test9enum_typeENK3$_0clEv"(%"class.{{.*}}.anon" addrspace(4)* %4)


  test( enum_type::B );
  return 0;
}
