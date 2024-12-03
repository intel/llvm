// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
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

  // CHECK: define {{.*}}spir_kernel void @_ZTSZ4test9enum_typeE15kernel_function(i32 noundef %_arg_val)

  // CHECK: getelementptr inbounds nuw %class.anon, ptr addrspace(4)
  // CHECK: call spir_func void @_ZZ4test9enum_typeENKUlvE_clEv(ptr addrspace(4) {{[^,]*}} %{{.+}})

  test( enum_type::B );
  return 0;
}
