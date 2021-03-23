// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice -fsycl-is-device -disable-llvm-passes -emit-llvm -x c++ %s -o - | FileCheck %s

struct A {
  int B[42];
};

const char *ret_char() {
  return "N";
}
// CHECK: ret i8 addrspace(4)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(4)* addrspacecast ([2 x i8] addrspace(1)* @.str to [2 x i8] addrspace(4)*), i64 0, i64 0)

const char *ret_arr() {
  static const char Arr[42] = {0};
  return Arr;
}
// CHECK: ret i8 addrspace(4)* getelementptr inbounds ([42 x i8], [42 x i8] addrspace(4)* addrspacecast ([42 x i8] addrspace(1)* @{{.*}}ret_arr{{.*}}Arr to [42 x i8] addrspace(4)*), i64 0, i64 0)

const char &ret_ref() {
  static const char a = 'A';
  return a;
}
// CHECK: ret i8 addrspace(4)* addrspacecast (i8 addrspace(1)* @{{.*}}ret_ref{{.*}} to i8 addrspace(4)*)

A ret_agg() {
  A a;
  return a;
}
// CHECK: define {{.*}}spir_func void @{{.*}}ret_agg{{.*}}(%struct{{.*}}.A addrspace(4)* noalias sret(%struct{{.*}}.A) align 4 %agg.result)

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel>([]() {
    ret_char();
    ret_arr();
    ret_ref();
    ret_agg();
  });
  return 0;
}
