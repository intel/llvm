// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device -disable-llvm-passes -S -emit-llvm -x c++ %s -o - | FileCheck %s

struct A {
  int B[42];
};

const char *ret_char() {
  return "N";
}
// CHECK: ret i8 addrspace(4)* addrspacecast (i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str, i64 0, i64 0) to i8 addrspace(4)*)

const char *ret_arr() {
  static char Arr[42];
  return Arr;
}
// CHECK: ret i8 addrspace(4)* getelementptr inbounds ([42 x i8], [42 x i8] addrspace(4)* addrspacecast ([42 x i8] addrspace(1)* @{{.*}}ret_arr{{.*}}Arr to [42 x i8] addrspace(4)*), i64 0, i64 0)

const char &ret_ref() {
  static char a = 'A';
  return a;
}
// CHECK: ret i8 addrspace(4)* addrspacecast (i8 addrspace(1)* @{{.*}}ret_ref{{.*}} to i8 addrspace(4)*)

A ret_agg() {
  A a;
  return a;
}
// CHECK: define spir_func void @{{.*}}ret_agg{{.*}}(%struct.{{.*}}.A addrspace(4)* noalias sret %agg.result)

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
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
