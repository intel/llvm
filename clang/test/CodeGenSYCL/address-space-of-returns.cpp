// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

struct A {
  int B[42];
};

const char *ret_char() {
  return "N";
}
// CHECK: ret i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(1)* @.str, i64 0, i64 0) to i8 addrspace(4)*)

const char *ret_arr() {
  const static char Arr[36] = "Carrots, cabbage, radish, potatoes!";
  return Arr;
}

// CHECK: ret i8 addrspace(4)* getelementptr inbounds ([36 x i8], [36 x i8] addrspace(4)* addrspacecast ([36 x i8] addrspace(1)* @{{.*}}ret_arr{{.*}}Arr to [36 x i8] addrspace(4)*), i64 0, i64 0)

const char &ret_ref() {
  const static char a = 'A';
  return a;
}
// CHECK: ret i8 addrspace(4)* addrspacecast (i8 addrspace(1)* @{{.*}}ret_ref{{.*}} to i8 addrspace(4)*)

A ret_agg() {
  A a;
  return a;
}
// CHECK: define spir_func void @{{.*}}ret_agg{{.*}}(%struct.{{.*}}.A addrspace(4)* {{.*}} %agg.result)

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
