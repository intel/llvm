// RUN: %clang_cc1 -triple spir64-unknown-linux -fsycl-is-device -disable-llvm-passes -emit-llvm -x c++ %s -o - | FileCheck %s

struct A {
  int B[42];
};

const char *ret_char() {
  return "N";
}
// CHECK: ret ptr addrspace(4) addrspacecast (ptr addrspace(1) @.str to ptr addrspace(4))

const char *ret_arr() {
  static const char Arr[42] = {0};
  return Arr;
}
// CHECK: ret ptr addrspace(4) addrspacecast (ptr addrspace(1) @{{.*}}ret_arr{{.*}}Arr to ptr addrspace(4))

const char &ret_ref() {
  static const char a = 'A';
  return a;
}
// CHECK: ret ptr addrspace(4) addrspacecast (ptr addrspace(1) @{{.*}}ret_ref{{.*}} to ptr addrspace(4))

A ret_agg() {
  A a;
  return a;
}
// CHECK: define {{.*}}spir_func void @{{.*}}ret_agg{{.*}}(ptr addrspace(4) dead_on_unwind noalias writable sret(%struct{{.*}}.A) align 4 %agg.result)

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
