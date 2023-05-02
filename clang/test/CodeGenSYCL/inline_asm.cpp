// RUN: %clang_cc1 -O1 -fsycl-is-device -triple spir64-unknown-unknown -opaque-pointers -emit-llvm -x c++ %s -o - | FileCheck %s

class kernel;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [100 x i32], align 4
  // CHECK: %[[IDX:.*]] = addrspacecast ptr %[[ARRAY_A]] to ptr addrspace(4)
  int a[100], i = 0;
  // CHECK: call void asm sideeffect
  // CHECK: ".decl V52 v_type=G type=d num_elts=16 align=GRF
  // CHECK: svm_gather.4.1 (M1, 16) $0.0 V52.0
  // CHECK: add(M1, 16) V52(0, 0)<1> V52(0, 0)<1; 1, 0> 0x1
  // CHECK: svm_scatter.4.1 (M1, 16) $0.0 V52.0",
  // CHECK: "rw"(ptr addrspace(4) %[[IDX]])
  // TODO: nonnull attribute missing?
  asm volatile(".decl V52 v_type=G type=d num_elts=16 align=GRF\n"
               "svm_gather.4.1 (M1, 16) %0.0 V52.0\n"
               "add(M1, 16) V52(0, 0)<1> V52(0, 0)<1; 1, 0> 0x1\n"
               "svm_scatter.4.1 (M1, 16) %0.0 V52.0"
               :
               : "rw"(&a[i]));
}

int main() {
  kernel_single_task<class kernel>([]() {});
  return 0;
}
