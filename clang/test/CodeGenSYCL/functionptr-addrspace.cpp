// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -triple spir64 -verify -emit-llvm %s -o - | FileCheck %s

// expected-no-diagnostics

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

// CHECK: define {{.*}}spir_func{{.*}}invoke_function{{.*}}(ptr noundef %fptr, ptr addrspace(4) noundef %ptr)
void invoke_function(int (*fptr)(), int *ptr) {}

int f() { return 0; }

int main() {
  kernel_single_task<class fake_kernel>([=]() {
    int (*p)() = f;
    int (&r)() = *p;
    int a = 10;
    invoke_function(p, &a);
    invoke_function(r, &a);
    invoke_function(f, &a);
  });

  // Test function pointer as kernel argument. Function pointers should have program address space i.e. 0.

  int (*fptr)();
  int *ptr;

  // define dso_local spir_kernel void @{{.*}}fake_kernel_2{{.*}}(i32 ()* align 4 %_arg_fptr, i32 addrspace(1)* align 4 %_arg_ptr)
  kernel_single_task<class fake_kernel_2>([=]() {
    invoke_function(fptr, ptr);
  });

  return 0;
}
