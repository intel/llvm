// RUN: %clang %s -S -emit-llvm --sycl -o - | FileCheck %s
#include "CL/sycl.hpp"
// CHECK: %opencl.pipe_wo_t
// CHECK: %opencl.pipe_ro_t

class SomePipe;
void foo() {
  using Pipe = cl::sycl::pipe<SomePipe, int>;
  // CHECK: %WPipe = alloca %opencl.pipe_wo_t
  Pipe::write(42);
  // CHECK: %RPipe = alloca %opencl.pipe_ro_t
  int a = Pipe::read();
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    foo();
  });
  return 0;
}

