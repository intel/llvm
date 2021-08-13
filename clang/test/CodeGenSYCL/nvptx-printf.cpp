// RUN: %clang_cc1 -fsycl-is-device -triple nvptx64-nvidia-cuda-sycldevice -std=c++11 -S -emit-llvm -x c++ %s -o - | FileCheck %s

#include "Inputs/sycl.hpp"

#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

static const CONSTANT char format_2[] = "Hello! %d %f\n";

int main() {
  // Make sure that device printf is dispatched to CUDA's vprintf syscall.
  // CHECK: alloca %printf_args
  // CHECK: call i32 @vprintf
  cl::sycl::kernel_single_task<class first_kernel>([]() { cl::sycl::ext::oneapi::experimental::printf(format_2, 123, 1.23); });
}
