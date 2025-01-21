// RUN: %clang_cc1 %s -fsycl-is-host -D__sycl_cuda_host \
// RUN:   -internal-isystem %S/../SemaCUDA/Inputs \
// RUN:   -fsycl-targets=nvptx64-nvidia-cuda -triple x86_64-unknown-linux \
// RUN:   -emit-llvm -o - -verify -verify-ignore-unexpected=note
// RUN: %clang_cc1 %s -fsycl-is-host -fcuda-is-device -D__cuda_device \
// RUN:   -internal-isystem %S/../SemaCUDA/Inputs \
// RUN:   -fsycl-targets=nvptx64-nvidia-cuda -triple x86_64-unknown-linux \
// RUN:   -emit-llvm -o - -verify -verify-ignore-unexpected=note
// RUN: %clang_cc1 %s -fsycl-is-device -D__sycl_device \
// RUN:   -internal-isystem %S/../SemaCUDA/Inputs \
// RUN:   -fsycl-targets=nvptx64-nvidia-cuda -triple x86_64-unknown-linux\
// RUN:   -emit-llvm -o - -verify -verify-ignore-unexpected=note

// This tests the errors emitted by SEMA in case of SYCL-CUDA compilation.

// | F  | T  |     |
// |----+----+-----+
// | d  | h  | (A) |
// | g  | h  | (B) |
// | g  | g  | (C) |
// | d  | g  | (D) |
// | hd | g  | (E) |

#include "cuda.h"

__host__ void host_fn() {}  //#HOST_FN
__global__ void kernel() {} //#KERNEL

// (A)
// expected-note@#HOST_FN {{candidate function not viable: call to __host__ function from __device__ function}}
__device__ void dev_fn_0() {
  host_fn();
// expected-error@-1 {{no matching function for call to 'host_fn'}}
}

// (B)
// expected-note@#HOST_FN {{candidate function not viable: call to __host__ function from __global__ function}}
__global__ void kernel_0() { host_fn(); }
// expected-error@-1 {{no matching function for call to 'host_fn'}}
void func_0(void) { kernel_0<<<1, 1>>>(); }

// (C)
// expected-note@#KERNEL {{candidate function not viable: call to __global__ function from __global__ function}}
__global__ void kernel_1() { kernel(); }
// expected-error@-1 {{no matching function for call to 'kernel'}}
void func_2(void) { kernel_1<<<1, 1>>>(); }

// (D)
__device__ void device_func() { kernel<<<1, 1>>>();}
// expected-error@-1 {{reference to __global__ function 'kernel' in __device__ function}}
void func_3(void) { device_func(); }

#if defined(__sycl_cuda_host) || defined(__sycl_device)
// expected-error@* {{reference to __host__ function 'cudaConfigureCall' in __device__ function}}
#endif
#ifdef __sycl_cuda_host
// expected-note@*:* {{'cudaConfigureCall' declared here}}
#endif

// (E)
#if defined(__sycl_device)
__host__ __device__ void hostdevice_func() { kernel<<<1, 1>>>();} //
// expected-error@-1 {{reference to __global__ function 'kernel' in __host__ __device__ function}}
void func_4(void) { hostdevice_func(); }
#endif
