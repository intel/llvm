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

// (A)
__host__ void host_fn_0() {}
// expected-note@-1 {{candidate function not viable: call to __host__ function from __device__ function}}
__device__ void dev_fn_0() {
  host_fn_0();
// expected-error@-1 {{no matching function for call to 'host_fn_0'}}
}

// (B)
__host__ void host_fn_1() {}
// expected-note@-1 {{candidate function not viable: call to __host__ function from __global__ function}}
__global__ void kernel() { host_fn_1(); }
// expected-error@-1 {{no matching function for call to 'host_fn_1'}}
void func_0(void) { kernel<<<1, 1>>>(); }

// (C)
__global__ void kernel_1() {}
// expected-note@-1 {{candidate function not viable: call to __global__ function from __global__ function}}
__global__ void kernel_2() { kernel_1(); }
// expected-error@-1 {{no matching function for call to 'kernel_1'}}
void func_2(void) { kernel_2<<<1, 1>>>(); }

// (D)
__global__ void kernel_3() {}
__device__ void device_func() { kernel_3<<<1, 1>>>();}
// expected-error@-1 {{reference to __global__ function 'kernel_3' in __device__ function}}
void func_3(void) { device_func(); }

#if defined(__sycl_cuda_host) || defined(__sycl_device)
// expected-error@* {{reference to __host__ function 'cudaConfigureCall' in __device__ function}}
#endif
#ifdef __sycl_cuda_host
// expected-note@*:* {{'cudaConfigureCall' declared here}}
#endif

// (E)
#if defined(__sycl_device)
__global__ void kernel_4() {}
__host__ __device__ void hostdevice_func() { kernel_4<<<1, 1>>>();} //
// expected-error@-1 {{reference to __global__ function 'kernel_4' in __host__ __device__ function}}
void func_4(void) { hostdevice_func(); }
#endif
