// End-to-End test for testing device image compression when we
// separately compile and link device images.
// This test is copied from Compression/compression_separate_compile.cpp
// and modified to test with the New Offloading Model.

// REQUIRES: zstd, opencl-aot, cpu, linux

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20397

// CPU AOT targets host isa, so we compile everything on the run system instead.
//////////////////////  Compile device images
// RUN: %{run-aux} %clangxx --offload-new-driver -fsycl -fsycl-targets=spir64_x86_64 -fsycl-host-compiler=clang++ -fsycl-host-compiler-options='-std=c++17 -Wno-attributes -Wno-deprecated-declarations -fPIC -DENABLE_KERNEL1' -DENABLE_KERNEL1 -c %s -o %t_kernel1_aot.o
// RUN: %{run-aux} %clangxx --offload-new-driver -fsycl -fsycl-targets=spir64_x86_64 -fsycl-host-compiler=clang++ -fsycl-host-compiler-options='-std=c++17 -Wno-attributes -Wno-deprecated-declarations -fPIC -DENABLE_KERNEL2' -DENABLE_KERNEL2 -c %s -o %t_kernel2_aot.o

//////////////////////   Link device images
// RUN: %{run-aux} %clangxx --offload-new-driver --offload-compress -fsycl -fsycl-link -fsycl-targets=spir64_x86_64 -fPIC %t_kernel1_aot.o %t_kernel2_aot.o -o %t_compressed_image.o -v

// Make sure the clang-linker-wrapper is called with the --compress
// option when using the new offloading model.
// RUN: %{run-aux} %clangxx --offload-new-driver --offload-compress -fsycl -fsycl-link -fsycl-targets=spir64_x86_64 -fPIC %t_kernel1_aot.o %t_kernel2_aot.o -o %t_compressed_image.o -### &> %t_driver_opts.txt
// RUN: %{run-aux} FileCheck -input-file=%t_driver_opts.txt %s --check-prefix=CHECK-DRIVER-OPTS

// CHECK-DRIVER-OPTS: clang-linker-wrapper{{.*}} "--compress"

//////////////////////   Compile the host program
// RUN: %{run-aux} %clangxx --offload-new-driver -fsycl -std=c++17 -Wno-attributes -Wno-deprecated-declarations -fPIC -c %s -o %t_main.o

//////////////////////   Link the host program and compressed device images
// RUN: %{run-aux} %clangxx --offload-new-driver -fsycl %t_main.o %t_kernel1_aot.o %t_kernel2_aot.o %t_compressed_image.o -o %t_compress.out

// RUN: %{run} %t_compress.out

#include <sycl/detail/core.hpp>

using namespace sycl;

// Kernel 1
#ifdef ENABLE_KERNEL1
class test_kernel1;
void run_kernel1(int *a, queue q) {
  q.single_task<test_kernel1>([=]() { *a *= 3; }).wait();
}
#endif

// Kernel 2
#ifdef ENABLE_KERNEL2
class test_kernel2;
void run_kernel2(int *a, queue q) {
  q.single_task<test_kernel2>([=]() { *a += 42; }).wait();
}
#endif

// Main application.
#if not defined(ENABLE_KERNEL1) && not defined(ENABLE_KERNEL2)
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <iostream>

class kernel_init;
void run_kernel1(int *a, queue q);
void run_kernel2(int *a, queue q);
int main() {
  int retCode = 0;
  queue q;

  if (!q.get_device().get_info<info::device::usm_shared_allocations>())
    return 0;

  int *p = malloc_shared<int>(1, q);
  *p = 42;

  run_kernel1(p, q);
  run_kernel2(p, q);
  q.wait();

  retCode = *p != (42 * 3 + 42);

  free(p, q);
  return retCode;
}
#endif
