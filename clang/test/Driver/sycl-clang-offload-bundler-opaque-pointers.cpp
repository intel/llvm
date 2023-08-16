///
/// Checks that opaque-pointers are being set for the llvm context accordingly to the bundle targets,
/// such that for multiple targets, if the first bc input module uses typed pointers, the bitcode
/// reader will not fail because opaque pointers cannot be converted to typed pointers.
///

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  auto ptr = sycl::malloc_shared<int>(1, q);
  q.single_task([=]() {
	  ptr[0] = 1;
  });
  return 0;
}

// UNSUPPORTED: system-windows

//
// Generate inputs for clang-offload-bundler.
//
// RUN: %clangxx -fsycl -fsycl-targets=spir64 -fsycl-device-only %s -c -emit-llvm -o %s.spir64.bc
// RUN: %clangxx -fsycl -fsycl-targets=nvptx-nvidia-cuda -fsycl-device-only %s -c -emit-llvm -o %s.nvptx.bc
// RUN: %clangxx -fsycl -target x86_64-unknown-linux-gnu %s -c -o %s.host.o

//
// Check clang-offload-bundler obj for opaque pointers support error when spir64 with 
// typed pointers is the first input, followed by an opaque pointers module (nvptx64).
//
// RUN: clang-offload-bundler -type=o -targets=sycl-spir64-unknown-unknown,sycl-nvptx64-nvidia-cuda-sm_50,host-x86_64-unknown-linux-gpu -output=%s.o -input=%s.spir64.bc -input=%s.nvptx.bc -input=%s.host.o 2>&1 \
// RUN: | FileCheck %s -check-prefix=CHECK-STD --allow-empty
// CHECK-STD-NOT: error: Opaque pointers are only supported in -opaque-pointers mode

//
// Check the nvptx module actually contains opaque pointers and no typed pointers.
//
// RUN: llvm-dis %s.nvptx.bc -o %s.nvptx.ll
// RUN: FileCheck %s --check-prefix SYCL_OFFLOAD_BUNDLER_MULTI_TARGET_OPAQUE_POINTERS --input-file=%s.nvptx.ll
// SYCL_OFFLOAD_BUNDLER_MULTI_TARGET_OPAQUE_POINTERS: target triple = "nvptx-nvidia-cuda"
// SYCL_OFFLOAD_BUNDLER_MULTI_TARGET_OPAQUE_POINTERS: ptr addrspace(1)
// SYCL_OFFLOAD_BUNDLER_MULTI_TARGET_OPAQUE_POINTERS-NOT: i32 addrspace(1)*

//
// Check the spir64 module actually contains typed pointers and not opaque pointers.
//
// NOTE This check part should be removed once opaque pointers are supported for SPIR-V target
//
// RUN: llvm-dis %s.spir64.bc -o %s.spir64.ll
// RUN: FileCheck %s --check-prefix SYCL_OFFLOAD_BUNDLER_MULTI_TARGET_NO_OPAQUE_POINTERS --input-file=%s.spir64.ll
// SYCL_OFFLOAD_BUNDLER_MULTI_TARGET_NO_OPAQUE_POINTERS: target triple = "spir64-unknown-unknown"
// SYCL_OFFLOAD_BUNDLER_MULTI_TARGET_NO_OPAQUE_POINTERS: i32 addrspace(1)*
// SYCL_OFFLOAD_BUNDLER_MULTI_TARGET_NO_OPAQUE_POINTERS-NOT: ptr addrspace(1)
