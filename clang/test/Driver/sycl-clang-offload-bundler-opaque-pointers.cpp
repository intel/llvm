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
// is the first input, followed by other opaque pointers modules, i.e., here nvptx64.
//
// RUN: clang-offload-bundler -type=o -targets=sycl-spir64-unknown-unknown,sycl-nvptx64-nvidia-cuda-sm_50,host-x86_64-unknown-linux-gpu -output=%s.o -input=%s.spir64.bc -input=%s.nvptx.bc -input=%s.host.o 2>&1 \
// RUN: | FileCheck %s -check-prefix=CHECK-STD --allow-empty
// CHECK-STD-NOT: error: Opaque pointers are only supported in -opaque-pointers mode

//
// Check the nvptx module actually contains opaque pointers and no typed pointers.
//
// RUN: llvm-dis %s.nvptx.bc -o %s.nvptx.ll
// RUN: FileCheck %s --check-prefix SYCL_OFFLOAD_BUNDLER_MULTI_TARGET_OPAQUE_POINTERS_NVPTX --input-file=%s.nvptx.ll
// SYCL_OFFLOAD_BUNDLER_MULTI_TARGET_OPAQUE_POINTERS_NVPTX: target triple = "nvptx-nvidia-cuda"
// SYCL_OFFLOAD_BUNDLER_MULTI_TARGET_OPAQUE_POINTERS_NVPTX: ptr addrspace(1)
// SYCL_OFFLOAD_BUNDLER_MULTI_TARGET_OPAQUE_POINTERS_NVPTX-NOT: i32 addrspace(1)*

//
// Check the spir64 module also contains opaque pointers.
//
// SPIR targets now have opaque pointer support, hence we just double check it.
// They didn't before, so the check was the other way around verifying typed ptrs.
//
// RUN: llvm-dis %s.spir64.bc -o %s.spir64.ll
// RUN: FileCheck %s --check-prefix SYCL_OFFLOAD_BUNDLER_MULTI_TARGET_OPAQUE_POINTERS_SPIR --input-file=%s.spir64.ll
// SYCL_OFFLOAD_BUNDLER_MULTI_TARGET_OPAQUE_POINTERS_SPIR: target triple = "spir64-unknown-unknown"
// SYCL_OFFLOAD_BUNDLER_MULTI_TARGET_OPAQUE_POINTERS_SPIR-NOT: i32 addrspace(1)*
// SYCL_OFFLOAD_BUNDLER_MULTI_TARGET_OPAQUE_POINTERS_SPIR: ptr addrspace(1)
