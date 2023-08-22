///
/// Checks that opaque-pointers are being set for llvm context used accordingly
/// to the bundle targets and clang-offload-bundler is able to read the bitcode.
///

// UNSUPPORTED: system-windows

#include "sycl/detail/defines_elementary.hpp"
SYCL_EXTERNAL void foo(int *p) {
    *p = 1;
}

//
// Generate inputs for clang-offload-bundler.
//
// 
// RUN: %clangxx -fsycl -fsycl-targets=spir64 -fsycl-device-only -Xclang -no-opaque-pointers %s -S -emit-llvm -o %s.spv.ll
// RUN: %clangxx -fsycl -fsycl-targets=nvptx-nvidia-cuda -fsycl-device-only %s -S -emit-llvm -o %s.nv.ll
// RUN: %clangxx -fsycl -target x86_64-unknown-linux-gnu %s -c -o %s.host.o

//
// Check clang-offload-bundler obj for opaque pointers support error when spir64
// is the first bc input, followed by an opaque pointers module bc, i.e. nvptx.
//
// RUN: llvm-as %s.spv.ll -o %s.spv.bc | llvm-as %s.nv.ll -o %s.nv.bc \
// RUN: | clang-offload-bundler -type=o -targets=sycl-spir64-unknown-unknown,sycl-nvptx64-nvidia-cuda-sm_50,host-x86_64-unknown-linux-gpu -output=%s.o -input=%s.spv.bc -input=%s.nv.bc -input=%s.host.o 2>&1 \
// RUN: | FileCheck %s -check-prefix=CHECK-STD --allow-empty
// CHECK-STD-NOT: error: Opaque pointers are only supported in -opaque-pointers mode

//
// Verify the spir64 module contains typed pointers (set -no-opaque-pointers).
//
// RUN: FileCheck %s --check-prefix CHECK_OPAQUE_POINTERS_SPIRV --input-file=%s.spv.ll
// CHECK_OPAQUE_POINTERS_SPIRV: target triple = "spir64-unknown-unknown"
// CHECK_OPAQUE_POINTERS_SPIRV: i32 addrspace(4)*
// CHECK_OPAQUE_POINTERS_SPIRV-NOT: ptr addrspace(2)

//
// Verify the nvptx module contains opaque pointers.
//
// RUN: FileCheck %s --check-prefix CHECK_OPAQUE_POINTERS_NVPTX --input-file=%s.nv.ll
// CHECK_OPAQUE_POINTERS_NVPTX: target triple = "nvptx-nvidia-cuda"
// CHECK_OPAQUE_POINTERS_NVPTX: ptr addrspace(2)
// CHECK_OPAQUE_POINTERS_NVPTX-NOT: i32 addrspace(4)*
