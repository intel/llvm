///
/// Checks that opaque-pointers are being set for llvm context used accordingly
/// to the bundle targets and clang-offload-bundler is able to read the bitcode.
///

// REQUIRES: x86-registered-target
// UNSUPPORTED: system-windows

void foo(int *p) {
  *p = 1;
}

//
// Generate inputs for clang-offload-bundler.
//
// RUN: %clangxx -target openmp-x86_64-pc-linux-gnu -Xclang -no-opaque-pointers %s -S -emit-llvm -o %s.tgt1.ll
// RUN: %clangxx -target openmp-x86_64-pc-linux-gnu -Xclang -no-opaque-pointers %s -c -emit-llvm -o %s.tgt1.bc
// RUN: %clangxx -target openmp-nvptx64-nvidia-cuda %s -S -emit-llvm -o %s.tgt2.ll
// RUN: %clangxx -target openmp-nvptx64-nvidia-cuda %s -c -emit-llvm -o %s.tgt2.bc
// RUN: %clangxx -target x86_64-unknown-linux-gnu %s -c -o %s.host.o

//
// Check clang-offload-bundler obj for opaque pointers support error when typed
// pointer module is the first input target, followed by an opaque pointers one.
//
// RUN: clang-offload-bundler -type=o -targets=openmp-x86_64-pc-linux-gnu,openmp-nvptx64-nvidia-cuda,host-x86_64-unknown-linux-gnu -output=%s.bundle.o -input=%s.tgt1.bc -input=%s.tgt2.bc -input=%s.host.o 2>&1 \
// RUN: | FileCheck %s -check-prefix=CHECK-STD --allow-empty
// CHECK-STD-NOT: error: Opaque pointers are only supported in -opaque-pointers mode

//
// Verify the target 1 module contains typed pointers (set -no-opaque-pointers).
//
// RUN: FileCheck %s --check-prefix CHECK-TYPED_POINTERS_TARGET --input-file=%s.tgt1.ll
// CHECK-TYPED_POINTERS_TARGET: target triple = "x86_64-pc-linux-gnu-openmp"
// CHECK-TYPED_POINTERS_TARGET: i32* noundef
// CHECK-TYPED_POINTERS_TARGET-NOT: ptr noundef

//
// Verify the target 2 module contains opaque pointers.
//
// RUN: FileCheck %s --check-prefix CHECK-OPAQUE_POINTERS_TARGET --input-file=%s.tgt2.ll
// CHECK-OPAQUE_POINTERS_TARGET: target triple = "nvptx64-nvidia-cuda-openmp"
// CHECK-OPAQUE_POINTERS_TARGET: ptr noundef
// CHECK-OPAQUE_POINTERS_TARGET-NOT: i32* noundef
