// RUN: polygeist-opt -allow-unregistered-dialect %s -split-input-file -mlir-print-local-scope \
// RUN: | FileCheck %s

// Verify the generic printed output can be parsed.
// RUN: polygeist-opt -allow-unregistered-dialect %s -split-input-file -mlir-print-local-scope \
// RUN: | polygeist-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @llvm_array_memref(%arg0: memref<!llvm.array<16 x i8>>)
func.func @llvm_array_memref(%arg0: memref<!llvm.array<16 x i8>>) {
  return
}
