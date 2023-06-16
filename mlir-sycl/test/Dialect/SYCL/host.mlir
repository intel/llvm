// RUN: sycl-mlir-opt %s | sycl-mlir-opt | FileCheck %s
// RUN: sycl-mlir-opt %s --mlir-print-op-generic | sycl-mlir-opt | FileCheck %s

// CHECK-LABEL: test_host_constructor
// CHECK-NEXT:  sycl.host.constructor() : () -> !llvm.ptr
func.func @test_host_constructor() -> !llvm.ptr {
  %0 = sycl.host.constructor() : () -> !llvm.ptr
  return %0 : !llvm.ptr
}

// CHECK-LABEL: test_host_constructor_addrspace
// CHECK-NEXT:  sycl.host.constructor() : () -> !llvm.ptr<4>
func.func @test_host_constructor_addrspace() -> !llvm.ptr<4> {
  %0 = sycl.host.constructor() : () -> !llvm.ptr<4>
  return %0 : !llvm.ptr<4>
}

// CHECK-LABEL: test_host_constructor_args
// CHECK-NEXT:  sycl.host.constructor(%arg0, %arg1, %arg2) : (i32, !llvm.ptr, !llvm.ptr<4>) -> !llvm.ptr
func.func @test_host_constructor_args(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr<4>) -> !llvm.ptr {
  %0 = sycl.host.constructor(%arg0, %arg1, %arg2) : (i32, !llvm.ptr, !llvm.ptr<4>) -> !llvm.ptr
  return %0 : !llvm.ptr
}

// CHECK-LABEL: test_host_constructor_args
// CHECK-NEXT:  sycl.host.constructor(%arg0, %arg1, %arg2) : (i32, !llvm.ptr, !llvm.ptr<4>) -> !llvm.ptr
func.func @test_host_constructor_args_addrspace(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr<4>) -> !llvm.ptr<4> {
  %0 = sycl.host.constructor(%arg0, %arg1, %arg2) : (i32, !llvm.ptr, !llvm.ptr<4>) -> !llvm.ptr<4>
  return %0 : !llvm.ptr<4>
}

