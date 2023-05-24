// RUN: mlir-opt -convert-spirv-to-llvm='use-opaque-pointers=1' %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.ControlBarrier
//===----------------------------------------------------------------------===//

// CHECK: llvm.func @_Z22__spirv_ControlBarrierjjj(i32, i32, i32)
// CHECK-LABEL: @control_barrier
spirv.func @control_barrier() "None" {
  // CHECK-NEXT: %0 = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: %1 = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: %2 = llvm.mlir.constant(272 : i32) : i32
  // CHECK-NEXT: llvm.call @_Z22__spirv_ControlBarrierjjj(%0, %1, %2) : (i32, i32, i32) -> ()
  spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
  spirv.Return
}
