// RUN: polygeist-opt --convert-to-llvm-abi --split-input-file %s | FileCheck %s

// CHECK: gpu.func @kernel([[A0:%.*]]: !llvm.ptr<i32, 1>, %arg1: !sycl_range_1_, %arg2: !sycl_range_1_, %arg3: !sycl_id_1_) 
// CHECK-SAME: kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-NEXT:      [[P2M:%.*]] = "polygeist.pointer2memref"([[A0]]) : (!llvm.ptr<i32, 1>) -> memref<?xi32, 1>
// CHECK-NEXT:      sycl.call([[P2M]]) {Function = @foo, MangledName = @foo} : (memref<?xi32, 1>) -> ()

gpu.module @module {
gpu.func @kernel(%arg0: memref<?xi32, 1>, %arg1: !sycl.range<1>, %arg2: !sycl.range<1>, %arg3: !sycl.id<1>) kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
  sycl.call(%arg0) {Function = @foo, MangledName = @foo} : (memref<?xi32, 1>) -> ()
  gpu.return
}
}

// -----

// CHECK-LABEL: gpu.func @return() -> (!llvm.ptr<i32, 1>, i64) {
// CHECK:   [[MEMREF:%.*]] = sycl.call() {Function = @foo, MangledName = @foo} : () -> memref<?xi32, 1>
// CHECK:   [[M2P:%.*]] = "polygeist.memref2pointer"([[MEMREF]]) : (memref<?xi32, 1>) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:   gpu.return [[M2P]], {{.*}} : !llvm.ptr<i32, 1>, i64

gpu.module @module {
gpu.func @return() -> (memref<?xi32, 1>, i64) { 
  %memref = sycl.call() {Function = @foo, MangledName = @foo} : () -> memref<?xi32, 1>
  %c0 = arith.constant 0 : i64
  gpu.return %memref, %c0: memref<?xi32, 1>, i64
}
}
