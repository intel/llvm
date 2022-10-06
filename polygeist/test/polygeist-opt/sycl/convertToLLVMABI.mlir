// RUN: polygeist-opt --convert-to-llvm-abi --split-input-file %s | FileCheck %s

// CHECK: gpu.func @kernel([[A0:%.*]]: !llvm.ptr<i32, 1>, %arg1: !sycl_range_1_, %arg2: !sycl_range_1_, %arg3: !sycl_id_1_) 
// CHECK-SAME: kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-NEXT:      [[P2M:%.*]] = "polygeist.pointer2memref"([[A0]]) : (!llvm.ptr<i32, 1>) -> memref<?xi32, 1>
// CHECK-NEXT:      sycl.call([[P2M]]) {Function = @foo, MangledName = @foo} : (memref<?xi32, 1>) -> ()

gpu.module @device_functions {
gpu.func @kernel(%arg0: memref<?xi32, 1>, %arg1: !sycl.range<1>, %arg2: !sycl.range<1>, %arg3: !sycl.id<1>) kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
  sycl.call(%arg0) {Function = @foo, MangledName = @foo} : (memref<?xi32, 1>) -> ()
  gpu.return
}
}
