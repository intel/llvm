// RUN: polygeist-opt --convert-to-llvm-abi --split-input-file %s | FileCheck %s

// CHECK: gpu.func @kernel([[A0:%.*]]: !llvm.ptr<i32, 1>, %arg1: !sycl_range_1_, %arg2: !sycl_range_1_, %arg3: !sycl_id_1_) 
// CHECK-SAME: kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
// CHECK-NEXT:      [[P2M:%.*]] = "polygeist.pointer2memref"([[A0]]) : (!llvm.ptr<i32, 1>) -> memref<?xi32, 1>
// CHECK-NEXT:      [[M2P:%.*]] = "polygeist.memref2pointer"([[P2M]]) : (memref<?xi32, 1>) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:      sycl.call([[M2P]]) {FunctionName = @foo, MangledFunctionName = @foo} : (!llvm.ptr<i32, 1>) -> ()

gpu.module @module {
gpu.func @kernel(%arg0: memref<?xi32, 1>, %arg1: !sycl.range<1>, %arg2: !sycl.range<1>, %arg3: !sycl.id<1>) kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]} {
  sycl.call(%arg0) {FunctionName = @foo, MangledFunctionName = @foo} : (memref<?xi32, 1>) -> ()
  gpu.return
}
}

// -----

// CHECK-LABEL: gpu.func @return() -> (!llvm.ptr<i32, 1>, i64) {
// CHECK-NEXT:    [[MEMREF:%.*]] = sycl.call() {FunctionName = @foo, MangledFunctionName = @foo} : () -> !llvm.ptr<i32, 1>
// CHECK-NEXT:    [[P2M:%.*]] = "polygeist.pointer2memref"([[MEMREF]]) : (!llvm.ptr<i32, 1>) -> memref<?xi32, 1>
// CHECK:   [[M2P:%.*]] = "polygeist.memref2pointer"([[P2M]]) : (memref<?xi32, 1>) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:   gpu.return [[M2P]], {{.*}} : !llvm.ptr<i32, 1>, i64

gpu.module @module {
gpu.func @return() -> (memref<?xi32, 1>, i64) { 
  %memref = sycl.call() {FunctionName = @foo, MangledFunctionName = @foo} : () -> memref<?xi32, 1>
  %c0 = arith.constant 0 : i64
  gpu.return %memref, %c0: memref<?xi32, 1>, i64
}
}

// -----

// CHECK:  func.func @caller([[A0:.*]]: !llvm.ptr<i32, 1>) -> !llvm.ptr<i32, 1> {
// CHECK-NEXT:    [[P2M:%.*]] = "polygeist.pointer2memref"([[A0]]) : (!llvm.ptr<i32, 1>) -> memref<?xi32, 1>
// CHECK-NEXT:    [[M2P:%.*]] = "polygeist.memref2pointer"([[P2M]]) : (memref<?xi32, 1>) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:    [[SYCLCALL:%.*]] = sycl.call([[M2P]]) {FunctionName = @callee, MangledFunctionName = @callee} : (!llvm.ptr<i32, 1>) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:    [[P2M:%.*]] = "polygeist.pointer2memref"([[SYCLCALL]]) : (!llvm.ptr<i32, 1>) -> memref<?xi32, 1>
// CHECK-NEXT:    [[M2P:%.*]] = "polygeist.memref2pointer"([[P2M]]) : (memref<?xi32, 1>) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:    [[FUNCCALL:%.*]] = call @callee([[M2P]]) : (!llvm.ptr<i32, 1>) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:    [[P2M:%.*]] = "polygeist.pointer2memref"([[FUNCCALL]]) : (!llvm.ptr<i32, 1>) -> memref<?xi32, 1>
// CHECK-NEXT:    [[M2P:%.*]] = "polygeist.memref2pointer"([[P2M]]) : (memref<?xi32, 1>) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:    return [[M2P]] : !llvm.ptr<i32, 1>

func.func @caller(%arg0: memref<?xi32, 1>) -> memref<?xi32, 1> { 
  %syclcall = sycl.call(%arg0) {FunctionName = @callee, MangledFunctionName = @callee} : (memref<?xi32, 1>) -> memref<?xi32, 1>
  %funccall = func.call @callee(%syclcall) : (memref<?xi32, 1>) -> memref<?xi32, 1>
  func.return %funccall: memref<?xi32, 1>
}

// CHECK: func.func private @callee(!llvm.ptr<i32, 1>) -> !llvm.ptr<i32, 1>
func.func private @callee(%arg0: memref<?xi32, 1>) -> memref<?xi32, 1>

// -----

// CHECK: !sycl_id_1_ = !sycl.id<1>
// CHECK: func.func @constructor_caller([[A0:%.*]]: !llvm.ptr<!sycl_id_1_, 1>) {
// CHECK-NEXT:    [[P2M:%.*]] = "polygeist.pointer2memref"([[A0]]) : (!llvm.ptr<!sycl_id_1_, 1>) -> memref<?x!sycl_id_1_, 1>
// CHECK-NEXT:    [[M2P:%.*]] = "polygeist.memref2pointer"([[P2M]]) : (memref<?x!sycl_id_1_, 1>) -> !llvm.ptr<!sycl_id_1_, 1>
// CHECK-NEXT:    call @constructor([[M2P]]) : (!llvm.ptr<!sycl_id_1_, 1>) -> ()

func.func @constructor_caller(%arg0: memref<?x!sycl.id<1>, 1>) {
  sycl.constructor(%arg0) {MangledFunctionName = @constructor, Type = @foo} : (memref<?x!sycl.id<1>, 1>) -> ()
  func.return
}

// CHECK: func.func private @constructor(!llvm.ptr<!sycl_id_1_, 1>)
func.func private @constructor(%arg0: memref<?x!sycl.id<1>, 1>)
