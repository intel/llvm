// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm -verify-diagnostics %s | FileCheck %s

!sycl_array1 = !sycl.array<[1], (memref<1xi64>)>

func.func @test1(%arg0: memref<?x!sycl.range<1>>) -> memref<?x!sycl_array1> {
  // CHECK: llvm.func @test1
  // CHECK: [[SRC:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr<struct<"class.cl::sycl::range.1"
  // CHECK-NEXT: [[SRC_IV0:%.*]] = llvm.insertvalue %arg0, [[SRC]][0]
  // CHECK-NEXT: [[SRC_IV1:%.*]] = llvm.insertvalue %arg1, [[SRC_IV0]][1]
  // CHECK-NEXT: [[SRC_IV2:%.*]] = llvm.insertvalue %arg2, [[SRC_IV1]][2]
  // CHECK-NEXT: [[SRC_IV3:%.*]] = llvm.insertvalue %arg3, [[SRC_IV2]][3, 0]
  // CHECK-NEXT: [[SRC_IV4:%.*]] = llvm.insertvalue %arg4, [[SRC_IV3]][4, 0]      
  // CHECK: [[RES:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr<struct<"class.cl::sycl::detail::array.1"

  // CHECK: [[SRC_FIELD0:%.*]] = llvm.extractvalue [[SRC_IV4]][0]
  // CHECK-NEXT: [[BITCAST0:%.*]] = llvm.bitcast [[SRC_FIELD0]] : !llvm.ptr<struct<"class.cl::sycl::range.1", {{.*}} to !llvm.ptr<struct<"class.cl::sycl::detail::array.1"
  // CHECK-NEXT: [[SRC_FIELD1:%.*]] = llvm.extractvalue [[SRC_IV4]][1]
  // CHECK-NEXT: [[BITCAST1:%.*]] = llvm.bitcast [[SRC_FIELD1]] : !llvm.ptr<struct<"class.cl::sycl::range.1", {{.*}} to !llvm.ptr<struct<"class.cl::sycl::detail::array.1"
  // CHECK-NEXT: [[RES_IV0:%.*]] = llvm.insertvalue [[BITCAST0]], [[RES]][0] {{.*}}
  // CHECK-NEXT: [[RES_IV1:%.*]] = llvm.insertvalue [[BITCAST1]], [[RES_IV0]][1] {{.*}}  
  // CHECK-NEXT: [[SRC_FIELD2:%.*]] = llvm.extractvalue [[SRC_IV4]][2]
  // CHECK-NEXT: [[RES_IV2:%.*]] = llvm.insertvalue [[SRC_FIELD2]], [[RES_IV1]][2]
  // CHECK-NEXT: [[SRC_FIELD3:%.*]] = llvm.extractvalue [[SRC_IV4]][3, 0] 
  // CHECK-NEXT: [[RES_IV3:%.*]] = llvm.insertvalue [[SRC_FIELD3]], [[RES_IV2]][3, 0]
  // CHECK-NEXT: [[SRC_FIELD4:%.*]] = llvm.extractvalue [[SRC_IV4]][4, 0] 
  // CHECK-NEXT: [[RES_IV4:%.*]] = llvm.insertvalue [[SRC_FIELD4]], [[RES_IV3]][4, 0]
  // CHECK-NEXT: llvm.return [[RES_IV4]] : !llvm.struct<(ptr<struct<"class.cl::sycl::detail::array.1"

  %0 = "sycl.cast"(%arg0) : (memref<?x!sycl.range<1>>) -> memref<?x!sycl_array1>
  func.return %0 : memref<?x!sycl_array1>
}
