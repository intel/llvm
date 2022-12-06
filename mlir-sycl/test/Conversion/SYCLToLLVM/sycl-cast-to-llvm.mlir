// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm -verify-diagnostics %s | FileCheck %s

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
func.func @test1(%arg0: memref<?x!sycl_range_1_>) -> memref<?x!sycl_array_1_> {
  // CHECK: llvm.func @test1
  // CHECK: [[SRC:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr<struct<"class.sycl::_V1::range.1"
  // CHECK-DAG: [[SRC_IV0:%.*]] = llvm.insertvalue %arg0, [[SRC]][0]
  // CHECK-DAG: [[SRC_IV1:%.*]] = llvm.insertvalue %arg1, [[SRC_IV0]][1]
  // CHECK-DAG: [[SRC_IV2:%.*]] = llvm.insertvalue %arg2, [[SRC_IV1]][2]
  // CHECK-DAG: [[SRC_IV3:%.*]] = llvm.insertvalue %arg3, [[SRC_IV2]][3, 0]
  // CHECK-DAG: [[SRC_IV4:%.*]] = llvm.insertvalue %arg4, [[SRC_IV3]][4, 0]      

  // CHECK: [[SRC_FIELD0:%.*]] = llvm.extractvalue [[SRC_IV4]][0]
  // CHECK-NEXT: [[BITCAST0:%.*]] = llvm.bitcast [[SRC_FIELD0]] : !llvm.ptr<struct<"class.sycl::_V1::range.1", {{.*}} to !llvm.ptr<struct<"class.sycl::_V1::detail::array.1"
  // CHECK-NEXT: [[SRC_FIELD1:%.*]] = llvm.extractvalue [[SRC_IV4]][1]
  // CHECK-NEXT: [[BITCAST1:%.*]] = llvm.bitcast [[SRC_FIELD1]] : !llvm.ptr<struct<"class.sycl::_V1::range.1", {{.*}} to !llvm.ptr<struct<"class.sycl::_V1::detail::array.1"
  // CHECK-DAG: [[SRC_FIELD2:%.*]] = llvm.extractvalue [[SRC_IV4]][2]      
  // CHECK-DAG: [[SRC_FIELD3:%.*]] = llvm.extractvalue [[SRC_IV4]][3, 0]
  // CHECK-DAG: [[SRC_FIELD4:%.*]] = llvm.extractvalue [[SRC_IV4]][4, 0]

  // CHECK-DAG: [[RES:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr<struct<"class.sycl::_V1::detail::array.1"
  // CHECK-DAG: [[RES_IV0:%.*]] = llvm.insertvalue [[BITCAST0]], [[RES]][0] {{.*}}
  // CHECK-DAG: [[RES_IV1:%.*]] = llvm.insertvalue [[BITCAST1]], {{.*}}[1] {{.*}}  
  // CHECK-DAG: [[RES_IV2:%.*]] = llvm.insertvalue [[SRC_FIELD2]], {{.*}}[2]  
  // CHECK-DAG: [[RES_IV3:%.*]] = llvm.insertvalue [[SRC_FIELD3]], {{.*}}[3, 0]
  // CHECK-DAG: [[RES_IV4:%.*]] = llvm.insertvalue [[SRC_FIELD4]], {{.*}}[4, 0]

  // CHECK: llvm.return [[RES_IV2]] : !llvm.struct<(ptr<struct<"class.sycl::_V1::detail::array.1"  

  %0 = "sycl.cast"(%arg0) : (memref<?x!sycl_range_1_>) -> memref<?x!sycl_array_1_>
  func.return %0 : memref<?x!sycl_array_1_>
}

// -----

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
func.func @test2(%arg0: memref<?x!sycl_id_1_>) -> memref<?x!sycl_array_1_> {
  // CHECK: llvm.func @test2
  // CHECK: [[SRC:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr<struct<"class.sycl::_V1::id.1"
  // CHECK-DAG: [[SRC_IV0:%.*]] = llvm.insertvalue %arg0, [[SRC]][0]
  // CHECK-DAG: [[SRC_IV1:%.*]] = llvm.insertvalue %arg1, [[SRC_IV0]][1]
  // CHECK-DAG: [[SRC_IV2:%.*]] = llvm.insertvalue %arg2, [[SRC_IV1]][2]
  // CHECK-DAG: [[SRC_IV3:%.*]] = llvm.insertvalue %arg3, [[SRC_IV2]][3, 0]
  // CHECK-DAG: [[SRC_IV4:%.*]] = llvm.insertvalue %arg4, [[SRC_IV3]][4, 0]      

  // CHECK: [[SRC_FIELD0:%.*]] = llvm.extractvalue [[SRC_IV4]][0]
  // CHECK-NEXT: [[BITCAST0:%.*]] = llvm.bitcast [[SRC_FIELD0]] : !llvm.ptr<struct<"class.sycl::_V1::id.1", {{.*}} to !llvm.ptr<struct<"class.sycl::_V1::detail::array.1"
  // CHECK-NEXT: [[SRC_FIELD1:%.*]] = llvm.extractvalue [[SRC_IV4]][1]
  // CHECK-NEXT: [[BITCAST1:%.*]] = llvm.bitcast [[SRC_FIELD1]] : !llvm.ptr<struct<"class.sycl::_V1::id.1", {{.*}} to !llvm.ptr<struct<"class.sycl::_V1::detail::array.1"
  // CHECK-DAG: [[SRC_FIELD2:%.*]] = llvm.extractvalue [[SRC_IV4]][2]      
  // CHECK-DAG: [[SRC_FIELD3:%.*]] = llvm.extractvalue [[SRC_IV4]][3, 0]
  // CHECK-DAG: [[SRC_FIELD4:%.*]] = llvm.extractvalue [[SRC_IV4]][4, 0]

  // CHECK-DAG: [[RES:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr<struct<"class.sycl::_V1::detail::array.1"
  // CHECK-DAG: [[RES_IV0:%.*]] = llvm.insertvalue [[BITCAST0]], [[RES]][0] {{.*}}
  // CHECK-DAG: [[RES_IV1:%.*]] = llvm.insertvalue [[BITCAST1]], {{.*}}[1] {{.*}}  
  // CHECK-DAG: [[RES_IV2:%.*]] = llvm.insertvalue [[SRC_FIELD2]], {{.*}}[2]  
  // CHECK-DAG: [[RES_IV3:%.*]] = llvm.insertvalue [[SRC_FIELD3]], {{.*}}[3, 0]
  // CHECK-DAG: [[RES_IV4:%.*]] = llvm.insertvalue [[SRC_FIELD4]], {{.*}}[4, 0]

  // CHECK: llvm.return [[RES_IV2]] : !llvm.struct<(ptr<struct<"class.sycl::_V1::detail::array.1"  

  %0 = "sycl.cast"(%arg0) : (memref<?x!sycl_id_1_>) -> memref<?x!sycl_array_1_>
  func.return %0: memref<?x!sycl_array_1_>
}

// -----

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>, !llvm.struct<(ptr<i32, 1>)>)>
func.func @test3(%arg0: memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?x!sycl.accessor_common> {
  // CHECK-LABEL: llvm.func @test3
  // CHECK: [[SRC:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr<struct<"class.sycl::_V1::accessor.1"
  // CHECK-DAG: [[SRC_IV0:%.*]] = llvm.insertvalue %arg0, [[SRC]][0]
  // CHECK-DAG: [[SRC_IV1:%.*]] = llvm.insertvalue %arg1, [[SRC_IV0]][1]
  // CHECK-DAG: [[SRC_IV2:%.*]] = llvm.insertvalue %arg2, [[SRC_IV1]][2]
  // CHECK-DAG: [[SRC_IV3:%.*]] = llvm.insertvalue %arg3, [[SRC_IV2]][3, 0]
  // CHECK-DAG: [[SRC_IV4:%.*]] = llvm.insertvalue %arg4, [[SRC_IV3]][4, 0]      

  // CHECK: [[SRC_FIELD0:%.*]] = llvm.extractvalue [[SRC_IV4]][0]
  // CHECK-NEXT: [[BITCAST0:%.*]] = llvm.bitcast [[SRC_FIELD0]] : !llvm.ptr<struct<"class.sycl::_V1::accessor.1", {{.*}} to !llvm.ptr<struct<"class.sycl::_V1::detail::accessor_common"
  // CHECK-NEXT: [[SRC_FIELD1:%.*]] = llvm.extractvalue [[SRC_IV4]][1]
  // CHECK-NEXT: [[BITCAST1:%.*]] = llvm.bitcast [[SRC_FIELD1]] : !llvm.ptr<struct<"class.sycl::_V1::accessor.1", {{.*}} to !llvm.ptr<struct<"class.sycl::_V1::detail::accessor_common"
  // CHECK-DAG: [[SRC_FIELD2:%.*]] = llvm.extractvalue [[SRC_IV4]][2]      
  // CHECK-DAG: [[SRC_FIELD3:%.*]] = llvm.extractvalue [[SRC_IV4]][3, 0]
  // CHECK-DAG: [[SRC_FIELD4:%.*]] = llvm.extractvalue [[SRC_IV4]][4, 0]

  // CHECK-DAG: [[RES:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr<struct<"class.sycl::_V1::detail::accessor_common"
  // CHECK-DAG: [[RES_IV0:%.*]] = llvm.insertvalue [[BITCAST0]], [[RES]][0] {{.*}}
  // CHECK-DAG: [[RES_IV1:%.*]] = llvm.insertvalue [[BITCAST1]], {{.*}}[1] {{.*}}  
  // CHECK-DAG: [[RES_IV2:%.*]] = llvm.insertvalue [[SRC_FIELD2]], {{.*}}[2]  
  // CHECK-DAG: [[RES_IV3:%.*]] = llvm.insertvalue [[SRC_FIELD3]], {{.*}}[3, 0]
  // CHECK-DAG: [[RES_IV4:%.*]] = llvm.insertvalue [[SRC_FIELD4]], {{.*}}[4, 0]

  // CHECK: llvm.return [[RES_IV2]] : !llvm.struct<(ptr<struct<"class.sycl::_V1::detail::accessor_common"  

  %0 = "sycl.cast"(%arg0) : (memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?x!sycl.accessor_common>
  func.return %0: memref<?x!sycl.accessor_common>
}

func.func @test4(%arg0: memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?x!sycl.owner_less_base> {
  // CHECK-LABEL: llvm.func @test4
  // CHECK: [[SRC:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr<struct<"class.sycl::_V1::accessor.1"
  // CHECK-DAG: [[SRC_IV0:%.*]] = llvm.insertvalue %arg0, [[SRC]][0]
  // CHECK-DAG: [[SRC_IV1:%.*]] = llvm.insertvalue %arg1, [[SRC_IV0]][1]
  // CHECK-DAG: [[SRC_IV2:%.*]] = llvm.insertvalue %arg2, [[SRC_IV1]][2]
  // CHECK-DAG: [[SRC_IV3:%.*]] = llvm.insertvalue %arg3, [[SRC_IV2]][3, 0]
  // CHECK-DAG: [[SRC_IV4:%.*]] = llvm.insertvalue %arg4, [[SRC_IV3]][4, 0]      

  // CHECK: [[SRC_FIELD0:%.*]] = llvm.extractvalue [[SRC_IV4]][0]
  // CHECK-NEXT: [[BITCAST0:%.*]] = llvm.bitcast [[SRC_FIELD0]] : !llvm.ptr<struct<"class.sycl::_V1::accessor.1", {{.*}} to !llvm.ptr<struct<"class.sycl::_V1::detail::OwnerLessBase"
  // CHECK-NEXT: [[SRC_FIELD1:%.*]] = llvm.extractvalue [[SRC_IV4]][1]
  // CHECK-NEXT: [[BITCAST1:%.*]] = llvm.bitcast [[SRC_FIELD1]] : !llvm.ptr<struct<"class.sycl::_V1::accessor.1", {{.*}} to !llvm.ptr<struct<"class.sycl::_V1::detail::OwnerLessBase"
  // CHECK-DAG: [[SRC_FIELD2:%.*]] = llvm.extractvalue [[SRC_IV4]][2]      
  // CHECK-DAG: [[SRC_FIELD3:%.*]] = llvm.extractvalue [[SRC_IV4]][3, 0]
  // CHECK-DAG: [[SRC_FIELD4:%.*]] = llvm.extractvalue [[SRC_IV4]][4, 0]

  // CHECK-DAG: [[RES:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr<struct<"class.sycl::_V1::detail::OwnerLessBase"
  // CHECK-DAG: [[RES_IV0:%.*]] = llvm.insertvalue [[BITCAST0]], [[RES]][0] {{.*}}
  // CHECK-DAG: [[RES_IV1:%.*]] = llvm.insertvalue [[BITCAST1]], {{.*}}[1] {{.*}}  
  // CHECK-DAG: [[RES_IV2:%.*]] = llvm.insertvalue [[SRC_FIELD2]], {{.*}}[2]  
  // CHECK-DAG: [[RES_IV3:%.*]] = llvm.insertvalue [[SRC_FIELD3]], {{.*}}[3, 0]
  // CHECK-DAG: [[RES_IV4:%.*]] = llvm.insertvalue [[SRC_FIELD4]], {{.*}}[4, 0]

  // CHECK: llvm.return [[RES_IV2]] : !llvm.struct<(ptr<struct<"class.sycl::_V1::detail::OwnerLessBase"  

  %0 = "sycl.cast"(%arg0) : (memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?x!sycl.owner_less_base>
  func.return %0: memref<?x!sycl.owner_less_base>
}
