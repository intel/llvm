// RUN: polygeist-opt --convert-polygeist-to-llvm='use-opaque-pointers=1' --split-input-file %s | FileCheck %s

// CHECK-LABEL: @test_1
// CHECK:      [[ZERO:%.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: [[GEP:%.*]] = llvm.getelementptr %{{.*}}[[[ZERO]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::id.1", {{.*}}
// CHECK-NEXT: llvm.return [[GEP]]

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
func.func @test_1(%arg0: memref<?x!llvm.struct<(!sycl_id_1_)>>) -> memref<?x!sycl_id_1_> {
  %c0 = arith.constant 0 : index
  %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!llvm.struct<(!sycl_id_1_)>>, index) -> memref<?x!sycl_id_1_>
  return %0 : memref<?x!sycl_id_1_>
}

// -----

// CHECK-LABEL: @test_2
// CHECK: [[GEP:%.*]] = llvm.getelementptr %{{.*}}[%{{.*}}, {{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::detail::AccessorImplDevice.1", {{.*}}
// CHECK-NEXT: llvm.return [[GEP]]

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_ = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>, !llvm.struct<(ptr<i32, 1>)>)>

func.func @test_2(%arg0: memref<?x!sycl_accessor_1_>) -> memref<?x!sycl_accessor_impl_device_1_> {
  %c0 = arith.constant 0 : index
  %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!sycl_accessor_1_>, index) -> memref<?x!sycl_accessor_impl_device_1_>
  return %0 : memref<?x!sycl_accessor_impl_device_1_>
}

// -----

// CHECK:  llvm.func @test_3([[A0:.*]]: !llvm.ptr) -> !llvm.ptr {
// CHECK: [[IDX_ZERO:%.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK: [[ZERO:%.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: [[GEP:%.*]] = llvm.getelementptr [[A0]][[[ZERO]], [[IDX_ZERO]]] : (!llvm.ptr, i64, i64) -> !llvm.ptr, i32
// CHECK-NEXT: llvm.return [[GEP]] : !llvm.ptr

func.func @test_3(%arg0: memref<?x!llvm.struct<(i32)>>) -> memref<?xi32> {
  %c0 = arith.constant 0 : index
  %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!llvm.struct<(i32)>>, index) -> memref<?xi32>
  return %0 : memref<?xi32>
}

// -----

// CHECK: llvm.func @test_4([[A0:%.*]]: !llvm.ptr, [[A5:%.*]]: i64) -> !llvm.ptr {
// CHECK: [[GEP:%.*]] = llvm.getelementptr [[A0]][[[A5]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(struct<"class.sycl::_V1::id.1", {{.*}})>
// CHECK-NEXT: llvm.return [[GEP]] : !llvm.ptr

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
func.func @test_4(%arg0: memref<1x!llvm.struct<(!sycl_id_1_)>>, %arg1: index) -> memref<?x!llvm.struct<(!sycl_id_1_)>> {
  %0 = "polygeist.subindex"(%arg0, %arg1) : (memref<1x!llvm.struct<(!sycl_id_1_)>>, index) -> memref<?x!llvm.struct<(!sycl_id_1_)>>
  return %0 : memref<?x!llvm.struct<(!sycl_id_1_)>>
}

// -----

// CHECK: llvm.func @test_5([[A0:%.*]]: !llvm.ptr<4>) -> !llvm.ptr<4> {
// CHECK-DAG: [[ZERO1:%.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG: [[ZERO2:%.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: [[GEP:%.*]] = llvm.getelementptr [[A0]][[[ZERO2]], [[ZERO2]], [[ZERO1]]] : (!llvm.ptr<4>, i64, i64, i64) -> !llvm.ptr<4>, i64

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
func.func @test_5(%arg0: memref<?x!sycl.array<[1], (memref<1xi64, 4>)>, 4>) -> memref<1xi64, 4> {
  %c0 = arith.constant 0 : index
  %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!sycl.array<[1], (memref<1xi64, 4>)>, 4>, index) -> memref<1xi64, 4>
  return %0 : memref<1xi64, 4>
}
