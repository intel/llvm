// RUN: polygeist-opt -split-input-file -test-memory-access %s 2>&1 | FileCheck %s

!sycl_id_1 = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_2 = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_1 = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2 = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_accessor_1_f32_rw_gb = !sycl.accessor<[1, f32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl_id_1, !sycl_range_1, !sycl_range_1)>, !llvm.struct<(memref<?xf32, 1>)>)>
!sycl_accessor_2_f32_rw_gb = !sycl.accessor<[2, f32, read_write, global_buffer], (!sycl.accessor_impl_device<[2], (!sycl_id_2, !sycl_range_2, !sycl_range_2)>, !llvm.struct<(memref<?xf32, 1>)>)>

// COM: Test 1-dim accessor memory access.
//      The underlying value of the accessor subscript is the loop IV.
// CHECK-LABEL: test_tag: test1_load1
// CHECK: matrix:
// CHECK-NEXT: 1
func.func @test1(%acc : memref<?x!sycl_accessor_1_f32_rw_gb, 4>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_1>
  %id = memref.cast %alloca : memref<1x!sycl_id_1> to memref<?x!sycl_id_1>  

  affine.for %ii = 0 to 64 {
    %i = arith.index_cast %ii : index to i64
    sycl.constructor @id(%id, %i) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_1>, i64)
    %subscr = sycl.accessor.subscript %acc[%id] : (memref<?x!sycl_accessor_1_f32_rw_gb, 4>, memref<?x!sycl_id_1>) -> memref<?xf32, 4>
    %load1 = affine.load %subscr[0] {tag = "test1_load1"} : memref<?xf32, 4>
  }
  return
}

// COM: Test 1-dim accessor memory access.
//      The underlying values of the accessor subscript operations are affine expression.
// CHECK-LABEL: test_tag: test1a_load1
// CHECK: matrix:
// CHECK-NEXT: 2
// CHECK-LABEL: test_tag: test1a_load2
// CHECK: matrix:
// CHECK-NEXT: 1
func.func @test1a(%acc : memref<?x!sycl_accessor_1_f32_rw_gb, 4>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_1>
  %alloca_0 = memref.alloca() : memref<1x!sycl_id_1>
  %cast = memref.cast %alloca : memref<1x!sycl_id_1> to memref<?x!sycl_id_1>
  %cast_0 = memref.cast %alloca : memref<1x!sycl_id_1> to memref<?x!sycl_id_1>
  %id = memref.memory_space_cast %cast : memref<?x!sycl_id_1> to  memref<?x!sycl_id_1, 4>
  %c1_i32 = arith.constant 1 : i32  
  %c2_i32 = arith.constant 2 : i32
  %c1_i64 = arith.extsi %c1_i32 : i32 to i64  
  %c2_i64 = arith.extsi %c2_i32 : i32 to i64
  
  affine.for %ii = 0 to 64 {
    %index_cast = arith.index_cast %ii : index to i64
    %mul1 = arith.muli %index_cast, %c2_i64 : i64
    %add1 = arith.addi %mul1, %c1_i64 : i64
    sycl.constructor @id1(%id, %add1) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_1, 4>, i64)
    %val1 = affine.load %alloca[0] : memref<1x!sycl_id_1>
    affine.store %val1, %alloca_0[0] : memref<1x!sycl_id_1>
    %subscr1 = sycl.accessor.subscript %acc[%cast_0] : (memref<?x!sycl_accessor_1_f32_rw_gb, 4>, memref<?x!sycl_id_1>) -> memref<?xf32, 4>
    %load1 = affine.load %subscr1[0] {tag = "test1a_load1"} : memref<?xf32, 4>

    %add2 = arith.addi %index_cast, %c1_i64 : i64
    %mul2 = arith.muli %add2, %c1_i64 : i64
    sycl.constructor @id2(%id, %mul2) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_1, 4>, i64)
    %val2 = affine.load %alloca[0] : memref<1x!sycl_id_1>
    affine.store %val2, %alloca_0[0] : memref<1x!sycl_id_1>
    %subscr2 = sycl.accessor.subscript %acc[%cast_0] : (memref<?x!sycl_accessor_1_f32_rw_gb, 4>, memref<?x!sycl_id_1>) -> memref<?xf32, 4>
    %load2 = affine.load %subscr2[0] {tag = "test1a_load2"} : memref<?xf32, 4>
  }
  return
}

// COM: Test 2-dim accessor memory access yielding an identity matrix.
// CHECK-LABEL: test_tag: test2_store1
// CHECK: matrix: 
// CHECK-NEXT: 1 0
// CHECK-NEXT: 0 1
func.func @test2(%acc : memref<?x!sycl_accessor_2_f32_rw_gb, 4>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_2>
  %id = memref.cast %alloca : memref<1x!sycl_id_2> to memref<?x!sycl_id_2>
  %cst = arith.constant 1.000000e+00 : f32

  affine.for %ii = 0 to 64 {
    %i = arith.index_cast %ii : index to i64
    affine.for %jj = 0 to 64 {
      %j = arith.index_cast %jj : index to i64
      sycl.constructor @id(%id, %i, %j) {MangledFunctionName = @dummy} : (memref<?x!sycl_id_2>, i64, i64)
      %subscr = sycl.accessor.subscript %acc[%id] : (memref<?x!sycl_accessor_2_f32_rw_gb, 4>, memref<?x!sycl_id_2>) -> memref<?xf32, 4>
      affine.store %cst, %subscr[0] {tag = "test2_store1"} : memref<?xf32, 4>
    }
  }
  return
}
