// RUN: polygeist-opt -arg-promotion -licm -raise-scf-to-affine -detect-reduction %s | FileCheck %s

// CHECK-LABEL: func.func private @matrix_multiply_reduction
// CHECK-DAG: [[ALLOCA:%.*]] = memref.alloca()
// CHECK-DAG: [[CAST:%.*]] = memref.cast [[ALLOCA]]
// CHECK-DAG: [[ALLOCA1:%.*]] = memref.alloca()
// CHECK-DAG: [[CAST1:%.*]] = memref.cast [[ALLOCA1]]
// CHECK: [[MSC:%.*]] = memref.memory_space_cast [[CAST1]]
// CHECK: scf.if
// CHECK:   sycl.constructor @id([[MSC]], {{.*}}) 
// CHECK:   [[LOAD:%.*]] = affine.load [[ALLOCA1]][0] : memref<1x!sycl_id_1_>
// CHECK:   affine.store [[LOAD]], [[ALLOCA]][0] : memref<1x!sycl_id_1_>
// CHECK:   [[C:%.*]] = sycl.accessor.subscript {{.*}}[[[CAST]]]
// CHECK:   [[INIT:%.*]] = affine.load [[C]][0]
// CHECK:   [[RES:%.*]] = affine.for{{.*}}iter_args([[RED:%.*]] = [[INIT]])
// CHECK-NOT: polygeist.subindex
// CHECK-NOT: memref.memory_space_cast
// CHECK:     [[ADD:%.*]] = arith.addf [[RED]], {{.*}}
// CHECK:     affine.yield [[ADD]]
// CHECK:   affine.store [[RES]], [[C]][0]

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_write_1_ = !sycl.accessor<[1, f32, write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xf32, 1>)>)>
!sycl_accessor_read_1_ = !sycl.accessor<[1, f32, read, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xf32, 1>)>)>

gpu.module @device_func {
  gpu.func @caller() kernel {
    %alloca = memref.alloca() : memref<1x!llvm.struct<(!sycl_accessor_write_1_, !sycl_accessor_read_1_, !sycl_accessor_read_1_)>, 4>
    %cast = memref.cast %alloca : memref<1x!llvm.struct<(!sycl_accessor_write_1_, !sycl_accessor_read_1_, !sycl_accessor_read_1_)>, 4> to memref<?x!llvm.struct<(!sycl_accessor_write_1_, !sycl_accessor_read_1_, !sycl_accessor_read_1_)>, 4>
    %i = arith.constant 0 : i32
    func.call @matrix_multiply_reduction(%cast, %i) : (memref<?x!llvm.struct<(!sycl_accessor_write_1_, !sycl_accessor_read_1_, !sycl_accessor_read_1_)>, 4>, i32) -> ()
    gpu.return
  }

  func.func private @matrix_multiply_reduction(%arg0: memref<?x!llvm.struct<(!sycl_accessor_write_1_, !sycl_accessor_read_1_, !sycl_accessor_read_1_)>, 4>, %arg1: i32) {
    %c2048 = arith.constant 2048 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2048_i32 = arith.constant 2048 : i32
    %c2 = arith.constant 2 : index
    %alloca = memref.alloca() : memref<1x!sycl_id_1_>
    %cast = memref.cast %alloca : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
    %alloca_0 = memref.alloca() : memref<1x!sycl_id_1_>
    %cast_1 = memref.cast %alloca_0 : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
    %alloca_2 = memref.alloca() : memref<1x!sycl_id_1_>
    %cast_3 = memref.cast %alloca_2 : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
    %alloca_4 = memref.alloca() : memref<1x!sycl_id_1_>
    %cast_5 = memref.cast %alloca_4 : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
    %alloca_6 = memref.alloca() : memref<1x!sycl_id_1_>
    %cast_7 = memref.cast %alloca_6 : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
    %alloca_8 = memref.alloca() : memref<1x!sycl_id_1_>
    %cast_9 = memref.cast %alloca_8 : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
    scf.for %arg2 = %c0 to %c2048 step %c1 {
      %4 = arith.index_cast %arg2 : index to i32
      %5 = "polygeist.subindex"(%arg0, %c1) : (memref<?x!llvm.struct<(!sycl_accessor_write_1_, !sycl_accessor_read_1_, !sycl_accessor_read_1_)>, 4>, index) -> memref<?x!sycl_accessor_read_1_, 4>
      %6 = arith.muli %arg1, %c2048_i32 : i32
      %7 = arith.addi %6, %4 : i32
      %8 = arith.extui %7 : i32 to i64
      %memspacecast = memref.memory_space_cast %cast_9 : memref<?x!sycl_id_1_> to memref<?x!sycl_id_1_, 4>
      sycl.constructor @id(%memspacecast, %8) {MangledFunctionName = @id} : (memref<?x!sycl_id_1_, 4>, i64)
      %9 = affine.load %alloca_8[0] : memref<1x!sycl_id_1_>
      affine.store %9, %alloca_6[0] : memref<1x!sycl_id_1_>
      %10 = sycl.accessor.subscript %5[%cast_7] {ArgumentTypes = [memref<?x!sycl_accessor_read_1_, 4>, memref<?x!sycl_id_1_>], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @accessor} : (memref<?x!sycl_accessor_read_1_, 4>, memref<?x!sycl_id_1_>) -> memref<?xf32, 4>
      %11 = affine.load %10[0] : memref<?xf32, 4>
      %12 = "polygeist.subindex"(%arg0, %c2) : (memref<?x!llvm.struct<(!sycl_accessor_write_1_, !sycl_accessor_read_1_, !sycl_accessor_read_1_)>, 4>, index) -> memref<?x!sycl_accessor_read_1_, 4>
      %13 = arith.muli %4, %c2048_i32 : i32
      %14 = arith.addi %13, %arg1 : i32
      %15 = arith.extui %14 : i32 to i64
      %memspacecast_10 = memref.memory_space_cast %cast_5 : memref<?x!sycl_id_1_> to memref<?x!sycl_id_1_, 4>
      sycl.constructor @id(%memspacecast_10, %15) {MangledFunctionName = @id} : (memref<?x!sycl_id_1_, 4>, i64)
      %16 = affine.load %alloca_4[0] : memref<1x!sycl_id_1_>
      affine.store %16, %alloca_2[0] : memref<1x!sycl_id_1_>
      %17 = sycl.accessor.subscript %12[%cast_3] {ArgumentTypes = [memref<?x!sycl_accessor_read_1_, 4>, memref<?x!sycl_id_1_>], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @accessor} : (memref<?x!sycl_accessor_read_1_, 4>, memref<?x!sycl_id_1_>) -> memref<?xf32, 4>
      %18 = affine.load %17[0] : memref<?xf32, 4>
      %19 = arith.mulf %11, %18 : f32
      %20 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!llvm.struct<(!sycl_accessor_write_1_, !sycl_accessor_read_1_, !sycl_accessor_read_1_)>, 4>, index) -> memref<?x!sycl_accessor_write_1_, 4>
      %21 = arith.addi %6, %arg1 : i32
      %22 = arith.extui %21 : i32 to i64
      %memspacecast_11 = memref.memory_space_cast %cast_1 : memref<?x!sycl_id_1_> to memref<?x!sycl_id_1_, 4>
      sycl.constructor @id(%memspacecast_11, %22) {MangledFunctionName = @id} : (memref<?x!sycl_id_1_, 4>, i64)
      %23 = affine.load %alloca_0[0] : memref<1x!sycl_id_1_>
      affine.store %23, %alloca[0] : memref<1x!sycl_id_1_>
      %24 = sycl.accessor.subscript %20[%cast] {ArgumentTypes = [memref<?x!sycl_accessor_write_1_, 4>, memref<?x!sycl_id_1_>], FunctionName = @"operator[]", MangledFunctionName = @"operator[]", TypeName = @accessor} : (memref<?x!sycl_accessor_write_1_, 4>, memref<?x!sycl_id_1_>) -> memref<?xf32, 4>
      %25 = affine.load %24[0] : memref<?xf32, 4>
      %26 = arith.addf %25, %19 : f32
      affine.store %26, %24[0] : memref<?xf32, 4>
    }
    return
  }
}
