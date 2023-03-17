// RUN: polygeist-opt --arg-promotion --split-input-file %s | FileCheck %s

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
!sycl_array_2_ = !sycl.array<[2], (memref<2xi64, 4>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_id_2_ = !sycl.id<[2], (!sycl_array_2_)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_range_2_ = !sycl.range<[2], (!sycl_array_2_)>
!sycl_group_2_ = !sycl.group<[2], (!sycl_range_2_, !sycl_range_2_, !sycl_range_2_, !sycl_id_2_)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_item_base_2_ = !sycl.item_base<[2, true], (!sycl_range_2_, !sycl_id_2_, !sycl_id_2_)>
!sycl_item_base_2_1 = !sycl.item_base<[2, false], (!sycl_range_2_, !sycl_id_2_)>
!sycl_accessor_1_f32_w_gb = !sycl.accessor<[1, f32, write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xf32, 1>)>)>
!sycl_item_2_ = !sycl.item<[2, true], (!sycl_item_base_2_)>
!sycl_item_2_1 = !sycl.item<[2, false], (!sycl_item_base_2_1)>
!sycl_nd_item_2_ = !sycl.nd_item<[2], (!sycl_item_2_, !sycl_item_2_1, !sycl_group_2_)>

gpu.module @device_func {
  gpu.func @test1() kernel {
    // CHECK-LABEL: gpu.func @test1() kernel
    // CHECK:         %memspacecast = memref.memory_space_cast %cast_1 : memref<?x!llvm.struct<(i32, !sycl_accessor_1_f32_w_gb)>> to memref<?x!llvm.struct<(i32, !sycl_accessor_1_f32_w_gb)>, 4>
    // CHECK-NEXT:    %c0 = arith.constant 0 : index
    // CHECK-NEXT:    %0 = "polygeist.subindex"(%memspacecast, %c0) : (memref<?x!llvm.struct<(i32, !sycl_accessor_1_f32_w_gb)>, 4>, index) -> memref<?xi32, 4>
    // CHECK-NEXT:    %c1 = arith.constant 1 : index
    // CHECK-NEXT:    %1 = "polygeist.subindex"(%memspacecast, %c1) : (memref<?x!llvm.struct<(i32, !sycl_accessor_1_f32_w_gb)>, 4>, index) -> memref<?x!sycl_accessor_1_f32_w_gb, 4>
    // CHECK-NEXT:    func.call @callee(%0, %1, %cast) : (memref<?xi32, 4>, memref<?x!sycl_accessor_1_f32_w_gb, 4>, memref<?x!sycl_nd_item_2_>) -> ()
    // CHECK-NEXT:    gpu.return

    %alloca = memref.alloca() : memref<1x!sycl_nd_item_2_>
    %cast = memref.cast %alloca : memref<1x!sycl_nd_item_2_> to memref<?x!sycl_nd_item_2_>
    %alloca_1 = memref.alloca() : memref<1x!llvm.struct<(i32, !sycl_accessor_1_f32_w_gb)>>
    %cast_1 = memref.cast %alloca_1 : memref<1x!llvm.struct<(i32, !sycl_accessor_1_f32_w_gb)>> to memref<?x!llvm.struct<(i32, !sycl_accessor_1_f32_w_gb)>>
    %memspacecast = memref.memory_space_cast %cast_1 : memref<?x!llvm.struct<(i32, !sycl_accessor_1_f32_w_gb)>> to memref<?x!llvm.struct<(i32, !sycl_accessor_1_f32_w_gb)>, 4>
    func.call @callee(%memspacecast, %cast) : (memref<?x!llvm.struct<(i32, !sycl_accessor_1_f32_w_gb)>, 4>, memref<?x!sycl_nd_item_2_>) -> ()
    gpu.return
  }

  func.func private @callee(%arg0: memref<?x!llvm.struct<(i32, !sycl_accessor_1_f32_w_gb)>, 4>, %arg1: memref<?x!sycl_nd_item_2_>) {
    // CHECK-LABEL: func.func private @callee
    // CHECK-SAME:    (%arg0: memref<?xi32, 4>, %arg1: memref<?x!sycl_accessor_1_f32_w_gb, 4>, %arg2: memref<?x!sycl_nd_item_2_>) {
    // CHECK-NOT:     {{.*}} = "polygeist.subindex"
    // CHECK:         {{.*}} = affine.load %arg0[0] : memref<?xi32, 4>
    // CHECK:         {{.*}} = sycl.accessor.subscript %arg1[{{.*}}]{{.*}} : (memref<?x!sycl_accessor_1_f32_w_gb, 4>, memref<?x!sycl_id_1_>) -> memref<?xf32, 4>

    %cst = arith.constant 1.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<1x!sycl_id_1_>
    %cast = memref.cast %alloca : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
    %alloca_0 = memref.alloca() : memref<1x!sycl_id_1_>
    %cast_1 = memref.cast %alloca_0 : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
    %0 = "polygeist.subindex"(%arg0, %c1) : (memref<?x!llvm.struct<(i32, !sycl_accessor_1_f32_w_gb)>, 4>, index) -> memref<?x!sycl_accessor_1_f32_w_gb, 4>
    %1 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!llvm.struct<(i32, !sycl_accessor_1_f32_w_gb)>, 4>, index) -> memref<?xi32, 4>
    %2 = affine.load %1[0] : memref<?xi32, 4>
    %3 = arith.extsi %2 : i32 to i64
    %memspacecast = memref.memory_space_cast %cast_1 : memref<?x!sycl_id_1_> to memref<?x!sycl_id_1_, 4>
    sycl.constructor @id(%memspacecast, %3) {MangledFunctionName = @_ZN4sycl3_V12idILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE} : (memref<?x!sycl_id_1_, 4>, i64)
    %4 = affine.load %alloca_0[0] : memref<1x!sycl_id_1_>
    affine.store %4, %alloca[0] : memref<1x!sycl_id_1_>
    %5 = sycl.accessor.subscript %0[%cast] {ArgumentTypes = [memref<?x!sycl_accessor_1_f32_w_gb, 4>, memref<?x!sycl_id_1_>], FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V18accessorIfLi1ELNS0_6access4modeE1025ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERfNS0_2idILi1EEE, TypeName = @accessor} : (memref<?x!sycl_accessor_1_f32_w_gb, 4>, memref<?x!sycl_id_1_>) -> memref<?xf32, 4>
    affine.store %cst, %5[0] : memref<?xf32, 4>
    return
  }  
}

