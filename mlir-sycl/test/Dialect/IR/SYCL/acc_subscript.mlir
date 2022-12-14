// RUN: sycl-mlir-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s

// Ensure atomic access mode is OK when we have atomic return type for sycl.accessor.subscript.
// CHECK-LABEL: func.func @AccessorSubscript
!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_accessor_1_i32_ato_gb = !sycl.accessor<[1, i32, atomic, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>, !llvm.struct<(ptr<i32, 1>)>)>
!sycl_atomic_i32_1_ = !sycl.atomic<[i32,1], (memref<?xi32, 1>)>
func.func @AccessorSubscript(%arg0: memref<?x!sycl_accessor_1_i32_ato_gb, 4>, %arg1: i64) {
  %0 = "sycl.accessor.subscript"(%arg0, %arg1) {BaseType = memref<?x!sycl_accessor_1_i32_ato_gb, 4>, FunctionName = @"operator[]", MangledFunctionName = @_ZNK4sycl3_V18accessorIjLi1ELNS0_6access4modeE1029ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EEENSt9enable_ifIXaaeqT_Li1EeqcvS3_Li1029ELS3_1029EENS0_6atomicIjLNS2_13address_spaceE1EEEE4typeEm, TypeName = @accessor} : (memref<?x!sycl_accessor_1_i32_ato_gb, 4>, i64) -> !sycl_atomic_i32_1_
  return
}
