// RUN: sycl-mlir-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s

// Ensure sycl.id and sycl.range types can be arguments of sycl.constructor.
// CHECK-LABEL: func.func @AccessorImplDevice
!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
func.func @AccessorImplDevice(%arg0: memref<?x!sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>>, %arg1: !sycl_id_1_, %arg2: !sycl_range_1_) {
  sycl.constructor(%arg0, %arg1, %arg2, %arg2) {MangledFunctionName = @_ZN4sycl3_V16detail18AccessorImplDeviceILi1EEC1ENS0_2idILi1EEENS0_5rangeILi1EEES7_, TypeName = @AccessorImplDevice} : (memref<?x!sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>>, !sycl_id_1_, !sycl_range_1_, !sycl_range_1_) -> ()
  return
}

// Ensure integer pointer can be arguments of sycl.constructor.
// CHECK-LABEL: func.func @TestConstructorII32Ptr
func.func @TestConstructorII32Ptr(%arg0: memref<?x!sycl_id_1_, 4>, %arg1: memref<?xi32, 1>) {
  sycl.constructor(%arg0, %arg1) {MangledFunctionName = @_ZN4sycl3_V19multi_ptrIjLNS0_6access13address_spaceE1ELNS2_9decoratedE1EEC1EPU3AS1j, TypeName = @multi_ptr} : (memref<?x!sycl_id_1_, 4>, memref<?xi32, 1>) -> ()
  return
}

// CHECK-LABEL: func.func @SubGroupConstructor
func.func @SubGroupConstructor(%arg0: memref<?x!sycl.sub_group, 4>, %arg1: memref<?x!sycl.sub_group, 4>) {
  sycl.constructor(%arg0, %arg1) {MangledFunctionName = @_ZN4sycl3_V13ext6oneapi9sub_groupC1ERKS3_, TypeName = @sub_group} : (memref<?x!sycl.sub_group, 4>, memref<?x!sycl.sub_group, 4>) -> ()
  return
}

// CHECK-LABEL: func.func @MinimumConstructor
func.func @MinimumConstructor(%arg0: memref<?x!sycl.minimum<i32>, 4>, %arg1: memref<?x!sycl.minimum<i32>, 4>) {
  sycl.constructor(%arg0, %arg1) {MangledFunctionName = @_ZN4sycl3_V17minimumIiEC1ERKS2_, TypeName = @minimum} : (memref<?x!sycl.minimum<i32>, 4>, memref<?x!sycl.minimum<i32>, 4>) -> ()
  return
}
