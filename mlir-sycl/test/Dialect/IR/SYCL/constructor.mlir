// RUN: sycl-mlir-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s

// Ensure sycl.id and sycl.range types can be arguments of sycl.constructor.
// CHECK-LABEL: func.func @AccessorImplDevice
func.func @AccessorImplDevice(%arg0: memref<?x!sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>>, %arg1: !sycl.id<1>, %arg2: !sycl.range<1>) {
  sycl.constructor(%arg0, %arg1, %arg2, %arg2) {MangledFunctionName = @_ZN4sycl3_V16detail18AccessorImplDeviceILi1EEC1ENS0_2idILi1EEENS0_5rangeILi1EEES7_, TypeName = @AccessorImplDevice} : (memref<?x!sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>>, !sycl.id<1>, !sycl.range<1>, !sycl.range<1>) -> ()
  return
}

// Ensure sycl.id and sycl.range types can be arguments of sycl.constructor.
// CHECK-LABEL: func.func @TestConstructorII32Ptr
func.func @TestConstructorII32Ptr(%arg0: memref<?x!sycl.id<1>, 4>, %arg1: memref<?xi32, 1>) {
  sycl.constructor(%arg0, %arg1) {MangledFunctionName = @_ZN4sycl3_V19multi_ptrIjLNS0_6access13address_spaceE1ELNS2_9decoratedE1EEC1EPU3AS1j, Type = @multi_ptr} : (memref<?x!sycl.id<1>, 4>, memref<?xi32, 1>) -> ()
  return
}
