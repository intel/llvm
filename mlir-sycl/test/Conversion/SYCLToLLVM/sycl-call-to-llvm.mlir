// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm -verify-diagnostics %s | FileCheck %s

//===-------------------------------------------------------------------------------------------------===//
// sycl.call with non void return type
//===-------------------------------------------------------------------------------------------------===//

// CHECK: llvm.func @foo() -> [[RET_TYPE:i32]]
// CHECK: llvm.func @test() -> [[RET_TYPE]] {
// CHECK-NEXT:  %0 = llvm.call @foo() : () -> [[RET_TYPE]]
// CHECK-NEXT:  llvm.return %0 : [[RET_TYPE]]

func.func private @foo() -> (i32)

func.func @test() -> (i32) {
  %0 = sycl.call() {FunctionName = @foo, MangledFunctionName = @foo, TypeName = @accessor} : () -> i32
  return %0 : i32
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Member functions for sycl::accessor
//===-------------------------------------------------------------------------------------------------===//

!sycl_accessor_1_i32_read_write_global_buffer = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK: llvm.func @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE([[ARG_TYPES:!llvm.ptr<struct<"class.sycl::_V1::accessor.1",.*]])
func.func private @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE(memref<?x!sycl_accessor_1_i32_read_write_global_buffer>, memref<?xi32>, !sycl.range<1>, !sycl.range<1>, !sycl.id<1>)

func.func @accessorInit1(%arg0: memref<?x!sycl_accessor_1_i32_read_write_global_buffer>, %arg1: memref<?xi32>, %arg2: !sycl.range<1>, %arg3: !sycl.range<1>, %arg4: !sycl.id<1>) {
  // CHECK: llvm.call @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE({{.*}}) : ([[ARG_TYPES]]) -> ()
  sycl.call(%arg0, %arg1, %arg2, %arg3, %arg4) {FunctionName = @__init, MangledFunctionName = @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE, TypeName = @accessor} : (memref<?x!sycl_accessor_1_i32_read_write_global_buffer>, memref<?xi32>, !sycl.range<1>, !sycl.range<1>, !sycl.id<1>) -> ()
  return
}

// -----
