// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm -verify-diagnostics %s | FileCheck %s

//===-------------------------------------------------------------------------------------------------===//
// sycl.call with non void return type
//===-------------------------------------------------------------------------------------------------===//

// CHECK: llvm.func @foo() -> [[MEMREF:!llvm.struct<\(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>\)>]]
// CHECK: llvm.func @test() -> [[MEMREF]] {
// CHECK-NEXT:  %0 = llvm.call @foo() : () -> [[MEMREF]]
// CHECK-NEXT:  llvm.return %0 : [[MEMREF]]
func.func @test() -> (memref<?xi32>) {
  %0 = sycl.call() {Function = @foo, MangledName = @foo, Type = @accessor} : () -> memref<?xi32>
  return %0 : memref<?xi32>
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Member functions for sycl::accessor
//===-------------------------------------------------------------------------------------------------===//

!sycl_accessor_1_i32_read_write_global_buffer = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK: llvm.func @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE([[ARG_TYPES:!llvm.struct<\(ptr<struct<"class.cl::sycl::accessor.1",.*]])
func.func @accessorInit1(%arg0: memref<?x!sycl_accessor_1_i32_read_write_global_buffer>, %arg1: memref<?xi32>, %arg2: !sycl.range<1>, %arg3: !sycl.range<1>, %arg4: !sycl.id<1>) {
  // CHECK: llvm.call @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE({{.*}}) : ([[ARG_TYPES]]) -> ()
  sycl.call(%arg0, %arg1, %arg2, %arg3, %arg4) {Function = @__init, MangledName = @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE, Type = @accessor} : (memref<?x!sycl_accessor_1_i32_read_write_global_buffer>, memref<?xi32>, !sycl.range<1>, !sycl.range<1>, !sycl.id<1>) -> ()
  return
}

// -----
