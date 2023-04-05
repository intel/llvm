// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm='use-opaque-pointers=0' -verify-diagnostics %s | FileCheck %s

//===-------------------------------------------------------------------------------------------------===//
// sycl.call with non void return type
//===-------------------------------------------------------------------------------------------------===//

// CHECK: llvm.func @foo() -> [[RET_TYPE:i32]]
// CHECK: llvm.func @test() -> [[RET_TYPE]] {
// CHECK-NEXT:  %0 = llvm.call @foo() : () -> [[RET_TYPE]]
// CHECK-NEXT:  llvm.return %0 : [[RET_TYPE]]

func.func private @foo() -> (i32)

func.func @test() -> (i32) {
  %0 = sycl.call @foo() {MangledFunctionName = @foo, TypeName = @accessor} : () -> i32
  return %0 : i32
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Member functions for sycl::accessor
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK: llvm.func @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE([[ARG_TYPES:!llvm.ptr<struct<"class.sycl::_V1::accessor.1",.*]])
func.func private @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE(memref<?x!sycl_accessor_1_i32_rw_gb>, memref<?xi32>, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_)

func.func @accessorInit1(%arg0: memref<?x!sycl_accessor_1_i32_rw_gb>, %arg1: memref<?xi32>, %arg2: !sycl_range_1_, %arg3: !sycl_range_1_, %arg4: !sycl_id_1_) {
  // CHECK: llvm.call @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE({{.*}}) : ([[ARG_TYPES]]) -> ()
  sycl.call @__init(%arg0, %arg1, %arg2, %arg3, %arg4) {MangledFunctionName = @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE, TypeName = @accessor} : (memref<?x!sycl_accessor_1_i32_rw_gb>, memref<?xi32>, !sycl_range_1_, !sycl_range_1_, !sycl_id_1_) -> ()
  return
}

// -----
