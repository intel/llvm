// RUN: polygeist-opt -split-input-file -test-id-range-analysis %s 2>&1 | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (i64)>
!sycl_id_2_ = !sycl.id<[2], (i64)>
!sycl_id_3_ = !sycl.id<[3], (i64)>
  
// CHECK-LABEL: test_tag: constant_id:
// CHECK:         operand #0
// CHECK:         id:
// CHECK:           constant<42, 42>
llvm.func @constant_id() -> !llvm.ptr {
  %c1_i32 = arith.constant 1 : i32
  %c42_i64 = arith.constant 42 : i64
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%0, %c42_i64, %c42_i64) {type = !sycl_id_2_} : (!llvm.ptr, i64, i64) -> ()
  %1 = llvm.load %0 {tag = "constant_id"} : !llvm.ptr -> i32
  llvm.return %0 : !llvm.ptr
}


// CHECK-LABEL: test_tag: non_constant_id:
// CHECK:         operand #0
// CHECK:         id:
// CHECK:           fixed<3>
llvm.func @non_constant_id() -> !llvm.ptr {
  %c1_i32 = arith.constant 1 : i32
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::id.5", (struct<"class.sycl::_V1::detail::array.7", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.call @_Z6numberv() : () -> i64
  %2 = llvm.call @_Z6numberv() : () -> i64
  %3 = llvm.call @_Z6numberv() : () -> i64
  sycl.host.constructor(%0, %1, %2, %3) {type = !sycl_id_3_} : (!llvm.ptr, i64, i64, i64) -> ()
  %4 = llvm.load %0 {tag = "non_constant_id"} : !llvm.ptr -> i32
  llvm.return %0 : !llvm.ptr
}

llvm.func @_Z6numberv() -> i64


// CHECK-LABEL: test_tag: constrol_flow_constant:
// CHECK:         operand #0
// CHECK:         id:
// CHECK:           constant<42, 42>
llvm.func @control_flow_constant(%arg0 : i1) -> !llvm.ptr {
  %c1_i32 = arith.constant 1 : i32
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  scf.if %arg0 {
    %c42_i64 = arith.constant 42 : i64
    sycl.host.constructor(%0, %c42_i64, %c42_i64) {type = !sycl_id_2_} : (!llvm.ptr, i64, i64) -> ()
  } else {
    %c42_alias = arith.constant 42 : i64
    sycl.host.constructor(%0, %c42_alias, %c42_alias) {type = !sycl_id_2_} : (!llvm.ptr, i64, i64) -> ()
  }
  %1 = llvm.load %0 {tag = "constrol_flow_constant"} : !llvm.ptr -> i32
  llvm.return %0 : !llvm.ptr
}


// CHECK-LABEL: test_tag: constrol_flow_non_constant:
// CHECK:         operand #0
// CHECK:         id:
// CHECK:           fixed<2>
llvm.func @control_flow_non_constant(%arg0 : i1) -> !llvm.ptr {
  %c1_i32 = arith.constant 1 : i32
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  scf.if %arg0 {
    %c42_i64 = arith.constant 42 : i64
    sycl.host.constructor(%0, %c42_i64, %c42_i64) {type = !sycl_id_2_} : (!llvm.ptr, i64, i64) -> ()
  } else {
    %c25_i64 = arith.constant 25 : i64
    sycl.host.constructor(%0, %c25_i64, %c25_i64) {type = !sycl_id_2_} : (!llvm.ptr, i64, i64) -> ()
  }
  %1 = llvm.load %0 {tag = "constrol_flow_non_constant"} : !llvm.ptr -> i32
  llvm.return %0 : !llvm.ptr
}


// CHECK-LABEL: test_tag: multiple_use_1:
// CHECK:         operand #0
// CHECK:         id:
// CHECK:           constant<42>
// CHECK-LABEL: test_tag: multiple_use_2:
// CHECK:         operand #0
// CHECK:         id:
// CHECK:           constant<42>
llvm.func @multiple_use() -> !llvm.ptr {
  %c1_i32 = arith.constant 1 : i32
  %c42_i64 = arith.constant 42 : i64
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%0, %c42_i64) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
  %1 = llvm.load %0 {tag = "multiple_use_1"} : !llvm.ptr -> i32
  %2 = llvm.load %0 {tag = "multiple_use_2"} : !llvm.ptr -> i32
  llvm.return %0 : !llvm.ptr
}


// CHECK-LABEL: test_tag: control_flow_non_fixed:
// CHECK:         operand #0
// CHECK:         id:
// CHECK:           <unknown>
llvm.func @control_flow_non_fixed(%arg0 : i1, %arg1 : !llvm.ptr) -> !llvm.ptr {
  %c1_i32 = arith.constant 1 : i32
  %c42_i64 = arith.constant 42 : i64
  scf.if %arg0 {
    sycl.host.constructor(%arg1, %c42_i64) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
  } else {
    sycl.host.constructor(%arg1, %c42_i64, %c42_i64) {type = !sycl_id_2_} : (!llvm.ptr, i64, i64) -> ()
  }
  %0 = llvm.load %arg1 {tag = "control_flow_non_fixed"} : !llvm.ptr -> i32
  llvm.return %arg1 : !llvm.ptr
}


// -----

!sycl_range_1_ = !sycl.range<[1], (i64)>
!sycl_range_2_ = !sycl.range<[2], (i64)>
!sycl_range_3_ = !sycl.range<[3], (i64)>


// CHECK-LABEL: test_tag: constant_range:
// CHECK:         operand #0
// CHECK:         range:
// CHECK:           constant<42, 42>
llvm.func @constant_range() -> !llvm.ptr {
  %c1_i32 = arith.constant 1 : i32
  %c42_i64 = arith.constant 42 : i64
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%0, %c42_i64, %c42_i64) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
  %1 = llvm.load %0 {tag = "constant_range"} : !llvm.ptr -> i32
  llvm.return %0 : !llvm.ptr
}


// CHECK-LABEL: test_tag: non_constant_range:
// CHECK:         operand #0
// CHECK:         range:
// CHECK:           fixed<3>
llvm.func @non_constant_range() -> !llvm.ptr {
  %c1_i32 = arith.constant 1 : i32
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.5", (struct<"class.sycl::_V1::detail::array.7", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.call @_Z6numberv() : () -> i64
  %2 = llvm.call @_Z6numberv() : () -> i64
  %3 = llvm.call @_Z6numberv() : () -> i64
  sycl.host.constructor(%0, %1, %2, %3) {type = !sycl_range_3_} : (!llvm.ptr, i64, i64, i64) -> ()
  %4 = llvm.load %0 {tag = "non_constant_range"} : !llvm.ptr -> i32
  llvm.return %0 : !llvm.ptr
}

llvm.func @_Z6numberv() -> i64
