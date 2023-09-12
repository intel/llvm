// RUN: sycl-mlir-opt -test-nd-range-analysis %s 2>&1 | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (i64)>
!sycl_id_2_ = !sycl.id<[2], (i64)>
!sycl_id_3_ = !sycl.id<[3], (i64)>
!sycl_range_1_ = !sycl.range<[1], (i64)>
!sycl_range_2_ = !sycl.range<[2], (i64)>
!sycl_range_3_ = !sycl.range<[3], (i64)>
!sycl_nd_range_1_ = !sycl.nd_range<[1], (i64)>
!sycl_nd_range_2_ = !sycl.nd_range<[2], (i64)>
!sycl_nd_range_3_ = !sycl.nd_range<[3], (i64)>

// CHECK-LABEL: test_tag: constant_ndr_offset:
// CHECK-NEXT:    operand #0
// CHECK-NEXT:    nd_range:
// CHECK-NEXT:      <global_size: constant<42>, local_size: constant<21>, offset: constant<10>>
llvm.func @constant_ndr_offset() -> !llvm.ptr {
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i64
  %c21 = arith.constant 21 : i64
  %c42 = arith.constant 42 : i64
  %ndr = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %global_size = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %local_size = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %offset = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%global_size, %c42) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%local_size, %c21) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%offset, %c10) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%ndr, %global_size, %local_size, %offset) {type = !sycl_nd_range_1_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  %1 = llvm.load %ndr {tag = "constant_ndr_offset"} : !llvm.ptr -> i32
  llvm.return %ndr : !llvm.ptr
}

// CHECK-LABEL: test_tag: constant_ndr:
// CHECK-NEXT:    operand #0
// CHECK-NEXT:    nd_range:
// CHECK-NEXT:      <global_size: constant<42, 21>, local_size: constant<21, 42>, offset: constant<0, 0>>
llvm.func @constant_ndr() -> !llvm.ptr {
  %c1 = arith.constant 1 : i32
  %c21 = arith.constant 21 : i64
  %c42 = arith.constant 42 : i64
  %ndr = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::nd_range.1", (struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %global_size = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %local_size = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%global_size, %c42, %c21) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
  sycl.host.constructor(%local_size, %c21, %c42) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
  sycl.host.constructor(%ndr, %global_size, %local_size) {type = !sycl_nd_range_2_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  %1 = llvm.load %ndr {tag = "constant_ndr"} : !llvm.ptr -> i32
  llvm.return %ndr : !llvm.ptr
}

// CHECK-LABEL: test_tag: copy_ndr:
// CHECK-NEXT:    operand #0
// CHECK-NEXT:    nd_range:
// CHECK-NEXT:      <global_size: constant<42, 21>, local_size: constant<21, 42>, offset: constant<0, 0>>
llvm.func @copy_ndr() -> !llvm.ptr {
  %c1 = arith.constant 1 : i32
  %c21 = arith.constant 21 : i64
  %c42 = arith.constant 42 : i64
  %ndr = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::nd_range.1", (struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %ndr_cpy = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::nd_range.1", (struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %global_size = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %local_size = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%global_size, %c42, %c21) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
  sycl.host.constructor(%local_size, %c21, %c42) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
  sycl.host.constructor(%ndr, %global_size, %local_size) {type = !sycl_nd_range_2_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%ndr_cpy, %ndr) {type = !sycl_nd_range_2_} : (!llvm.ptr, !llvm.ptr) -> ()
  %1 = llvm.load %ndr_cpy {tag = "copy_ndr"} : !llvm.ptr -> i32
  llvm.return %ndr_cpy : !llvm.ptr
}

// CHECK-LABEL: test_tag: propagate_dims_ndr:
// CHECK-NEXT:    operand #0
// CHECK-NEXT:    nd_range:
// CHECK-NEXT:      <global_size: fixed<3>, local_size: fixed<3>, offset: constant<0, 0, 0>>
llvm.func @propagate_dims_ndr(%global_size: !llvm.ptr,
                              %local_size: !llvm.ptr) -> !llvm.ptr {
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i64
  %c21 = arith.constant 21 : i64
  %c42 = arith.constant 42 : i64
  %ndr = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::nd_range.2", (struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<3 x i64>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %offset = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.2", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%offset) {type = !sycl_id_3_} : (!llvm.ptr) -> ()
  sycl.host.constructor(%ndr, %global_size, %local_size, %offset) {type = !sycl_nd_range_3_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  %1 = llvm.load %ndr {tag = "propagate_dims_ndr"} : !llvm.ptr -> i32
  llvm.return %ndr : !llvm.ptr
}

// CHECK-LABEL: test_tag: fixed_dims_ndr:
// CHECK-NEXT:    operand #0
// CHECK-NEXT:    nd_range:
// CHECK-NEXT:      <global_size: fixed<3>, local_size: fixed<3>, offset: fixed<3>>
llvm.func @fixed_dims_ndr(%global_size: !llvm.ptr,
                          %local_size: !llvm.ptr,
                          %offset: !llvm.ptr) -> !llvm.ptr {
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i64
  %c21 = arith.constant 21 : i64
  %c42 = arith.constant 42 : i64
  %ndr = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::nd_range.2", (struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.2", (array<3 x i64>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%ndr, %global_size, %local_size, %offset) {type = !sycl_nd_range_3_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  %1 = llvm.load %ndr {tag = "fixed_dims_ndr"} : !llvm.ptr -> i32
  llvm.return %ndr : !llvm.ptr
}

// CHECK-LABEL: test_tag: diff_dims_ndr:
// CHECK-NEXT:    operand #0
// CHECK-NEXT:    nd_range:
// CHECK-NEXT:      <global_size: fixed<2>, local_size: fixed<2>, offset: fixed<2>>
llvm.func @diff_dims_ndr() -> !llvm.ptr {
  %c1 = arith.constant 1 : i32
  %c21 = arith.constant 21 : i64
  %c42 = arith.constant 42 : i64
  %ndr = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::nd_range.1", (struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %global_size = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %local_size = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%global_size, %c42, %c21, %c21) {type = !sycl_range_3_} : (!llvm.ptr, i64, i64, i64) -> ()
  sycl.host.constructor(%local_size, %c21, %c42) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
  sycl.host.constructor(%ndr, %global_size, %local_size) {type = !sycl_nd_range_2_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  %1 = llvm.load %ndr {tag = "diff_dims_ndr"} : !llvm.ptr -> i32
  llvm.return %ndr : !llvm.ptr
}

// CHECK-LABEL: test_tag: join_ndr:
// CHECK-NEXT:    operand #0
// CHECK-NEXT:    nd_range:
// CHECK-NEXT:      <global_size: fixed<1>, local_size: constant<21>, offset: constant<10>>
llvm.func @join_ndr(%arg0: i1) -> !llvm.ptr {
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i64
  %c21 = arith.constant 21 : i64
  %c42 = arith.constant 42 : i64
  %ndr = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %global_size = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %local_size = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %offset = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%local_size, %c21) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  scf.if %arg0 {
    sycl.host.constructor(%global_size, %c42) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
    sycl.host.constructor(%offset, %c10) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
    sycl.host.constructor(%ndr, %global_size, %local_size, %offset) {type = !sycl_nd_range_1_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  } else {
    sycl.host.constructor(%global_size, %c10) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
    sycl.host.constructor(%offset, %c10) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
    sycl.host.constructor(%ndr, %global_size, %local_size, %offset) {type = !sycl_nd_range_1_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  }
  %1 = llvm.load %ndr {tag = "join_ndr"} : !llvm.ptr -> i32
  llvm.return %ndr : !llvm.ptr
}

// CHECK-LABEL: test_tag: join_ndr_top:
// CHECK-NEXT:    operand #0
// CHECK-NEXT:    nd_range:
// CHECK-NEXT:      <unknown>
llvm.func @join_ndr_top(%arg0: i1) -> !llvm.ptr {
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i64
  %c21 = arith.constant 21 : i64
  %c42 = arith.constant 42 : i64
  %ndr = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %global_size = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %local_size = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %offset = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  scf.if %arg0 {
    sycl.host.constructor(%global_size, %c42, %c42) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
    sycl.host.constructor(%local_size, %c21, %c21) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
    sycl.host.constructor(%offset, %c10, %c10) {type = !sycl_id_2_} : (!llvm.ptr, i64, i64) -> ()
    sycl.host.constructor(%ndr, %global_size, %local_size, %offset) {type = !sycl_nd_range_2_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  } else {
    sycl.host.constructor(%global_size, %c10) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
    sycl.host.constructor(%local_size, %c21) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
    sycl.host.constructor(%offset, %c10) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
    sycl.host.constructor(%ndr, %global_size, %local_size, %offset) {type = !sycl_nd_range_1_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  }
  %1 = llvm.load %ndr {tag = "join_ndr_top"} : !llvm.ptr -> i32
  llvm.return %ndr : !llvm.ptr
}
