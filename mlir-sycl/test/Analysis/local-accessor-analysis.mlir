// RUN: sycl-mlir-opt -split-input-file -test-accessor-analysis %s 2>&1 | FileCheck %s

llvm.func @__gxx_personality_v0(...) -> i32

!sycl_id_1_ = !sycl.id<[1], (i64)>
!sycl_range_1_ = !sycl.range<[1], (i64)>
!sycl_local_accessor_1_21llvm2Evoid_w_gb = !sycl.local_accessor<[1, !llvm.void], (!sycl_range_1_)>
!sycl_local_accessor_1_21llvm2Evoid_rw_gb = !sycl.local_accessor<[1, !llvm.void], (!llvm.void)>
!sycl_accessor_1_21llvm2Evoid_w_gb = !sycl.accessor<[1, !llvm.void, write, global_buffer], (!llvm.void)>

// CHECK-LABEL: test_tag: full_information:
// CHECK:        operand #0
// CHECK-NEXT:     accessor:
// CHECK-NEXT:       (local_accessor)
// CHECK-NEXT:       Needs range: Yes
// CHECK-NEXT:       Range: range{128}
llvm.func @full_information(%handler: !llvm.ptr) -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %range = arith.constant 128 : i64
  %acc = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::local_accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%1, %range) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%acc, %1, %handler, %2, %3) {type = !sycl_local_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb1
^bb1:
  %4 = llvm.load %acc {tag = "full_information"} : !llvm.ptr -> i32
  llvm.return %acc : !llvm.ptr
}

// CHECK-LABEL: test_tag: no_information:
// CHECK:        operand #0
// CHECK-NEXT:     accessor:
// CHECK-NEXT:       (local_accessor)
// CHECK-NEXT:       Needs range: Yes
// CHECK-NEXT:       Range: <unknown>
llvm.func @no_information(%handler: !llvm.ptr, %range: !llvm.ptr) -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %acc = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::local_accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%acc, %range, %handler, %2, %3) {type = !sycl_local_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb1
^bb1:
  %5 = llvm.load %acc {tag = "no_information"} : !llvm.ptr -> i32
  llvm.return %acc : !llvm.ptr
}

// CHECK-LABEL: test_tag: join_accessor:
// CHECK:        operand #0
// CHECK-NEXT:     accessor:
// CHECK-NEXT:       (local_accessor)
// CHECK-NEXT:       Needs range: Yes
// CHECK-NEXT:       Range: range{128}
llvm.func @join_accessor(%handler: !llvm.ptr, %arg0 : i1) -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %range = arith.constant 128 : i64
  %acc = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::local_accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.cond_br %arg0, ^bb1, ^bb2
^bb1:
  sycl.host.constructor(%1, %range) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%acc, %1, %handler, %2, %3) {type = !sycl_local_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb3
^bb2:
  sycl.host.constructor(%1, %range) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%acc, %1, %handler, %2, %3) {type = !sycl_local_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb3
^bb3:
  %4 = llvm.load %acc {tag = "join_accessor"} : !llvm.ptr -> i32
  llvm.return %acc : !llvm.ptr
}

// COM: Test scenario: Two different local accessors cannot alias
// CHECK-LABEL: test_tag: no_alias
// CHECK:        Alias (op#1 x op#2): NoAlias
llvm.func @no_alias(%handler: !llvm.ptr, %range: !llvm.ptr, %props: !llvm.ptr, %codeloc: !llvm.ptr) -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::local_accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::local_accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%acc1, %range, %handler, %props, %codeloc) {type = !sycl_local_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%acc2, %range, %handler, %props, %codeloc) {type = !sycl_local_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb1
^bb1:
  sycl.host.constructor(%0, %acc2, %acc2) {type = !sycl_range_1_, tag = "no_alias"} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return %acc1 : !llvm.ptr
}

// CHECK-LABEL: test_tag: join_top:
// CHECK:        operand #0
// CHECK-NEXT:     accessor:
// CHECK-NEXT:       <TOP>
llvm.func @join_top(%arg0: i1, %acc: !llvm.ptr, %handler: !llvm.ptr, %range: !llvm.ptr, %buf: !llvm.ptr, %props: !llvm.ptr, %codeloc: !llvm.ptr) -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  llvm.cond_br %arg0, ^bb1, ^bb2
^bb1:
  // Device accessor
  sycl.host.constructor(%acc, %buf, %range, %props, %codeloc) {type = !sycl_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb3
^bb2:
  // Local accessor
  sycl.host.constructor(%acc, %range, %handler, %props, %codeloc) {type = !sycl_local_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb3
^bb3:
  %4 = llvm.load %acc {tag = "join_top"} : !llvm.ptr -> i32
  llvm.return %acc : !llvm.ptr
}

// COM: A device and a local accessor cannot alias.
// CHECK-LABEL: test_tag: no_alias_device
// CHECK:        Alias (op#1 x op#2): NoAlias
llvm.func @no_alias_device(%arg0: i1, %handler: !llvm.ptr, %range: !llvm.ptr, %buf: !llvm.ptr, %props: !llvm.ptr, %codeloc: !llvm.ptr) -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %dev_acc = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %local_acc = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::local_accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%dev_acc, %buf, %range, %props, %codeloc) {type = !sycl_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%local_acc, %range, %handler, %props, %codeloc) {type = !sycl_local_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%0, %dev_acc, %local_acc) {type = !sycl_range_1_, tag = "no_alias_device"} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return %0 : !llvm.ptr
}
