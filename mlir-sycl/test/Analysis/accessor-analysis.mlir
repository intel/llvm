// RUN: sycl-mlir-opt -split-input-file -test-accessor-analysis %s 2>&1 | FileCheck %s

llvm.func @__gxx_personality_v0(...) -> i32

!sycl_id_1_ = !sycl.id<[1], (i64)>
!sycl_range_1_ = !sycl.range<[1], (i64)>
!sycl_accessor_1_21llvm2Evoid_w_gb = !sycl.accessor<[1, !llvm.void, write, global_buffer], (!sycl_range_1_, !sycl_id_1_)>
!sycl_accessor_1_21llvm2Evoid_rw_gb = !sycl.accessor<[1, !llvm.void, read_write, global_buffer], (!llvm.void)>

// CHECK-LABEL: test_tag: full_information:
// CHECK:        operand #0
// CHECK-NEXT:     accessor:
// CHECK-NEXT:       Needs range: Yes
// CHECK-NEXT:       Range: range{128}
// CHECK-NEXT:       Needs offset: Yes
// CHECK-NEXT:       Offset: id{64}
// CHECK-NEXT:       Buffer: %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:       Buffer information:     Sub-Buffer: No
// CHECK-NEXT:       Size: range{256}
llvm.func @full_information() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %c256_i64 = arith.constant 256 : i64
  %range = arith.constant 128 : i64
  %offset = arith.constant 64 : i64
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%1, %c256_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%0, %1, %2, %3) {type = !sycl.buffer<[1, !llvm.void]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
   sycl.host.constructor(%acc, %0, %range, %offset, %2, %3) {type = !sycl_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb1
^bb1:
  %4 = llvm.load %acc {tag = "full_information"} : !llvm.ptr -> i32
  llvm.return %acc : !llvm.ptr
}

// CHECK-LABEL: test_tag: no_information:
// CHECK:        operand #0
// CHECK-NEXT:     accessor:
// CHECK-NEXT:       Needs range: Yes
// CHECK-NEXT:       Range: <unknown>
// CHECK-NEXT:       Needs offset: Yes
// CHECK-NEXT:       Offset: <unknown>
// CHECK-NEXT:       Buffer: %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:       Buffer information: <unknown>
llvm.func @no_information() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %c256_i64 = arith.constant 256 : i64
  %range = arith.constant 128 : i64
  %offset = arith.constant 64 : i64
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %4 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%acc, %0, %1, %2, %3, %4) {type = !sycl_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb1
^bb1:
  %5 = llvm.load %acc {tag = "no_information"} : !llvm.ptr -> i32
  llvm.return %acc : !llvm.ptr
}

// CHECK-LABEL: test_tag: join_accessor:
// CHECK:        operand #0
// CHECK-NEXT:     accessor:
// CHECK-NEXT:       Needs range: Yes
// CHECK-NEXT:       Range: <unknown>
// CHECK-NEXT:       Needs offset: Yes
// CHECK-NEXT:       Offset: <unknown>
// CHECK-NEXT:       Buffer: <unknown>
// CHECK-NEXT:       Buffer information:     Sub-Buffer: No
// CHECK-NEXT:       Size: <unknown>
llvm.func @join_accessor(%arg0 : i1) -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %c64_i64 = arith.constant 64 : i64
  %c128_i64 = arith.constant 128 : i64
  %c256_i64 = arith.constant 256 : i64
  %buf1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %buf2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc_range = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc_offset = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.cond_br %arg0, ^bb1, ^bb2
^bb1:
  sycl.host.constructor(%1, %c256_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%buf1, %1, %2, %3) {type = !sycl.buffer<[1, !llvm.void]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%acc_range, %c128_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%acc_offset, %c64_i64) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%acc, %buf1, %acc_range, %acc_offset, %2, %3) {type = !sycl_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb3
^bb2:
  sycl.host.constructor(%buf2, %1, %2, %3) {type = !sycl.buffer<[1, !llvm.void]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%acc_range, %c64_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%acc_offset, %c128_i64) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%acc, %buf2, %acc_range, %acc_offset, %2, %3) {type = !sycl_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb3
^bb3:
  %4 = llvm.load %acc {tag = "join_accessor"} : !llvm.ptr -> i32
  llvm.return %acc : !llvm.ptr
}

// COM: Test scenario: Two accessors on the same buffer, with no range
// information available for the accesors should yield may-alias.
// CHECK-LABEL: test_tag: alias_same_buffer_no_range_info:
// CHECK:        Alias (op#1 x op#2): MayAlias
llvm.func @alias_same_buffer_no_range_info() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %c256_i64 = arith.constant 256 : i64
  %range1 = arith.constant 128 : i64
  %offset1 = arith.constant 64 : i64
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %range2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %offset2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%1, %c256_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%0, %1, %2, %3) {type = !sycl.buffer<[1, !llvm.void]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%acc1, %0, %range1, %offset1, %2, %3) {type = !sycl_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%acc2, %0, %range2, %offset2, %2, %3) {type = !sycl_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb1
^bb1:
  sycl.host.constructor(%1, %acc1, %acc2) {type = !sycl_range_1_, tag = "alias_same_buffer_no_range_info"} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return %acc1 : !llvm.ptr
}


// COM: Test scenario: Two accessors on the same buffer, with no range
// information required for the accesors, should yield must-alias, as 
// both accessor cover the whole buffer.
// CHECK-LABEL: test_tag: alias_same_buffer_no_range_required:
// CHECK:        Alias (op#1 x op#2): MustAlias
llvm.func @alias_same_buffer_no_range_required() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %buf1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%acc1, %buf1, %2, %3) {type = !sycl_accessor_1_21llvm2Evoid_rw_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%acc2, %buf1, %2, %3) {type = !sycl_accessor_1_21llvm2Evoid_rw_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb1
^bb1:
  sycl.host.constructor(%1, %acc1, %acc2) {type = !sycl_range_1_, tag = "alias_same_buffer_no_range_required"} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return %acc1 : !llvm.ptr
}


// COM: Test scenario: Two accessors on the same buffer, with range
// information available for the accesors and no overlap between them should
// yield a no-alias.
// CHECK-LABEL: test_tag: alias_same_buffer_no_overlap:
// CHECK:        Alias (op#1 x op#2): NoAlias
llvm.func @alias_same_buffer_no_overlap() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %c256_i64 = arith.constant 256 : i64
  %range1 = arith.constant 128 : i64
  %offset1 = arith.constant 0 : i64
  %range2 = arith.constant 128 : i64
  %offset2 = arith.constant 128 : i64
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%1, %c256_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%0, %1, %2, %3) {type = !sycl.buffer<[1, !llvm.void]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%acc1, %0, %range1, %offset1, %2, %3) {type = !sycl_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%acc2, %0, %range2, %offset2, %2, %3) {type = !sycl_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb1
^bb1:
  sycl.host.constructor(%1, %acc1, %acc2) {type = !sycl_range_1_, tag = "alias_same_buffer_no_overlap"} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return %acc1 : !llvm.ptr
}


// COM: Test scenario: Two accessors on the same buffer, with range
// information available for the accesors and the two accessors' access
// range fully overlapping should yield must-alias.
// CHECK-LABEL: test_tag: alias_same_buffer_full_overlap:
// CHECK:        Alias (op#1 x op#2): MustAlias
llvm.func @alias_same_buffer_full_overlap() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %c256_i64 = arith.constant 256 : i64
  %range1 = arith.constant 128 : i64
  %offset1 = arith.constant 0 : i64
  %range2 = arith.constant 128 : i64
  %offset2 = arith.constant 0 : i64
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%1, %c256_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%0, %1, %2, %3) {type = !sycl.buffer<[1, !llvm.void]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%acc1, %0, %range1, %offset1, %2, %3) {type = !sycl_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%acc2, %0, %range2, %offset2, %2, %3) {type = !sycl_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb1
^bb1:
  sycl.host.constructor(%1, %acc1, %acc2) {type = !sycl_range_1_, tag = "alias_same_buffer_full_overlap"} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return %acc1 : !llvm.ptr
}


// COM: Test scenario: Two accessors on the two different buffers, with buffer
// information available should yield a no-alias, as both buffers are not 
// sub-buffers.
// CHECK-LABEL: test_tag: alias_two_buffers_with_info:
// CHECK:        Alias (op#1 x op#2): NoAlias
llvm.func @alias_two_buffers_with_info() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %c256_i64 = arith.constant 256 : i64
  %buf1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %buf2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%1, %c256_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%buf1, %1, %2, %3) {type = !sycl.buffer<[1, !llvm.void]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%buf2, %1, %2, %3) {type = !sycl.buffer<[1, !llvm.void]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%acc1, %buf1, %2, %3) {type = !sycl_accessor_1_21llvm2Evoid_rw_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%acc2, %buf2, %2, %3) {type = !sycl_accessor_1_21llvm2Evoid_rw_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb1
^bb1:
  sycl.host.constructor(%1, %acc1, %acc2) {type = !sycl_range_1_, tag = "alias_two_buffers_with_info"} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return %acc1 : !llvm.ptr
}


// COM: Test scenario: Two accessors on the two different buffers, with no
// buffer information available should yield a may-alias, as the buffers may
// be sub-buffers.
// CHECK-LABEL: test_tag: alias_two_buffers_no_info:
// CHECK:        Alias (op#1 x op#2): MayAlias
llvm.func @alias_two_buffers_no_info() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %buf1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %buf2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%acc1, %buf1, %2, %3) {type = !sycl_accessor_1_21llvm2Evoid_rw_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%acc2, %buf2, %2, %3) {type = !sycl_accessor_1_21llvm2Evoid_rw_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb1
^bb1:
  sycl.host.constructor(%1, %acc1, %acc2) {type = !sycl_range_1_, tag = "alias_two_buffers_no_info"} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.return %acc1 : !llvm.ptr
}
