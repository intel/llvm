// RUN: sycl-mlir-opt -split-input-file -test-buffer-analysis %s 2>&1 | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (i64)>
!sycl_range_1_ = !sycl.range<[1], (i64)>

llvm.func @__gxx_personality_v0(...) -> i32

// CHECK-LABEL: test_tag: unknown_size:
// CHECK:         operand #0
// CHECK-NEXT:    buffer:
// CHECK-NEXT:      Sub-Buffer: No
// CHECK-NEXT:      Size: <unknown>
llvm.func @unknown_size() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%0, %1, %2, %3) {type = !sycl.buffer<[1, !llvm.void]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb1
^bb1:
  %4 = llvm.load %0 {tag = "unknown_size"} : !llvm.ptr -> i32
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: test_tag: constant_size:
// CHECK:         operand #0
// CHECK-NEXT:    buffer:
// CHECK-NEXT:      Sub-Buffer: No
// CHECK-NEXT:      Size: range{42}
llvm.func @constant_size() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %c42_i64 = arith.constant 42 : i64
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%1, %c42_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%0, %1, %2, %3) {type = !sycl.buffer<[1, !llvm.void]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb1
^bb1:
  %4 = llvm.load %0 {tag = "constant_size"} : !llvm.ptr -> i32
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: test_tag: sub_buffer_constant_size:
// CHECK:         operand #0
// CHECK-NEXT:    buffer:
// CHECK-NEXT:      Sub-Buffer: Yes
// CHECK-NEXT:      Size: range{128}
// CHECK-NEXT:      Base buffer: %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      Base buffer size: range{256}
// CHECK-NEXT:      Sub-buffer offset: id{64}
llvm.func @sub_buffer_constant_size() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %c64_i64 = arith.constant 64 : i64
  %c128_i64 = arith.constant 128 : i64
  %c256_i64 = arith.constant 256 : i64
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %subbuf = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %subbuf_range = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %subbuf_offset = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%1, %c256_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%subbuf_range, %c128_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%subbuf_offset, %c64_i64) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%0, %1, %2, %3) {type = !sycl.buffer<[1, !llvm.void]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb1
^bb1:
  sycl.host.constructor(%subbuf, %0, %subbuf_offset, %subbuf_range, %3) {type = !sycl.buffer<[1, !llvm.void, true]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb2
^bb2:
  %4 = llvm.load %subbuf {tag = "sub_buffer_constant_size"} : !llvm.ptr -> i32
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: test_tag: sub_buffer_unknown_offset_range:
// CHECK:         operand #0
// CHECK-NEXT:    buffer:
// CHECK-NEXT:      Sub-Buffer: Yes
// CHECK-NEXT:      Size: <unknown>
// CHECK-NEXT:      Base buffer: %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      Base buffer size: range{256}
// CHECK-NEXT:      Sub-buffer offset: <unknown>
llvm.func @sub_buffer_unknown_offset_range(%arg0 : i1) -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %c64_i64 = arith.constant 64 : i64
  %c128_i64 = arith.constant 128 : i64
  %c256_i64 = arith.constant 256 : i64
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %subbuf = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %subbuf_range = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %subbuf_offset = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%1, %c256_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%0, %1, %2, %3) {type = !sycl.buffer<[1, !llvm.void]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.cond_br %arg0, ^bb1, ^bb2
^bb1:
  sycl.host.constructor(%subbuf_range, %c128_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%subbuf_offset, %c64_i64) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
  llvm.br ^bb3
^bb2:
  sycl.host.constructor(%subbuf_range, %c64_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%subbuf_offset, %c128_i64) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
  llvm.br ^bb3
^bb3:
  sycl.host.constructor(%subbuf, %0, %subbuf_offset, %subbuf_range, %3) {type = !sycl.buffer<[1, !llvm.void, true]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb4
^bb4:
  %4 = llvm.load %subbuf {tag = "sub_buffer_unknown_offset_range"} : !llvm.ptr -> i32
  llvm.return %0 : !llvm.ptr
}

// CHECK-LABEL: test_tag: maybe_sub_buffer:
// CHECK:         operand #0
// CHECK-NEXT:    buffer:
// CHECK-NEXT:      Sub-Buffer: Maybe
// CHECK-NEXT:      Size: range{128}
llvm.func @maybe_sub_buffer(%arg0 : i1) -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %c64_i64 = arith.constant 64 : i64
  %c128_i64 = arith.constant 128 : i64
  %c256_i64 = arith.constant 256 : i64
  %0 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %subbuf = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %subbuf_range = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %subbuf_offset = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%1, %c128_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%0, %1, %2, %3) {type = !sycl.buffer<[1, !llvm.void]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.cond_br %arg0, ^bb1, ^bb2
^bb1:
  sycl.host.constructor(%subbuf_range, %c128_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%subbuf_offset, %c64_i64) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%subbuf, %0, %subbuf_offset, %subbuf_range, %3) {type = !sycl.buffer<[1, !llvm.void, true]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb3
^bb2:
  sycl.host.constructor(%subbuf, %1, %2, %3) {type = !sycl.buffer<[1, !llvm.void]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb3
^bb3:
  %4 = llvm.load %subbuf {tag = "maybe_sub_buffer"} : !llvm.ptr -> i32
  llvm.return %0 : !llvm.ptr
}


// CHECK-LABEL: test_tag: join_base_buffer:
// CHECK:         operand #0
// CHECK-NEXT:    buffer:
// CHECK-NEXT:      Sub-Buffer: Yes
// CHECK-NEXT:      Size: range{128}
// CHECK-NEXT:      Base buffer: <unknown>
// CHECK-NEXT:      Base buffer size: <unknown>
// CHECK-NEXT:      Sub-buffer offset: id{64}
llvm.func @join_base_buffer(%arg0 : i1) -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %c1_i32 = arith.constant 1 : i32
  %c64_i64 = arith.constant 64 : i64
  %c128_i64 = arith.constant 128 : i64
  %c256_i64 = arith.constant 256 : i64
  %c512_i64 = arith.constant 512 : i64
  %buf1 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %buf2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %subbuf = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %buf1_range = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %buf2_range = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %subbuf_range = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %subbuf_offset = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %c1_i32 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %c1_i32 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  sycl.host.constructor(%buf1_range, %c256_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%buf2_range, %c512_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%subbuf_range, %c128_i64) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%subbuf_offset, %c64_i64) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%buf1, %buf1_range, %2, %3) {type = !sycl.buffer<[1, !llvm.void]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%buf2, %buf2_range, %2, %3) {type = !sycl.buffer<[1, !llvm.void]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.cond_br %arg0, ^bb1, ^bb2
^bb1:
  sycl.host.constructor(%subbuf, %buf1, %subbuf_offset, %subbuf_range, %3) {type = !sycl.buffer<[1, !llvm.void, true]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb3
^bb2:
  sycl.host.constructor(%subbuf, %buf2, %subbuf_offset, %subbuf_range, %3) {type = !sycl.buffer<[1, !llvm.void, true]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  llvm.br ^bb3
^bb3:
  %4 = llvm.load %subbuf {tag = "join_base_buffer"} : !llvm.ptr -> i32
  llvm.return %buf1 : !llvm.ptr
}
