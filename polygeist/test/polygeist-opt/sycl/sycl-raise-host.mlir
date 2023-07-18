// RUN: polygeist-opt --sycl-raise-host --split-input-file %s | FileCheck %s

// CHECK-LABEL: gpu.module @device_functions
gpu.module @device_functions {
// CHECK:         gpu.func @foo() kernel
  gpu.func @foo() kernel {
    gpu.return
  }
}

// CHECK-LABEL: llvm.mlir.global private unnamed_addr constant @kernel_ref("foo\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
llvm.mlir.global private unnamed_addr constant @kernel_ref("foo\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}

llvm.func @__gxx_personality_v0(...) -> i32

// CHECK-LABEL:   llvm.func @f(
// CHECK-SAME:                 %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                 %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64) -> i32 attributes {personality = @__gxx_personality_v0} {
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[VAL_5:.*]] = sycl.host.get_kernel @device_functions::@foo : !llvm.ptr
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::handler", (i64)>
// CHECK-NEXT:      sycl.host.handler.set_kernel %[[VAL_0]] -> @device_functions::@foo : !llvm.ptr
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.invoke @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm(%[[VAL_6]], %[[VAL_1]], %[[VAL_2]], %[[VAL_5]], %[[VAL_3]]) to ^bb2 unwind ^bb1 {RaiseSetKernelVisited} : (!llvm.ptr, i64, i64, !llvm.ptr, i64) -> !llvm.ptr
// CHECK-NEXT:    ^bb1:
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
// CHECK-NEXT:      llvm.resume %[[VAL_8]] : !llvm.struct<(ptr, i32)>
// CHECK-NEXT:    ^bb2:
// CHECK-NEXT:      llvm.return %[[VAL_4]] : i32
// CHECK-NEXT:    }
llvm.func @f(%handler: !llvm.ptr, %pos: i64, %count: i64, %count2: i64) -> i32 attributes {personality = @__gxx_personality_v0} {
  %kn = llvm.mlir.addressof @kernel_ref : !llvm.ptr
  %gep = llvm.getelementptr inbounds %handler[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::handler", (i64)>
  %set = llvm.invoke @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm(%gep, %pos, %count, %kn, %count2) to ^bb1 unwind ^bb0 : (!llvm.ptr, i64, i64, !llvm.ptr, i64) -> !llvm.ptr
^bb0:
  %lp = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %lp : !llvm.struct<(ptr, i32)>
^bb1:
  %c0 = arith.constant 0 : i32
  llvm.return %c0 : i32
}

// -----

llvm.func @__gxx_personality_v0(...) -> i32

// CHECK-LABEL:   llvm.func @raise_buffer() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::range", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           sycl.host.constructor(%[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]]) {type = !sycl.buffer<[1, !llvm.void]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// CHECK:           llvm.return %[[VAL_1]] : !llvm.ptr
// CHECK:         }
llvm.func @raise_buffer() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %0 = arith.constant 1 : i32
  %1 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::range", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %4 = llvm.alloca %0 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr

  llvm.invoke @_ZN4sycl3_V16bufferIiLi1ENS0_6detail17aligned_allocatorIiEEvEC2ERKNS0_5rangeILi1EEERKNS0_13property_listENS2_13code_locationE(%1, %2, %3, %4) to ^bb1 unwind ^bb0 : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()

^bb0:
  %lp = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %lp : !llvm.struct<(ptr, i32)>
^bb1:
  llvm.return %1 : !llvm.ptr
}

// CHECK-LABEL:   llvm.func @raise_sub_buffer() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::id", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::range", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           sycl.host.constructor(%[[VAL_2]], %[[VAL_1]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]]) {type = !sycl.buffer<[1, !llvm.void, true]>} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// CHECK:           llvm.return %[[VAL_1]] : !llvm.ptr
// CHECK:         }
llvm.func @raise_sub_buffer() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %0 = arith.constant 1 : i32
  %1 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::id", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %4 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::range", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %5 = llvm.alloca %0 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr

  llvm.invoke @_ZN4sycl3_V16bufferIiLi1ENS0_6detail17aligned_allocatorIiEEvEC2ERS5_RKNS0_2idILi1EEERKNS0_5rangeILi1EEENS2_13code_locationE(%2, %1, %3, %4, %5) to ^bb1 unwind ^bb0 : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()

^bb0:
  %lp = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %lp : !llvm.struct<(ptr, i32)>
^bb1:
  llvm.return %1 : !llvm.ptr
}


// -----

llvm.func @__gxx_personality_v0(...) -> i32

// CHECK-LABEL: !sycl_accessor_1_21llvm2Evoid_rw_gb = !sycl.accessor<[1, !llvm.void, read_write, global_buffer], (!llvm.void)>
// CHECK:       !sycl_id_1_ = !sycl.id<[1], (!llvm.void)>
// CHECK:       !sycl_range_1_ = !sycl.range<[1], (!llvm.void)>
// CHECK:       !sycl_accessor_1_21llvm2Evoid_w_gb = !sycl.accessor<[1, !llvm.void, write, global_buffer], (!sycl_range_1_, !sycl_id_1_)>

// CHECK-LABEL:   llvm.func @raise_accessor() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 128 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 64 : i64
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_6:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_7:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::accessor.5", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           sycl.host.constructor(%[[VAL_4]], %[[VAL_3]], %[[VAL_5]], %[[VAL_6]]) {type = !sycl_accessor_1_21llvm2Evoid_rw_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// CHECK:           sycl.host.constructor(%[[VAL_7]], %[[VAL_3]], %[[VAL_1]], %[[VAL_2]], %[[VAL_5]], %[[VAL_6]]) {type = !sycl_accessor_1_21llvm2Evoid_w_gb} : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:
// CHECK:           llvm.return %[[VAL_3]] : !llvm.ptr
// CHECK:         }
llvm.func @raise_accessor() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %0 = arith.constant 1 : i32
  %range = arith.constant 128 : i64
  %offset = arith.constant 64 : i64
  %1 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %4 = llvm.alloca %0 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %5 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::accessor.5", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr

  llvm.invoke @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE1ENS0_3ext6oneapi22accessor_property_listIJEEEEC2IiLi1ENS0_6detail17aligned_allocatorIiEEvEERNS0_6bufferIT_XT0_ET1_NSt9enable_ifIXaagtT0_Li0EleT0_Li3EEvE4typeEEERKNS0_13property_listENSC_13code_locationE(%2, %1, %3, %4) to ^bb1 unwind ^bb0 : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()

^bb0:
  %lp = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %lp : !llvm.struct<(ptr, i32)>
^bb1:
  llvm.invoke @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_6targetE2014ELNS2_11placeholderE1ENS0_3ext6oneapi22accessor_property_listIJEEEEC2IiLi1ENS0_6detail17aligned_allocatorIiEEvEERNS0_6bufferIT_XT0_ET1_NSt9enable_ifIXaagtT0_Li0EleT0_Li3EEvE4typeEEENS0_5rangeILi1EEENS0_2idILi1EEERKNS0_13property_listENSC_13code_locationE(%5, %1, %range, %offset, %3, %4) to ^bb3 unwind ^bb2 : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr) -> ()
^bb2:
  %lp1 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %lp1 : !llvm.struct<(ptr, i32)>
^bb3:
  llvm.return %1 : !llvm.ptr
}


// -----

// CHECK-LABEL: !sycl_id_1_ = !sycl.id<[1], (i64)>
// CHECK-LABEL: !sycl_id_2_ = !sycl.id<[2], (i64)>
// CHECK-LABEL: !sycl_id_3_ = !sycl.id<[3], (i64)>

// CHECK-LABEL:   llvm.func @raise_id_1() -> !llvm.ptr {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 42 : i64
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           sycl.host.constructor(%[[VAL_2]], %[[VAL_1]]) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
llvm.func @raise_id_1() -> !llvm.ptr {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 42 : i64
  %2 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %1, %2 {alignment = 8 : i64} : i64, !llvm.ptr
  llvm.return %2 : !llvm.ptr
}


// CHECK-LABEL:   llvm.func @raise_id_2() -> !llvm.ptr {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 42 : i64
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           sycl.host.constructor(%[[VAL_2]], %[[VAL_1]], %[[VAL_1]]) {type = !sycl_id_2_} : (!llvm.ptr, i64, i64) -> ()
llvm.func @raise_id_2() -> !llvm.ptr {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 42 : i64
  %2 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %1, %2 {alignment = 8 : i64} : i64, !llvm.ptr
  %3 = llvm.getelementptr inbounds %2[8] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %1, %3 {alignment = 8 : i64} : i64, !llvm.ptr
  llvm.return %2 : !llvm.ptr
}


// CHECK-LABEL:   llvm.func @raise_id_3() -> !llvm.ptr {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::id.5", (struct<"class.sycl::_V1::detail::array.7", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_2:.*]] = llvm.call @_Z6numberv() : () -> i64
// CHECK:           %[[VAL_3:.*]] = llvm.call @_Z6numberv() : () -> i64
// CHECK:           %[[VAL_4:.*]] = llvm.call @_Z6numberv() : () -> i64
// CHECK:           sycl.host.constructor(%[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]]) {type = !sycl_id_3_} : (!llvm.ptr, i64, i64, i64) -> ()
llvm.func @raise_id_3() -> !llvm.ptr {
  %0 = arith.constant 1 : i32
  %1 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::id.5", (struct<"class.sycl::_V1::detail::array.7", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.call @_Z6numberv() : () -> i64
  %3 = llvm.call @_Z6numberv() : () -> i64
  %4 = llvm.call @_Z6numberv() : () -> i64
  llvm.store %2, %1 {alignment = 8 : i64} : i64, !llvm.ptr
  %5 = llvm.getelementptr inbounds %1[8] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %3, %5 {alignment = 8 : i64} : i64, !llvm.ptr
  %6 = llvm.getelementptr inbounds %1[16] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %4, %6 {alignment = 8 : i64} : i64, !llvm.ptr
  llvm.return %1 : !llvm.ptr
}

llvm.func @_Z6numberv() -> (i64)

// COM: We should not find this pattern with dimensions == 1
// CHECK-LABEL:   llvm.func @raise_id_default() {
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::id.5", (struct<"class.sycl::_V1::detail::array.7", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_1]]) {type = !sycl_id_1_} : (!llvm.ptr) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]]) {type = !sycl_id_2_} : (!llvm.ptr) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_3]]) {type = !sycl_id_3_} : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
llvm.func @raise_id_default() {
  %c0 = llvm.mlir.constant(0 : i8) : i8
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len1 = llvm.mlir.constant(8 : i64) : i64
  %len2 = llvm.mlir.constant(16 : i64) : i64
  %len3 = llvm.mlir.constant(24 : i64) : i64
  %id1 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %id2 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %id3 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id.5", (struct<"class.sycl::_V1::detail::array.7", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  "llvm.intr.memset"(%id1, %c0, %len1) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  "llvm.intr.memset"(%id2, %c0, %len2) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  "llvm.intr.memset"(%id3, %c0, %len3) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  llvm.return
}

// COM: We should not find this pattern with dimensions >= 2
// CHECK-LABEL:   llvm.func @raise_id_store_default() {
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::id.5", (struct<"class.sycl::_V1::detail::array.7", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_1]]) {type = !sycl_id_1_} : (!llvm.ptr) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]]) {type = !sycl_id_2_} : (!llvm.ptr) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_3]]) {type = !sycl_id_3_} : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
llvm.func @raise_id_store_default() {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %id1 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %id2 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %id3 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id.5", (struct<"class.sycl::_V1::detail::array.7", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %c0, %id1 : i64, !llvm.ptr
  llvm.store %c0, %id2 : i64, !llvm.ptr
  %id2.1 = llvm.getelementptr inbounds %id2[1] : (!llvm.ptr) -> !llvm.ptr, i64
  llvm.store %c0, %id2.1 : i64, !llvm.ptr
  llvm.store %c0, %id3 : i64, !llvm.ptr
  %id3.1 = llvm.getelementptr inbounds %id3[1] : (!llvm.ptr) -> !llvm.ptr, i64
  llvm.store %c0, %id3.1 : i64, !llvm.ptr
  %id3.2 = llvm.getelementptr inbounds %id3[2] : (!llvm.ptr) -> !llvm.ptr, i64
  llvm.store %c0, %id3.2 : i64, !llvm.ptr
  llvm.return
}

// COM: We should not find this pattern with dimensions == 1
// CHECK-LABEL:   llvm.func @raise_id_copy(
// CHECK-SAME:                             %[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr, %[[VAL_2:.*]]: !llvm.ptr) {
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::id.5", (struct<"class.sycl::_V1::detail::array.7", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_4]], %[[VAL_0]]) {type = !sycl_id_1_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_5]], %[[VAL_1]]) {type = !sycl_id_2_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_6]], %[[VAL_2]]) {type = !sycl_id_3_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
llvm.func @raise_id_copy(%other1:!llvm.ptr, %other2:!llvm.ptr, %other3:!llvm.ptr) {
  %false = llvm.mlir.constant(0 : i1) : i1
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len1 = llvm.mlir.constant(8 : i64) : i64
  %len2 = llvm.mlir.constant(16 : i64) : i64
  %len3 = llvm.mlir.constant(24 : i64) : i64
  %id1 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %id2 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %id3 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id.5", (struct<"class.sycl::_V1::detail::array.7", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  "llvm.intr.memcpy"(%id1, %other1, %len1, %false) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
  "llvm.intr.memcpy"(%id2, %other2, %len2, %false) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
  "llvm.intr.memcpy"(%id3, %other3, %len3, %false) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
  llvm.return
}

// ----- 

// CHECK-LABEL: !sycl_range_1_ = !sycl.range<[1], (i64)>
// CHECK-LABEL: !sycl_range_2_ = !sycl.range<[2], (i64)>
// CHECK-LABEL: !sycl_range_3_ = !sycl.range<[3], (i64)>

// CHECK-LABEL:   llvm.func @raise_range_1() -> !llvm.ptr {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 42 : i64
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           sycl.host.constructor(%[[VAL_2]], %[[VAL_1]]) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
llvm.func @raise_range_1() -> !llvm.ptr {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 42 : i64
  %2 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %1, %2 {alignment = 8 : i64} : i64, !llvm.ptr
  llvm.return %2 : !llvm.ptr
}


// CHECK-LABEL:   llvm.func @raise_range_2() -> !llvm.ptr {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 42 : i64
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           sycl.host.constructor(%[[VAL_2]], %[[VAL_1]], %[[VAL_1]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
llvm.func @raise_range_2() -> !llvm.ptr {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 42 : i64
  %2 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %1, %2 {alignment = 8 : i64} : i64, !llvm.ptr
  %3 = llvm.getelementptr inbounds %2[8] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %1, %3 {alignment = 8 : i64} : i64, !llvm.ptr
  llvm.return %2 : !llvm.ptr
}


// CHECK-LABEL:   llvm.func @raise_range_3() -> !llvm.ptr {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::range.5", (struct<"class.sycl::_V1::detail::array.7", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_2:.*]] = llvm.call @_Z6numberv() : () -> i64
// CHECK:           %[[VAL_3:.*]] = llvm.call @_Z6numberv() : () -> i64
// CHECK:           %[[VAL_4:.*]] = llvm.call @_Z6numberv() : () -> i64
// CHECK:           sycl.host.constructor(%[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]]) {type = !sycl_range_3_} : (!llvm.ptr, i64, i64, i64) -> ()
llvm.func @raise_range_3() -> !llvm.ptr {
  %0 = arith.constant 1 : i32
  %1 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::range.5", (struct<"class.sycl::_V1::detail::array.7", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.call @_Z6numberv() : () -> i64
  %3 = llvm.call @_Z6numberv() : () -> i64
  %4 = llvm.call @_Z6numberv() : () -> i64
  llvm.store %2, %1 {alignment = 8 : i64} : i64, !llvm.ptr
  %5 = llvm.getelementptr inbounds %1[8] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %3, %5 {alignment = 8 : i64} : i64, !llvm.ptr
  %6 = llvm.getelementptr inbounds %1[16] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %4, %6 {alignment = 8 : i64} : i64, !llvm.ptr
  llvm.return %1 : !llvm.ptr
}

llvm.func @_Z6numberv() -> (i64)

// COM: We should not find this pattern with dimensions == 1
// CHECK-LABEL:   llvm.func @raise_range_copy(
// CHECK-SAME:                                %[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr, %[[VAL_2:.*]]: !llvm.ptr) {
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::range.5", (struct<"class.sycl::_V1::detail::array.7", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_4]], %[[VAL_0]]) {type = !sycl_range_1_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_5]], %[[VAL_1]]) {type = !sycl_range_2_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_6]], %[[VAL_2]]) {type = !sycl_range_3_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
llvm.func @raise_range_copy(%other1:!llvm.ptr, %other2:!llvm.ptr, %other3:!llvm.ptr) {
  %false = llvm.mlir.constant(0 : i1) : i1
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len1 = llvm.mlir.constant(8 : i64) : i64
  %len2 = llvm.mlir.constant(16 : i64) : i64
  %len3 = llvm.mlir.constant(24 : i64) : i64
  %range1 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %range2 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %range3 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range.5", (struct<"class.sycl::_V1::detail::array.7", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  "llvm.intr.memcpy"(%range1, %other1, %len1, %false) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
  "llvm.intr.memcpy"(%range2, %other2, %len2, %false) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
  "llvm.intr.memcpy"(%range3, %other3, %len3, %false) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
  llvm.return
}

// ----- 

// COM: Check the raising patterns are not applied to device code.

// CHECK-LABEL: gpu.module @device_functions {
// CHECK-NOT:   sycl.host
gpu.module @device_functions {
  llvm.func @raise_range_1() -> !llvm.ptr {
    %0 = arith.constant 1 : i32
    %1 = arith.constant 42 : i64
    %2 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %2 {alignment = 8 : i64} : i64, !llvm.ptr
    llvm.return %2 : !llvm.ptr
  }
}

// -----

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>

gpu.module @device_functions {
  gpu.func @foo() kernel {
    gpu.return
  }
}

llvm.mlir.global private unnamed_addr constant @range_str("range\00")
    {addr_space = 0 : i32, alignment = 1 : i64, dso_local}

// COM: Check we can raise parallel_for(globalSize, kernel) to sycl.host.handler.set_nd_range

// CHECK-LABEL:   llvm.func @raise_set_globalsize(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !llvm.ptr) {
// CHECK-DAG:       %[[VAL_1:.*]] = llvm.mlir.constant(512 : i64) : i64
// CHECK-DAG:       %[[VAL_2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      sycl.host.handler.set_kernel %[[VAL_0]] -> @device_functions::@foo : !llvm.ptr
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_3]], %[[VAL_1]]) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:      sycl.host.handler.set_nd_range %[[VAL_0]] -> range %[[VAL_3]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
llvm.func @raise_set_globalsize(%handler: !llvm.ptr) {
  %c0 = llvm.mlir.constant (0 : i32) : i32
  %c1 = llvm.mlir.constant (1 : i64) : i64
  sycl.host.handler.set_kernel %handler -> @device_functions::@foo : !llvm.ptr
  %c512 = llvm.mlir.constant (512 : i64) : i64
  %nullptr = llvm.mlir.null : !llvm.ptr
  %range = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  sycl.host.constructor(%range, %c512) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  %str = llvm.mlir.addressof @range_str : !llvm.ptr
  "llvm.intr.var.annotation"(%range, %str, %str, %c0, %nullptr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
  llvm.return
}

// -----

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>

gpu.module @device_functions {
  gpu.func @foo() kernel {
    gpu.return
  }
}

llvm.mlir.global private unnamed_addr constant @range_str("range\00")
    {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
llvm.mlir.global private unnamed_addr constant @offset_str("offset\00")
    {addr_space = 0 : i32, alignment = 1 : i64, dso_local}

// COM: Check we can raise parallel_for(globalSize, offset, kernel) to sycl.host.handler.set_nd_range

// CHECK-LABEL:   llvm.func @raise_set_globalsize_offset(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !llvm.ptr) {
// CHECK-DAG:       %[[VAL_1:.*]] = llvm.mlir.constant(512 : i64) : i64
// CHECK-DAG:       %[[VAL_2:.*]] = llvm.mlir.constant(100 : i64) : i64
// CHECK-DAG:       %[[VAL_3:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      sycl.host.handler.set_kernel %[[VAL_0]] -> @device_functions::@foo : !llvm.ptr
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_4]], %[[VAL_1]]) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_5]], %[[VAL_2]]) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:      sycl.host.handler.set_nd_range %[[VAL_0]] -> range %[[VAL_4]], offset %[[VAL_5]] : !llvm.ptr, !llvm.ptr, !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
llvm.func @raise_set_globalsize_offset(%handler: !llvm.ptr) {
  %c0 = llvm.mlir.constant (0 : i32) : i32
  %c1 = llvm.mlir.constant (1 : i64) : i64
  sycl.host.handler.set_kernel %handler -> @device_functions::@foo : !llvm.ptr
  %c100 = llvm.mlir.constant (100 : i64) : i64
  %c512 = llvm.mlir.constant (512 : i64) : i64
  %nullptr = llvm.mlir.null : !llvm.ptr
  %range = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  sycl.host.constructor(%range, %c512) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  %offset = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  sycl.host.constructor(%offset, %c100) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
  %rangeStr = llvm.mlir.addressof @range_str : !llvm.ptr
  %offsetStr = llvm.mlir.addressof @offset_str : !llvm.ptr
  "llvm.intr.var.annotation"(%range, %rangeStr, %rangeStr, %c0, %nullptr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
  "llvm.intr.var.annotation"(%offset, %offsetStr, %offsetStr, %c0, %nullptr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
  llvm.return
}

// -----

// COM: Check we can raise nd_range copy constructor

// CHECK-LABEL:   llvm.func @raise_nd_range_copy(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !llvm.ptr) {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_0]]) {type = !sycl_nd_range_1_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

llvm.func @raise_nd_range_copy(%other: !llvm.ptr) {
  %c1 = llvm.mlir.constant (1 : i32) : i32
  %len = llvm.mlir.constant (24 : i64) : i64
  %nd = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)>)> : (i32) -> !llvm.ptr
  "llvm.intr.memcpy"(%nd, %other, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  llvm.return
}

// -----

// COM: Check we can raise nd_range constructor using memcpy

// CHECK-LABEL:   llvm.func @raise_nd_range_memcpy(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !llvm.ptr) {
// CHECK-DAG:       %[[VAL_2:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK-DAG:       %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:       %[[VAL_5:.*]] = llvm.mlir.constant(24 : i64) : i64
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_6]], %[[VAL_0]], %[[VAL_1]]) {type = !sycl_nd_range_3_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.getelementptr inbounds %[[VAL_6]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>)>
// CHECK-NEXT:      "llvm.intr.memset"(%[[VAL_7]], %[[VAL_2]], %[[VAL_5]]) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

llvm.func @raise_nd_range_memcpy(%globalSize: !llvm.ptr, %localSize: !llvm.ptr) {
  %c0 = llvm.mlir.constant (0 : i8) : i8
  %c1 = llvm.mlir.constant (1 : i32) : i32
  %len = llvm.mlir.constant (24 : i64) : i64
  %nd = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>)> : (i32) -> !llvm.ptr
  "llvm.intr.memcpy"(%nd, %globalSize, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  %ls_ref = llvm.getelementptr inbounds %nd[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>)>
  "llvm.intr.memcpy"(%ls_ref, %localSize, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  %off_ref = llvm.getelementptr inbounds %nd[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>)>
  "llvm.intr.memset"(%off_ref, %c0, %len) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  llvm.return
}

// -----

// COM: Check we can raise nd_range constructor using memcpy and passing an offset

// CHECK-LABEL:   llvm.func @raise_nd_range_memcpy_offset(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr, %[[VAL_2:.*]]: !llvm.ptr) {
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_4]], %[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {type = !sycl_nd_range_3_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

llvm.func @raise_nd_range_memcpy_offset(%globalSize: !llvm.ptr, %localSize: !llvm.ptr, %offset: !llvm.ptr) {
  %c1 = llvm.mlir.constant (1 : i32) : i32
  %len = llvm.mlir.constant (24 : i64) : i64
  %nd = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>)> : (i32) -> !llvm.ptr
  "llvm.intr.memcpy"(%nd, %globalSize, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  %ls_ref = llvm.getelementptr inbounds %nd[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>)>
  "llvm.intr.memcpy"(%ls_ref, %localSize, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  %off_ref = llvm.getelementptr inbounds %nd[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<3 x i64>)>)>)>
  "llvm.intr.memcpy"(%off_ref, %offset, %len) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  llvm.return
}

// -----

// COM: Check we can raise nd_range constructor using stores

// CHECK-LABEL:   llvm.func @raise_nd_range_store(
// CHECK-SAME:                                    %[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64) {
// CHECK-DAG:       %[[VAL_4:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK-DAG:       %[[VAL_6:.*]] = llvm.mlir.constant(24 : i64) : i64
// CHECK-DAG:       %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.alloca %[[VAL_7]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_9]], %[[VAL_0]], %[[VAL_1]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.alloca %[[VAL_7]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_10]], %[[VAL_2]], %[[VAL_3]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.alloca %[[VAL_7]] x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_11]], %[[VAL_9]], %[[VAL_10]]) {type = !sycl_nd_range_2_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.getelementptr inbounds %[[VAL_11]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)>
// CHECK-NEXT:      "llvm.intr.memset"(%[[VAL_12]], %[[VAL_4]], %[[VAL_6]]) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

llvm.func @raise_nd_range_store(%g0: i64, %g1: i64, %l0: i64, %l1: i64) {
  %c0 = llvm.mlir.constant (0 : i8) : i8
  %len = llvm.mlir.constant (24 : i64) : i64
  %c1 = llvm.mlir.constant (1 : i32) : i32
  %nd = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)> : (i32) -> !llvm.ptr
  llvm.store %g0, %nd : i64, !llvm.ptr
  %g1_ref = llvm.getelementptr inbounds %nd[0, 0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)>
  llvm.store %g1, %g1_ref : i64, !llvm.ptr
  %l0_ref = llvm.getelementptr inbounds %nd[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)>
  llvm.store %l0, %l0_ref : i64, !llvm.ptr
  %l1_ref = llvm.getelementptr inbounds %nd[0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)>
  llvm.store %l1, %l1_ref : i64, !llvm.ptr
  %off_ref = llvm.getelementptr inbounds %nd[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)>
  "llvm.intr.memset"(%off_ref, %c0, %len) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  llvm.return
}

// -----

// COM: Check we can raise nd_range constructor using stores and passing an offset

// CHECK-LABEL:   llvm.func @raise_nd_range_store_offset(
// CHECK-SAME:                                           %[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: i64, %[[VAL_5:.*]]: i64) {
// CHECK-DAG:       %[[VAL_6:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_8]], %[[VAL_0]], %[[VAL_1]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_9]], %[[VAL_2]], %[[VAL_3]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_10]], %[[VAL_4]], %[[VAL_5]]) {type = !sycl_id_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_11]], %[[VAL_8]], %[[VAL_9]], %[[VAL_10]]) {type = !sycl_nd_range_2_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

llvm.func @raise_nd_range_store_offset(%g0: i64, %g1: i64, %l0: i64, %l1: i64, %off0: i64, %off1: i64) {
  %c1 = llvm.mlir.constant (1 : i32) : i32
  %nd = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)> : (i32) -> !llvm.ptr
  llvm.store %g0, %nd : i64, !llvm.ptr
  %g1_ref = llvm.getelementptr inbounds %nd[0, 0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)>
  llvm.store %g1, %g1_ref : i64, !llvm.ptr
  %l0_ref = llvm.getelementptr inbounds %nd[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)>
  llvm.store %l0, %l0_ref : i64, !llvm.ptr
  %l1_ref = llvm.getelementptr inbounds %nd[0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)>
  llvm.store %l1, %l1_ref : i64, !llvm.ptr
  %off0_ref = llvm.getelementptr inbounds %nd[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)>
  llvm.store %off0, %off0_ref : i64, !llvm.ptr
  %off1_ref = llvm.getelementptr inbounds %nd[0, 2, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)>
  llvm.store %off1, %off1_ref : i64, !llvm.ptr
  llvm.return
}

// -----

// COM: Check we can raise nd_range constructor using stores and GEPs with i8 element type

// CHECK-LABEL:   llvm.func @raise_nd_range_i8_gep(
// CHECK-SAME:                                     %[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: i64, %[[VAL_5:.*]]: i64) {
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_7]], %[[VAL_0]], %[[VAL_1]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_8]], %[[VAL_2]], %[[VAL_3]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_9]], %[[VAL_4]], %[[VAL_5]]) {type = !sycl_id_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_10]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]]) {type = !sycl_nd_range_2_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
llvm.func @raise_nd_range_i8_gep(%g0: i64, %g1: i64, %l0: i64, %l1: i64, %off0: i64, %off1: i64) {
  %c1 = llvm.mlir.constant (1 : i32) : i32
  %nd = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)> : (i32) -> !llvm.ptr
  llvm.store %g0, %nd : i64, !llvm.ptr
  %g1_ref = llvm.getelementptr inbounds %nd[8] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %g1, %g1_ref : i64, !llvm.ptr
  %l0_ref = llvm.getelementptr inbounds %nd[16] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %l0, %l0_ref : i64, !llvm.ptr
  %l1_ref = llvm.getelementptr inbounds %nd[24] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %l1, %l1_ref : i64, !llvm.ptr
  %off0_ref = llvm.getelementptr inbounds %nd[32] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %off0, %off0_ref : i64, !llvm.ptr
  %off1_ref = llvm.getelementptr inbounds %nd[40] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %off1, %off1_ref : i64, !llvm.ptr
  llvm.return
}
