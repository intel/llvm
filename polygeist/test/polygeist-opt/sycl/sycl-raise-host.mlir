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


// -----

llvm.func @__gxx_personality_v0(...) -> i32

// CHECK-LABEL: !sycl_accessor_1_21llvm2Evoid_rw_gb = !sycl.accessor<[1, !llvm.void, read_write, global_buffer], ()>

// CHECK-LABEL:   llvm.func @raise_accessor() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           sycl.host.constructor(%[[VAL_2]], %[[VAL_1]], %[[VAL_3]], %[[VAL_4]]) {type = !sycl_accessor_1_21llvm2Evoid_rw_gb} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// CHECK:           llvm.return %[[VAL_1]] : !llvm.ptr
// CHECK:         }
llvm.func @raise_accessor() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %0 = arith.constant 1 : i32
  %1 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %4 = llvm.alloca %0 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr

  llvm.invoke @_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE1ENS0_3ext6oneapi22accessor_property_listIJEEEEC2IiLi1ENS0_6detail17aligned_allocatorIiEEvEERNS0_6bufferIT_XT0_ET1_NSt9enable_ifIXaagtT0_Li0EleT0_Li3EEvE4typeEEERKNS0_13property_listENSC_13code_locationE(%2, %1, %3, %4) to ^bb1 unwind ^bb0 : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()

^bb0:
  %lp = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %lp : !llvm.struct<(ptr, i32)>
^bb1:
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
