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

// CHECK-LABEL: !sycl_accessor_1_21llvm2Evoid_rw_dev = !sycl.accessor<[1, !llvm.void, read_write, device], (!llvm.void)>
// CHECK:       !sycl_id_1_ = !sycl.id<[1], (!llvm.void)>
// CHECK:       !sycl_range_1_ = !sycl.range<[1], (!llvm.void)>
// CHECK:       !sycl_accessor_1_21llvm2Evoid_w_dev = !sycl.accessor<[1, !llvm.void, write, device], (!sycl_range_1_, !sycl_id_1_)>

// CHECK-LABEL:   llvm.func @raise_accessor() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 128 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 64 : i64
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_6:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_7:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::accessor.5", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           sycl.host.constructor(%[[VAL_4]], %[[VAL_3]], %[[VAL_5]], %[[VAL_6]]) {type = !sycl_accessor_1_21llvm2Evoid_rw_dev} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// CHECK:           sycl.host.constructor(%[[VAL_7]], %[[VAL_3]], %[[VAL_1]], %[[VAL_2]], %[[VAL_5]], %[[VAL_6]]) {type = !sycl_accessor_1_21llvm2Evoid_w_dev} : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr) -> ()
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

llvm.func @__gxx_personality_v0(...) -> i32

// CHECK-LABEL:   !sycl_local_accessor_1_21llvm2Evoid = !sycl.local_accessor<[1, !llvm.void], (!sycl_range_1_)>

// CHECK-LABEL:   llvm.func @raise_local_accessor(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::range", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           sycl.host.constructor(%[[VAL_2]], %[[VAL_3]], %[[VAL_0]], %[[VAL_4]]) {type = !sycl_local_accessor_1_21llvm2Evoid} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// CHECK:           llvm.return %[[VAL_2]] : !llvm.ptr
// CHECK:         }

llvm.func @raise_local_accessor(%handler: !llvm.ptr) -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %0 = arith.constant 1 : i32
  %acc = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %range = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::range", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %cl = llvm.alloca %0 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr

  llvm.invoke @_ZN4sycl3_V119local_accessor_baseINS0_3vecIdLi4EEELi1ELNS0_6access4modeE1026ELNS4_11placeholderE0EEC2ILi1EvEENS0_5rangeILi1EEERNS0_7handlerENS0_6detail13code_locationE(%acc, %range, %handler, %cl) to ^bb1 unwind ^bb0 : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()

^bb0:
  %lp = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %lp : !llvm.struct<(ptr, i32)>
^bb1:
  llvm.return %acc : !llvm.ptr
}

// -----

llvm.func @__gxx_personality_v0(...) -> i32
llvm.func @_ZN4sycl3_V16bufferIfLi1ENS0_6detail17aligned_allocatorIfEEvE10get_accessILNS0_6access4modeE1024ELNS7_6targetE2014EEENS0_8accessorIfLi1EXT_EXT0_ELNS7_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEERNS0_7handlerENS2_13code_locationE(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()

// CHECK-LABEL: !sycl_accessor_1_21llvm2Evoid_r_dev = !sycl.accessor<[1, !llvm.void, read, device], (!llvm.void)>
// CHECK:       !sycl_accessor_1_21llvm2Evoid_w_dev = !sycl.accessor<[1, !llvm.void, write, device], (!llvm.void)>
// CHECK:       !sycl_id_1_ = !sycl.id<[1], (!llvm.void)>
// CHECK:       !sycl_range_1_ = !sycl.range<[1], (!llvm.void)>
// CHECK:       !sycl_accessor_1_21llvm2Evoid_rw_dev = !sycl.accessor<[1, !llvm.void, read_write, device], (!sycl_range_1_, !sycl_id_1_)>

// CHECK-LABEL:   llvm.func @raise_get_access() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 128 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 64 : i64
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_6:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_7:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_8:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_9:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_10:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// COM: Read-only
// CHECK:           sycl.host.constructor(%[[VAL_6]], %[[VAL_3]], %[[VAL_9]], %[[VAL_10]]) {type = !sycl_accessor_1_21llvm2Evoid_r_dev} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// COM: Read-write with range and offset
// CHECK:           sycl.host.constructor(%[[VAL_7]], %[[VAL_4]], %[[VAL_1]], %[[VAL_2]], %[[VAL_9]], %[[VAL_10]]) {type = !sycl_accessor_1_21llvm2Evoid_rw_dev} : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// COM: Write-only
// CHECK:           sycl.host.constructor(%[[VAL_8]], %[[VAL_5]], %[[VAL_9]], %[[VAL_10]]) {type = !sycl_accessor_1_21llvm2Evoid_w_dev} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:
// CHECK:           llvm.return %[[VAL_3]] : !llvm.ptr
// CHECK:         }
llvm.func @raise_get_access() -> !llvm.ptr attributes {personality = @__gxx_personality_v0} {
  %0 = arith.constant 1 : i32
  %range = arith.constant 128 : i64
  %offset = arith.constant 64 : i64
  
  %buf_1 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %buf_2 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %buf_3 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::buffer", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  
  %acc_r = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc_rw = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %acc_w = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::accessor", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  
  %pl = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::property_list", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %cl = llvm.alloca %0 x !llvm.struct<"struct.sycl::_V1::detail::code_location", ()> {alignment = 8 : i64} : (i32) -> !llvm.ptr

  // COM: Read-only
  llvm.call @_ZN4sycl3_V16bufferIfLi1ENS0_6detail17aligned_allocatorIfEEvE10get_accessILNS0_6access4modeE1024ELNS7_6targetE2014EEENS0_8accessorIfLi1EXT_EXT0_ELNS7_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEERNS0_7handlerENS2_13code_locationE(%acc_r, %buf_1, %pl, %cl) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()

  // COM: Read-write, with range and offset
  llvm.invoke @_ZN4sycl3_V16bufferIfLi1ENS0_6detail17aligned_allocatorIfEEvE10get_accessILNS0_6access4modeE1026ELNS7_6targetE2014EEENS0_8accessorIfLi1EXT_EXT0_ELNS7_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEERNS0_7handlerENS0_5rangeILi1EEENS0_2idILi1EEENS2_13code_locationE(%acc_rw, %buf_2, %range, %offset, %pl, %cl) to ^bb1 unwind ^bb0 : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, !llvm.ptr) -> ()
^bb0:
  %lp = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %lp : !llvm.struct<(ptr, i32)>

^bb1:
  // COM: Write-only
  llvm.invoke @_ZN4sycl3_V16bufferIfLi1ENS0_6detail17aligned_allocatorIfEEvE10get_accessILNS0_6access4modeE1025ELNS7_6targetE2014EEENS0_8accessorIfLi1EXT_EXT0_ELNS7_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEERNS0_7handlerENS2_13code_locationE(%acc_w, %buf_3, %pl, %cl) to ^bb3 unwind ^bb2 : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
^bb2:
  %lp1 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.resume %lp1 : !llvm.struct<(ptr, i32)>

^bb3:
  llvm.return %buf_1 : !llvm.ptr
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
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len1 = llvm.mlir.constant(8 : i64) : i64
  %len2 = llvm.mlir.constant(16 : i64) : i64
  %len3 = llvm.mlir.constant(24 : i64) : i64
  %id1 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %id2 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %id3 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id.5", (struct<"class.sycl::_V1::detail::array.7", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  "llvm.intr.memcpy"(%id1, %other1, %len1) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  "llvm.intr.memcpy"(%id2, %other2, %len2) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  "llvm.intr.memcpy"(%id3, %other3, %len3) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
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
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %len1 = llvm.mlir.constant(8 : i64) : i64
  %len2 = llvm.mlir.constant(16 : i64) : i64
  %len3 = llvm.mlir.constant(24 : i64) : i64
  %range1 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %range2 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %range3 = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range.5", (struct<"class.sycl::_V1::detail::array.7", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  "llvm.intr.memcpy"(%range1, %other1, %len1) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  "llvm.intr.memcpy"(%range2, %other2, %len2) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  "llvm.intr.memcpy"(%range3, %other3, %len3) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
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
  %nullptr = llvm.mlir.zero : !llvm.ptr
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
  %nullptr = llvm.mlir.zero : !llvm.ptr
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
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.alloca %[[VAL_7]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.alloca %[[VAL_7]] x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_9]], %[[VAL_0]], %[[VAL_1]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_10]], %[[VAL_2]], %[[VAL_3]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
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
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_8]], %[[VAL_0]], %[[VAL_1]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_9]], %[[VAL_2]], %[[VAL_3]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_10]], %[[VAL_4]], %[[VAL_5]]) {type = !sycl_id_2_} : (!llvm.ptr, i64, i64) -> ()
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
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_7]], %[[VAL_0]], %[[VAL_1]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_8]], %[[VAL_2]], %[[VAL_3]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_9]], %[[VAL_4]], %[[VAL_5]]) {type = !sycl_id_2_} : (!llvm.ptr, i64, i64) -> ()
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

// -----

gpu.module @device_functions {
  gpu.func @foo() kernel {
    gpu.return
  }
}

llvm.mlir.global private unnamed_addr constant @kernel_str("kernel\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}

!lambda_class = !llvm.struct<"class.lambda", (i16, i32, !llvm.struct<"class.sycl::_V1::accessor", (ptr)>, !llvm.struct<"class.sycl::_V1::vec", (array<16 x i16>)>, f32, f32, array<5 x f32>, array<5 x f32>, !llvm.struct<"class.sycl::_V1::accessor", (ptr)>)>
!sycl_accessor_1_21llvm2Evoid_rw_dev = !sycl.accessor<[1, !llvm.void, read_write, device], (!llvm.void)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_local_accessor_1_21llvm2Evoid = !sycl.local_accessor<[1, !llvm.void], (!sycl_range_1_)>

llvm.func @_ZN5DummyD2Ev(%arg0: !llvm.ptr)
llvm.func @_ZN4sycl3_V17handler6unpackEv(%arg0: !llvm.ptr)

// COM: check that we correctly identify captured accessors, scalars, 
//      arrays and structs

// CHECK-LABEL:   llvm.mlir.global private constant @constant_array_0(dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<5xf32>) {addr_space = 0 : i32} : !llvm.array<5 x f32>

// CHECK-LABEL:   llvm.func @raise_set_captured(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !llvm.ptr) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1.000000e+01 : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant 1.100000e+01 : f32
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(6.000000e+00 : f32) : f32
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(5.000000e+00 : f32) : f32
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(dense<[3.000000e+00, 4.000000e+00]> : vector<2xf32>) : vector<2xf32>
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(dense<[1.000000e+00, 2.000000e+00]> : vector<2xf32>) : vector<2xf32>
// CHECK:           %[[VAL_7:.*]] = arith.constant dense<[1.000000e+01, 1.100000e+01]> : vector<2xf32>
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.constant(123 : i32) : i32
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(123 : i16) : i16
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           sycl.host.handler.set_kernel %[[VAL_0]] -> @device_functions::@foo : !llvm.ptr
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.undef : vector<2xf32>
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.undef : vector<2xf32>
// CHECK:           %[[VAL_14:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_15:.*]] = llvm.alloca %[[VAL_11]] x !llvm.struct<"class.sycl::_V1::accessor", (ptr)> : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_16:.*]] = llvm.alloca %[[VAL_11]] x !llvm.struct<"class.sycl::_V1::accessor", (ptr)> : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_17:.*]] = llvm.alloca %[[VAL_11]] x !llvm.struct<"class.sycl::_V1::vec", (array<16 x i16>)> : (i32) -> !llvm.ptr
// CHECK:           sycl.host.constructor(%[[VAL_15]], %[[VAL_14]], %[[VAL_14]], %[[VAL_14]]) {type = !sycl_accessor_1_21llvm2Evoid_rw_dev} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           sycl.host.constructor(%[[VAL_16]], %[[VAL_14]], %[[VAL_14]], %[[VAL_14]]) {type = !sycl_local_accessor_1_21llvm2Evoid} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           %[[VAL_18:.*]] = llvm.alloca %[[VAL_11]] x !llvm.struct<"class.lambda", (i16, i32, struct<"class.sycl::_V1::accessor", (ptr)>, struct<"class.sycl::_V1::vec", (array<16 x i16>)>, f32, f32, array<5 x f32>, array<5 x f32>, struct<"class.sycl::_V1::accessor", (ptr)>)> : (i32) -> !llvm.ptr
// CHECK:           llvm.store %[[VAL_9]], %[[VAL_18]] : i16, !llvm.ptr
// CHECK:           sycl.host.set_captured %[[VAL_18]][0] = %[[VAL_9]] : !llvm.ptr, i16
// CHECK:           %[[VAL_19:.*]] = llvm.getelementptr %[[VAL_18]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.lambda", (i16, i32, struct<"class.sycl::_V1::accessor", (ptr)>, struct<"class.sycl::_V1::vec", (array<16 x i16>)>, f32, f32, array<5 x f32>, array<5 x f32>, struct<"class.sycl::_V1::accessor", (ptr)>)>
// CHECK:           llvm.store %[[VAL_8]], %[[VAL_19]] : i32, !llvm.ptr
// CHECK:           sycl.host.set_captured %[[VAL_18]][1] = %[[VAL_8]] : !llvm.ptr, i32
// CHECK:           %[[VAL_20:.*]] = llvm.getelementptr %[[VAL_18]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.lambda", (i16, i32, struct<"class.sycl::_V1::accessor", (ptr)>, struct<"class.sycl::_V1::vec", (array<16 x i16>)>, f32, f32, array<5 x f32>, array<5 x f32>, struct<"class.sycl::_V1::accessor", (ptr)>)>
// CHECK:           %[[VAL_21:.*]] = llvm.load %[[VAL_15]] : !llvm.ptr -> !llvm.ptr
// CHECK:           llvm.store %[[VAL_21]], %[[VAL_20]] : !llvm.ptr, !llvm.ptr
// CHECK:           sycl.host.set_captured %[[VAL_18]][2] = %[[VAL_15]] : !llvm.ptr, !llvm.ptr (!sycl_accessor_1_21llvm2Evoid_rw_dev)
// CHECK:           %[[VAL_22:.*]] = llvm.getelementptr %[[VAL_18]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.lambda", (i16, i32, struct<"class.sycl::_V1::accessor", (ptr)>, struct<"class.sycl::_V1::vec", (array<16 x i16>)>, f32, f32, array<5 x f32>, array<5 x f32>, struct<"class.sycl::_V1::accessor", (ptr)>)>
// CHECK:           "llvm.intr.memcpy"(%[[VAL_22]], %[[VAL_17]], %[[VAL_10]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
// CHECK:           sycl.host.set_captured %[[VAL_18]][3] = %[[VAL_17]] : !llvm.ptr, !llvm.ptr
// CHECK:           %[[VAL_23:.*]] = llvm.getelementptr %[[VAL_18]][0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.lambda", (i16, i32, struct<"class.sycl::_V1::accessor", (ptr)>, struct<"class.sycl::_V1::vec", (array<16 x i16>)>, f32, f32, array<5 x f32>, array<5 x f32>, struct<"class.sycl::_V1::accessor", (ptr)>)>
// CHECK:           llvm.store %[[VAL_7]], %[[VAL_23]] : vector<2xf32>, !llvm.ptr
// CHECK:           sycl.host.set_captured %[[VAL_18]][4] = %[[VAL_1]] : !llvm.ptr, f32
// CHECK:           sycl.host.set_captured %[[VAL_18]][5] = %[[VAL_2]] : !llvm.ptr, f32
// CHECK:           %[[VAL_24:.*]] = llvm.getelementptr %[[VAL_18]][0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.lambda", (i16, i32, struct<"class.sycl::_V1::accessor", (ptr)>, struct<"class.sycl::_V1::vec", (array<16 x i16>)>, f32, f32, array<5 x f32>, array<5 x f32>, struct<"class.sycl::_V1::accessor", (ptr)>)>
// CHECK:           llvm.store %[[VAL_6]], %[[VAL_24]] : vector<2xf32>, !llvm.ptr
// CHECK:           %[[VAL_25:.*]] = llvm.getelementptr %[[VAL_18]][0, 6, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.lambda", (i16, i32, struct<"class.sycl::_V1::accessor", (ptr)>, struct<"class.sycl::_V1::vec", (array<16 x i16>)>, f32, f32, array<5 x f32>, array<5 x f32>, struct<"class.sycl::_V1::accessor", (ptr)>)>
// CHECK:           llvm.store %[[VAL_5]], %[[VAL_25]] : vector<2xf32>, !llvm.ptr
// CHECK:           %[[VAL_26:.*]] = llvm.getelementptr %[[VAL_18]][0, 6, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.lambda", (i16, i32, struct<"class.sycl::_V1::accessor", (ptr)>, struct<"class.sycl::_V1::vec", (array<16 x i16>)>, f32, f32, array<5 x f32>, array<5 x f32>, struct<"class.sycl::_V1::accessor", (ptr)>)>
// CHECK:           llvm.store %[[VAL_4]], %[[VAL_26]] : f32, !llvm.ptr
// CHECK:           %[[VAL_27:.*]] = llvm.mlir.addressof @constant_array_0 : !llvm.ptr<array<5 x f32>>
// CHECK:           sycl.host.set_captured %[[VAL_18]][6] = %[[VAL_27]] : !llvm.ptr, !llvm.ptr<array<5 x f32>>
// CHECK:           %[[VAL_28:.*]] = llvm.getelementptr %[[VAL_18]][0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.lambda", (i16, i32, struct<"class.sycl::_V1::accessor", (ptr)>, struct<"class.sycl::_V1::vec", (array<16 x i16>)>, f32, f32, array<5 x f32>, array<5 x f32>, struct<"class.sycl::_V1::accessor", (ptr)>)>
// CHECK:           llvm.store %[[VAL_12]], %[[VAL_28]] : vector<2xf32>, !llvm.ptr
// CHECK:           %[[VAL_29:.*]] = llvm.mlir.undef : vector<5xf32>
// CHECK:           %[[VAL_30:.*]] = vector.insert_strided_slice %[[VAL_12]], %[[VAL_29]] {offsets = [0], strides = [1]} : vector<2xf32> into vector<5xf32>
// CHECK:           %[[VAL_31:.*]] = llvm.getelementptr %[[VAL_18]][0, 7, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.lambda", (i16, i32, struct<"class.sycl::_V1::accessor", (ptr)>, struct<"class.sycl::_V1::vec", (array<16 x i16>)>, f32, f32, array<5 x f32>, array<5 x f32>, struct<"class.sycl::_V1::accessor", (ptr)>)>
// CHECK:           llvm.store %[[VAL_13]], %[[VAL_31]] : vector<2xf32>, !llvm.ptr
// CHECK:           %[[VAL_32:.*]] = vector.insert_strided_slice %[[VAL_13]], %[[VAL_30]] {offsets = [2], strides = [1]} : vector<2xf32> into vector<5xf32>
// CHECK:           %[[VAL_33:.*]] = llvm.getelementptr %[[VAL_18]][0, 7, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.lambda", (i16, i32, struct<"class.sycl::_V1::accessor", (ptr)>, struct<"class.sycl::_V1::vec", (array<16 x i16>)>, f32, f32, array<5 x f32>, array<5 x f32>, struct<"class.sycl::_V1::accessor", (ptr)>)>
// CHECK:           llvm.store %[[VAL_3]], %[[VAL_33]] : f32, !llvm.ptr
// CHECK:           %[[VAL_34:.*]] = vector.insert %[[VAL_3]], %[[VAL_32]] [4] : f32 into vector<5xf32>
// CHECK:           sycl.host.set_captured %[[VAL_18]][7] = %[[VAL_34]] : !llvm.ptr, vector<5xf32>
// CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[VAL_18]][0, 8] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.lambda", (i16, i32, struct<"class.sycl::_V1::accessor", (ptr)>, struct<"class.sycl::_V1::vec", (array<16 x i16>)>, f32, f32, array<5 x f32>, array<5 x f32>, struct<"class.sycl::_V1::accessor", (ptr)>)>
// CHECK:           %[[VAL_36:.*]] = llvm.load %[[VAL_16]] : !llvm.ptr -> !llvm.ptr
// CHECK:           llvm.store %[[VAL_36]], %[[VAL_35]] : !llvm.ptr, !llvm.ptr
// CHECK:           sycl.host.set_captured %[[VAL_18]][8] = %[[VAL_16]] : !llvm.ptr, !llvm.ptr (!sycl_local_accessor_1_21llvm2Evoid)
// CHECK:           %[[VAL_37:.*]] = llvm.alloca %[[VAL_11]] x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK:           llvm.store %[[VAL_18]], %[[VAL_37]] : !llvm.ptr, !llvm.ptr
// CHECK:           llvm.call @_ZN5DummyD2Ev(%[[VAL_18]]) : (!llvm.ptr) -> ()
// CHECK:           llvm.call @_ZN4sycl3_V17handler6unpackEv(%[[VAL_18]]) : (!llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }

llvm.func @raise_set_captured(%handler: !llvm.ptr) {
  // COM: ensure this function is detected as a handler
  sycl.host.handler.set_kernel %handler -> @device_functions::@foo : !llvm.ptr
  
  %c0 = llvm.mlir.constant (0 : i32) : i32
  %c1 = llvm.mlir.constant (1 : i32) : i32
  %c32 = llvm.mlir.constant (32 : i32) : i32
  %c123_16 = llvm.mlir.constant (123 : i16) : i16
  %c123_32 = llvm.mlir.constant (123 : i32) : i32
  %vec_lit = arith.constant dense<[10.0, 11.0]> : vector<2xf32>
  %vec_c1 = llvm.mlir.constant(dense<[1.0, 2.0]> : vector<2 x f32>) : vector<2 x f32>
  %vec_c2 = llvm.mlir.constant(dense<[3.0, 4.0]> : vector<2 x f32>) : vector<2 x f32>
  %c5 = llvm.mlir.constant(5.0 : f32) : f32
  %vec_u1 = llvm.mlir.undef : vector<2 x f32>
  %vec_u2 = llvm.mlir.undef : vector<2 x f32>
  %c6 = llvm.mlir.constant(6.0 : f32) : f32
  %nullptr = llvm.mlir.zero : !llvm.ptr
  %accessor = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::accessor", (ptr)> : (i32) -> !llvm.ptr
  %local_accessor = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::accessor", (ptr)> : (i32) -> !llvm.ptr
  %vector = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::vec", (array<16 x i16>)> : (i32) -> !llvm.ptr
  
  // COM: ensure the accessors are actually detected as such
  sycl.host.constructor(%accessor, %nullptr, %nullptr, %nullptr) {type = !sycl_accessor_1_21llvm2Evoid_rw_dev} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  sycl.host.constructor(%local_accessor, %nullptr, %nullptr, %nullptr) {type = !sycl_local_accessor_1_21llvm2Evoid} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()

  // COM: struct containing the captured values
  %lambda_obj = llvm.alloca %c1 x !lambda_class : (i32) -> !llvm.ptr
  
  // COM: the first capture is a store to the lambda object
  llvm.store %c123_16, %lambda_obj : i16, !llvm.ptr

  // COM: All other captures are 2-index GEPs to the lambda object
  %gep1 = llvm.getelementptr %lambda_obj[0, 1] : (!llvm.ptr) -> !llvm.ptr, !lambda_class
  llvm.store %c123_32, %gep1 : i32, !llvm.ptr
  
  // COM: Special handling for accessors
  %gep2 = llvm.getelementptr %lambda_obj[0, 2] : (!llvm.ptr) -> !llvm.ptr, !lambda_class
  %first_member = llvm.load %accessor : !llvm.ptr -> !llvm.ptr
  llvm.store %first_member, %gep2 : !llvm.ptr, !llvm.ptr

  // COM: Non-scalar values are memcpy'ed
  %gep3 = llvm.getelementptr %lambda_obj[0, 3] : (!llvm.ptr) -> !llvm.ptr, !lambda_class
  "llvm.intr.memcpy"(%gep3, %vector, %c32) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()

  // COM: Frontend sometimes groups scalars into vectors
  %gep4 = llvm.getelementptr %lambda_obj[0, 4] : (!llvm.ptr) -> !llvm.ptr, !lambda_class
  llvm.store %vec_lit, %gep4 : vector<2xf32>, !llvm.ptr

  // COM: Capture a constant array
  %gep6_0 = llvm.getelementptr %lambda_obj[0, 6] : (!llvm.ptr) -> !llvm.ptr, !lambda_class
  llvm.store %vec_c1, %gep6_0 : vector<2xf32>, !llvm.ptr
  %gep6_2 = llvm.getelementptr %lambda_obj[0, 6, 2] : (!llvm.ptr) -> !llvm.ptr, !lambda_class
  llvm.store %vec_c2, %gep6_2 : vector<2xf32>, !llvm.ptr
  %gep6_4 = llvm.getelementptr %lambda_obj[0, 6, 4] : (!llvm.ptr) -> !llvm.ptr, !lambda_class
  llvm.store %c5, %gep6_4 : f32, !llvm.ptr

  // Capture a non-constant array
  %gep7_0 = llvm.getelementptr %lambda_obj[0, 7] : (!llvm.ptr) -> !llvm.ptr, !lambda_class
  llvm.store %vec_u1, %gep7_0 : vector<2xf32>, !llvm.ptr
  %gep7_2 = llvm.getelementptr %lambda_obj[0, 7, 2] : (!llvm.ptr) -> !llvm.ptr, !lambda_class
  llvm.store %vec_u2, %gep7_2 : vector<2xf32>, !llvm.ptr
  %gep7_4 = llvm.getelementptr %lambda_obj[0, 7, 4] : (!llvm.ptr) -> !llvm.ptr, !lambda_class
  llvm.store %c6, %gep7_4 : f32, !llvm.ptr

  // COM: Special handling for local accessor
  %gep8 = llvm.getelementptr %lambda_obj[0, 8] : (!llvm.ptr) -> !llvm.ptr, !lambda_class
  %first_member_2 = llvm.load %local_accessor : !llvm.ptr -> !llvm.ptr
  llvm.store %first_member_2, %gep8 : !llvm.ptr, !llvm.ptr

  // COM: the annotation (indirectly) marks the struct as the lambda object
  %annotated_ptr = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
  llvm.store %lambda_obj, %annotated_ptr : !llvm.ptr, !llvm.ptr
  %kernel_str = llvm.mlir.addressof @kernel_str : !llvm.ptr
  "llvm.intr.var.annotation"(%annotated_ptr, %kernel_str, %kernel_str, %c0, %nullptr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
  
  // COM: mockup destruction of lambda object, should not interfere with the raising
  llvm.call @_ZN5DummyD2Ev(%lambda_obj) : (!llvm.ptr) -> ()

  // COM: mockup call to `sycl::handler::unpack`, should not interfere with the raising
  llvm.call @_ZN4sycl3_V17handler6unpackEv(%lambda_obj) : (!llvm.ptr) -> ()

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

// COM: Check we can raise sycl.host.handler.set_nd_range when there are several var annotations in the code.

// CHECK-LABEL:   llvm.func @raise_set_globalsize_offset(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                                           %[[VAL_1:.*]]: i1) {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(512 : i64) : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(100 : i64) : i64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      sycl.host.handler.set_kernel %[[VAL_0]] -> @device_functions::@foo : !llvm.ptr
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.alloca %[[VAL_4]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.alloca %[[VAL_4]] x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:      llvm.cond_br %[[VAL_1]], ^bb1, ^bb2
// CHECK-NEXT:    ^bb1:
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_5]], %[[VAL_2]]) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_6]], %[[VAL_3]]) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:      llvm.br ^bb3
// CHECK-NEXT:    ^bb2:
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_5]], %[[VAL_2]]) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_6]]) {type = !sycl_id_1_} : (!llvm.ptr) -> ()
// CHECK-NEXT:      sycl.host.handler.set_nd_range %[[VAL_0]] -> range %[[VAL_5]], offset %[[VAL_6]] : !llvm.ptr, !llvm.ptr, !llvm.ptr
// CHECK-NEXT:      llvm.br ^bb3
// CHECK-NEXT:    ^bb3:
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
llvm.func @raise_set_globalsize_offset(%handler: !llvm.ptr, %condition: i1) {
  %c0 = llvm.mlir.constant (0 : i32) : i32
  %c1 = llvm.mlir.constant (1 : i64) : i64
  sycl.host.handler.set_kernel %handler -> @device_functions::@foo : !llvm.ptr
  %c100 = llvm.mlir.constant (100 : i64) : i64
  %c512 = llvm.mlir.constant (512 : i64) : i64
  %nullptr = llvm.mlir.zero : !llvm.ptr
  %range = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  %offset = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  %rangeStr = llvm.mlir.addressof @range_str : !llvm.ptr
  %offsetStr = llvm.mlir.addressof @offset_str : !llvm.ptr
  llvm.cond_br %condition, ^b0, ^b1
^b0:
  sycl.host.constructor(%range, %c512) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%offset, %c100) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
  "llvm.intr.var.annotation"(%range, %rangeStr, %rangeStr, %c0, %nullptr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
  "llvm.intr.var.annotation"(%offset, %offsetStr, %offsetStr, %c0, %nullptr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
  llvm.br ^exit
^b1:
  sycl.host.constructor(%range, %c512) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
  sycl.host.constructor(%offset) {type = !sycl_id_1_} : (!llvm.ptr) -> ()
  "llvm.intr.var.annotation"(%range, %rangeStr, %rangeStr, %c0, %nullptr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
  "llvm.intr.var.annotation"(%offset, %offsetStr, %offsetStr, %c0, %nullptr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
  llvm.br ^exit
^exit:
  llvm.return
}

// -----

// COM: Check we do not violate dominance when raising nd_range constructor using stores

// CHECK-LABEL:   llvm.func @raise_nd_range_load_store_offset(
// CHECK-SAME:                                                %[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr, %[[VAL_2:.*]]: !llvm.ptr, %[[VAL_3:.*]]: !llvm.ptr, %[[VAL_4:.*]]: !llvm.ptr, %[[VAL_5:.*]]: !llvm.ptr) {
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_13:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_16:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.load %[[VAL_0]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i64
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_10]], %[[VAL_8]], %[[VAL_9]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_12:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> i64
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_13]], %[[VAL_11]], %[[VAL_12]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      %[[VAL_14:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[VAL_15:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> i64
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_16]], %[[VAL_14]], %[[VAL_15]]) {type = !sycl_id_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_7]], %[[VAL_10]], %[[VAL_13]], %[[VAL_16]]) {type = !sycl_nd_range_2_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }

llvm.func @raise_nd_range_load_store_offset(%g0: !llvm.ptr,
                                            %g1: !llvm.ptr,
                                            %l0: !llvm.ptr,
                                            %l1: !llvm.ptr,
                                            %off0: !llvm.ptr,
                                            %off1: !llvm.ptr) {
  %c1 = llvm.mlir.constant (1 : i32) : i32
  %nd = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)> : (i32) -> !llvm.ptr
  %g0_val = llvm.load %g0 : !llvm.ptr -> i64
  llvm.store %g0_val, %nd : i64, !llvm.ptr
  %g1_ref = llvm.getelementptr inbounds %nd[0, 0, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)>
  %g1_val = llvm.load %g1 : !llvm.ptr -> i64
  llvm.store %g1_val, %g1_ref : i64, !llvm.ptr
  %l0_ref = llvm.getelementptr inbounds %nd[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)>
  %l0_val = llvm.load %l0 : !llvm.ptr -> i64
  llvm.store %l0_val, %l0_ref : i64, !llvm.ptr
  %l1_ref = llvm.getelementptr inbounds %nd[0, 1, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)>
  %l1_val = llvm.load %l1 : !llvm.ptr -> i64
  llvm.store %l1_val, %l1_ref : i64, !llvm.ptr
  %off0_ref = llvm.getelementptr inbounds %nd[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)>
  %off0_val = llvm.load %off0 : !llvm.ptr -> i64
  llvm.store %off0_val, %off0_ref : i64, !llvm.ptr
  %off1_ref = llvm.getelementptr inbounds %nd[0, 2, 0, 0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"class.sycl::_V1::nd_range", (struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>, struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<2 x i64>)>)>)>
  %off1_val = llvm.load %off1 : !llvm.ptr -> i64
  llvm.store %off1_val, %off1_ref : i64, !llvm.ptr
  llvm.return
}

// -----

// COM: Check raising of set_nd_range, set_captured and set_kernel to schedule_kernel.
//      Note that the raising pattern is currently limited to `parallel_for`-style
//      launches with at least one argument.

gpu.module @device_functions {
  gpu.func @foo() kernel {
    gpu.return
  }
}

!lambda_class = !llvm.struct<"class.lambda", opaque>
!sycl_accessor_1_21llvm2Evoid_rw_dev = !sycl.accessor<[1, !llvm.void, read_write, device], (!llvm.void)>
!kernel_signatures_type = !llvm.array<7 x struct<"struct.sycl::_V1::detail::kernel_param_desc_t", (i32, i32, i32)>>

llvm.mlir.global internal constant @_ZN4sycl3_V16detailL17kernel_signaturesE() : !kernel_signatures_type {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %dummy = llvm.mlir.constant(-987654321 : i32) : i32
  %kind_accessor = llvm.mlir.constant(0 : i32) : i32
  %kind_std_layout = llvm.mlir.constant(1 : i32) : i32
  %kind_invalid = llvm.mlir.constant(15 : i32) : i32
  %undef = llvm.mlir.undef : !llvm.struct<"struct.sycl::_V1::detail::kernel_param_desc_t", (i32, i32, i32)>
  
  // COM: accessor arg
  %accessor_0 = llvm.insertvalue %kind_accessor, %undef[0] : !llvm.struct<"struct.sycl::_V1::detail::kernel_param_desc_t", (i32, i32, i32)> 
  %accessor_1 = llvm.insertvalue %zero, %accessor_0[1] : !llvm.struct<"struct.sycl::_V1::detail::kernel_param_desc_t", (i32, i32, i32)> 
  %accessor = llvm.insertvalue %zero, %accessor_1[2] : !llvm.struct<"struct.sycl::_V1::detail::kernel_param_desc_t", (i32, i32, i32)> 

  // COM: std_layout arg
  %std_layout_0 = llvm.insertvalue %kind_std_layout, %undef[0] : !llvm.struct<"struct.sycl::_V1::detail::kernel_param_desc_t", (i32, i32, i32)> 
  %std_layout_1 = llvm.insertvalue %zero, %std_layout_0[1] : !llvm.struct<"struct.sycl::_V1::detail::kernel_param_desc_t", (i32, i32, i32)> 
  %std_layout = llvm.insertvalue %zero, %std_layout_1[2] : !llvm.struct<"struct.sycl::_V1::detail::kernel_param_desc_t", (i32, i32, i32)> 

  // COM: sentinel
  %sentinel_0 = llvm.insertvalue %kind_invalid, %undef[0] : !llvm.struct<"struct.sycl::_V1::detail::kernel_param_desc_t", (i32, i32, i32)> 
  %sentinel_1 = llvm.insertvalue %dummy, %sentinel_0[1] : !llvm.struct<"struct.sycl::_V1::detail::kernel_param_desc_t", (i32, i32, i32)> 
  %sentinel = llvm.insertvalue %dummy, %sentinel_1[2] : !llvm.struct<"struct.sycl::_V1::detail::kernel_param_desc_t", (i32, i32, i32)> 
  
  // COM: mockup 3 kernel signatures here:
  //      - accessor
  //      - std_layout, accessor
  //      - std_layout, std_layout, std_layout
  //      - sentinel
  %0 = llvm.mlir.undef : !kernel_signatures_type
  %1 = llvm.insertvalue %accessor, %0[0] : !kernel_signatures_type
  %2 = llvm.insertvalue %std_layout, %1[1] : !kernel_signatures_type
  %3 = llvm.insertvalue %accessor, %2[2] : !kernel_signatures_type
  %4 = llvm.insertvalue %std_layout, %3[3] : !kernel_signatures_type
  %5 = llvm.insertvalue %std_layout, %4[4] : !kernel_signatures_type
  %6 = llvm.insertvalue %std_layout, %5[5] : !kernel_signatures_type
  %7 = llvm.insertvalue %sentinel, %6[6] : !kernel_signatures_type
  llvm.return %7 : !kernel_signatures_type
}

llvm.mlir.global private unnamed_addr constant @kernel_num_params_str("kernel_num_params\00")
llvm.mlir.global private unnamed_addr constant @kernel_param_desc_str("kernel_param_desc\00")

// CHECK-LABEL:   llvm.func @raise_schedule_parallel_for_range_accessor(
// CHECK-SAME:                                                          %[[VAL_0:.*]]: !llvm.ptr) {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"class.sycl::_V1::range", opaque> : (i32) -> !llvm.ptr
// CHECK:           sycl.host.handler.set_nd_range %[[VAL_0]] -> range %[[VAL_4]] : !llvm.ptr, !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"class.lambda", opaque> : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_6:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"class.sycl::_V1::buffer", opaque> : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_7:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"class.sycl::_V1::accessor", opaque> : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_8:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"class.sycl::_V1::property_list", opaque> : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_9:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"struct.sycl::_V1::detail::code_location", opaque> : (i32) -> !llvm.ptr
// CHECK:           sycl.host.constructor(%[[VAL_7]], %[[VAL_6]], %[[VAL_8]], %[[VAL_9]]) {type = !sycl_accessor_1_21llvm2Evoid_rw_dev} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           sycl.host.set_captured %[[VAL_5]][0] = %[[VAL_7]] : !llvm.ptr, !llvm.ptr (!sycl_accessor_1_21llvm2Evoid_rw_dev)
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.addressof @_ZN4sycl3_V16detailL17kernel_signaturesE : !llvm.ptr
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.addressof @kernel_num_params_str : !llvm.ptr
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.addressof @kernel_param_desc_str : !llvm.ptr
// CHECK:           %[[VAL_13:.*]] = llvm.alloca %[[VAL_2]] x i32 : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_14:.*]] = llvm.alloca %[[VAL_2]] x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK:           llvm.store %[[VAL_2]], %[[VAL_13]] : i32, !llvm.ptr
// CHECK:           llvm.store %[[VAL_10]], %[[VAL_14]] : !llvm.ptr, !llvm.ptr
// CHECK:           "llvm.intr.var.annotation"(%[[VAL_13]], %[[VAL_11]], %[[VAL_3]], %[[VAL_1]], %[[VAL_3]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
// CHECK:           "llvm.intr.var.annotation"(%[[VAL_14]], %[[VAL_12]], %[[VAL_3]], %[[VAL_1]], %[[VAL_3]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @device_functions::@foo[range %[[VAL_4]]](%[[VAL_7]]: !sycl_accessor_1_21llvm2Evoid_rw_dev) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }
llvm.func @raise_schedule_parallel_for_range_accessor(%handler: !llvm.ptr) {
  %c0 = llvm.mlir.constant (0 : i32) : i32
  %c1 = llvm.mlir.constant (1 : i32) : i32
  %nullptr = llvm.mlir.zero : !llvm.ptr

  %range = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range", opaque> : (i32) -> !llvm.ptr
  sycl.host.handler.set_nd_range %handler -> range %range : !llvm.ptr, !llvm.ptr

  %lambda = llvm.alloca %c1 x !lambda_class : (i32) -> !llvm.ptr
  %buffer = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::buffer", opaque> : (i32) -> !llvm.ptr
  %accessor = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::accessor", opaque> : (i32) -> !llvm.ptr
  %pl = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::property_list", opaque> : (i32) -> !llvm.ptr
  %cl = llvm.alloca %c1 x !llvm.struct<"struct.sycl::_V1::detail::code_location", opaque> : (i32) -> !llvm.ptr
  sycl.host.constructor(%accessor, %buffer, %pl, %cl) {type = !sycl_accessor_1_21llvm2Evoid_rw_dev} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> () 
  sycl.host.set_captured %lambda[0] = %accessor : !llvm.ptr, !llvm.ptr (!sycl_accessor_1_21llvm2Evoid_rw_dev)

  %kernel_signatures = llvm.mlir.addressof @_ZN4sycl3_V16detailL17kernel_signaturesE : !llvm.ptr
  %kernel_num_params_str = llvm.mlir.addressof @kernel_num_params_str : !llvm.ptr
  %kernel_param_desc_str = llvm.mlir.addressof @kernel_param_desc_str : !llvm.ptr
  %num_params = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %param_desc = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
  llvm.store %c1, %num_params : i32, !llvm.ptr
  llvm.store %kernel_signatures, %param_desc : !llvm.ptr, !llvm.ptr
  "llvm.intr.var.annotation"(%num_params, %kernel_num_params_str, %nullptr, %c0, %nullptr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
  "llvm.intr.var.annotation"(%param_desc, %kernel_param_desc_str, %nullptr, %c0, %nullptr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
  llvm.br ^bb1

^bb1:
  sycl.host.handler.set_kernel %handler -> @device_functions::@foo : !llvm.ptr

  llvm.return
}

// CHECK-LABEL:   llvm.func @raise_schedule_parallel_for_range_offset_std_layout_accessor(
// CHECK-SAME:                                                                            %[[VAL_0:.*]]: !llvm.ptr) {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"class.sycl::_V1::range", opaque> : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_6:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"class.sycl::_V1::id", opaque> : (i32) -> !llvm.ptr
// CHECK:           sycl.host.handler.set_nd_range %[[VAL_0]] -> range %[[VAL_5]], offset %[[VAL_6]] : !llvm.ptr, !llvm.ptr, !llvm.ptr
// CHECK:           %[[VAL_7:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"class.lambda", opaque> : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_8:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"class.sycl::_V1::buffer", opaque> : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_9:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"class.sycl::_V1::accessor", opaque> : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_10:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"class.sycl::_V1::property_list", opaque> : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_11:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"struct.sycl::_V1::detail::code_location", opaque> : (i32) -> !llvm.ptr
// CHECK:           sycl.host.constructor(%[[VAL_9]], %[[VAL_8]], %[[VAL_10]], %[[VAL_11]]) {type = !sycl_accessor_1_21llvm2Evoid_rw_dev} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           sycl.host.set_captured %[[VAL_7]][0] = %[[VAL_1]] : !llvm.ptr, i32
// CHECK:           sycl.host.set_captured %[[VAL_7]][1] = %[[VAL_9]] : !llvm.ptr, !llvm.ptr (!sycl_accessor_1_21llvm2Evoid_rw_dev)
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.addressof @_ZN4sycl3_V16detailL17kernel_signaturesE : !llvm.ptr
// CHECK:           %[[VAL_13:.*]] = llvm.getelementptr %[[VAL_12]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x struct<"struct.sycl::_V1::detail::kernel_param_desc_t", (i32, i32, i32)>>
// CHECK:           %[[VAL_14:.*]] = llvm.mlir.addressof @kernel_num_params_str : !llvm.ptr
// CHECK:           %[[VAL_15:.*]] = llvm.mlir.addressof @kernel_param_desc_str : !llvm.ptr
// CHECK:           %[[VAL_16:.*]] = llvm.alloca %[[VAL_2]] x i32 : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_17:.*]] = llvm.alloca %[[VAL_2]] x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK:           llvm.store %[[VAL_3]], %[[VAL_16]] : i32, !llvm.ptr
// CHECK:           llvm.store %[[VAL_13]], %[[VAL_17]] : !llvm.ptr, !llvm.ptr
// CHECK:           "llvm.intr.var.annotation"(%[[VAL_16]], %[[VAL_14]], %[[VAL_4]], %[[VAL_1]], %[[VAL_4]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
// CHECK:           "llvm.intr.var.annotation"(%[[VAL_17]], %[[VAL_15]], %[[VAL_4]], %[[VAL_1]], %[[VAL_4]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @device_functions::@foo[range %[[VAL_5]], offset %[[VAL_6]]](%[[VAL_1]], %[[VAL_9]]: !sycl_accessor_1_21llvm2Evoid_rw_dev) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }
llvm.func @raise_schedule_parallel_for_range_offset_std_layout_accessor(%handler: !llvm.ptr) {
  %c0 = llvm.mlir.constant (0 : i32) : i32
  %c1 = llvm.mlir.constant (1 : i32) : i32
  %c2 = llvm.mlir.constant (2 : i32) : i32
  %nullptr = llvm.mlir.zero : !llvm.ptr

  %range = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::range", opaque> : (i32) -> !llvm.ptr
  %offset = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::id", opaque> : (i32) -> !llvm.ptr
  sycl.host.handler.set_nd_range %handler -> range %range, offset %offset : !llvm.ptr, !llvm.ptr, !llvm.ptr

  %lambda = llvm.alloca %c1 x !lambda_class : (i32) -> !llvm.ptr
  %buffer = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::buffer", opaque> : (i32) -> !llvm.ptr
  %accessor = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::accessor", opaque> : (i32) -> !llvm.ptr
  %pl = llvm.alloca %c1 x !llvm.struct<"class.sycl::_V1::property_list", opaque> : (i32) -> !llvm.ptr
  %cl = llvm.alloca %c1 x !llvm.struct<"struct.sycl::_V1::detail::code_location", opaque> : (i32) -> !llvm.ptr
  sycl.host.constructor(%accessor, %buffer, %pl, %cl) {type = !sycl_accessor_1_21llvm2Evoid_rw_dev} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> () 
  sycl.host.set_captured %lambda[0] = %c0 : !llvm.ptr, i32
  sycl.host.set_captured %lambda[1] = %accessor : !llvm.ptr, !llvm.ptr (!sycl_accessor_1_21llvm2Evoid_rw_dev)

  %kernel_signatures = llvm.mlir.addressof @_ZN4sycl3_V16detailL17kernel_signaturesE : !llvm.ptr
  %sig_offset = llvm.getelementptr %kernel_signatures[0, 1] : (!llvm.ptr) -> !llvm.ptr, !kernel_signatures_type
  %kernel_num_params_str = llvm.mlir.addressof @kernel_num_params_str : !llvm.ptr
  %kernel_param_desc_str = llvm.mlir.addressof @kernel_param_desc_str : !llvm.ptr
  %num_params = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %param_desc = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
  llvm.store %c2, %num_params : i32, !llvm.ptr
  llvm.store %sig_offset, %param_desc : !llvm.ptr, !llvm.ptr
  "llvm.intr.var.annotation"(%num_params, %kernel_num_params_str, %nullptr, %c0, %nullptr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
  "llvm.intr.var.annotation"(%param_desc, %kernel_param_desc_str, %nullptr, %c0, %nullptr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
  llvm.br ^bb1

^bb1:
  sycl.host.handler.set_kernel %handler -> @device_functions::@foo : !llvm.ptr

  llvm.return
}

// CHECK-LABEL:   llvm.func @raise_schedule_parallel_for_ndrange_std_layout_x3(
// CHECK-SAME:                                                                 %[[VAL_0:.*]]: !llvm.ptr, %[[VAL_1:.*]]: !llvm.ptr, %[[VAL_2:.*]]: !llvm.ptr) {
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           sycl.host.handler.set_nd_range %[[VAL_0]] -> nd_range %[[VAL_1]] : !llvm.ptr, !llvm.ptr
// CHECK:           %[[VAL_7:.*]] = llvm.alloca %[[VAL_4]] x !llvm.struct<"class.lambda", opaque> : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_8:.*]] = llvm.alloca %[[VAL_4]] x !llvm.struct<"some_struct", opaque> : (i32) -> !llvm.ptr
// CHECK:           sycl.host.set_captured %[[VAL_7]][0] = %[[VAL_8]] : !llvm.ptr, !llvm.ptr
// CHECK:           sycl.host.set_captured %[[VAL_7]][1] = %[[VAL_2]] : !llvm.ptr, !llvm.ptr
// CHECK:           sycl.host.set_captured %[[VAL_7]][2] = %[[VAL_3]] : !llvm.ptr, i32
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.addressof @_ZN4sycl3_V16detailL17kernel_signaturesE : !llvm.ptr
// CHECK:           %[[VAL_10:.*]] = llvm.getelementptr %[[VAL_9]][0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x struct<"struct.sycl::_V1::detail::kernel_param_desc_t", (i32, i32, i32)>>
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.addressof @kernel_num_params_str : !llvm.ptr
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.addressof @kernel_param_desc_str : !llvm.ptr
// CHECK:           %[[VAL_13:.*]] = llvm.alloca %[[VAL_4]] x i32 : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_14:.*]] = llvm.alloca %[[VAL_4]] x !llvm.ptr : (i32) -> !llvm.ptr
// CHECK:           llvm.store %[[VAL_5]], %[[VAL_13]] : i32, !llvm.ptr
// CHECK:           llvm.store %[[VAL_10]], %[[VAL_14]] : !llvm.ptr, !llvm.ptr
// CHECK:           "llvm.intr.var.annotation"(%[[VAL_13]], %[[VAL_11]], %[[VAL_6]], %[[VAL_3]], %[[VAL_6]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
// CHECK:           "llvm.intr.var.annotation"(%[[VAL_14]], %[[VAL_12]], %[[VAL_6]], %[[VAL_3]], %[[VAL_6]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
// CHECK:           llvm.br ^bb1
// CHECK:         ^bb1:
// CHECK:           sycl.host.schedule_kernel %[[VAL_0]] -> @device_functions::@foo[nd_range %[[VAL_1]]](%[[VAL_8]], %[[VAL_2]], %[[VAL_3]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
llvm.func @raise_schedule_parallel_for_ndrange_std_layout_x3(%handler: !llvm.ptr, %ndrange: !llvm.ptr, %some_arg: !llvm.ptr) {
  %c0 = llvm.mlir.constant (0 : i32) : i32
  %c1 = llvm.mlir.constant (1 : i32) : i32
  %c3 = llvm.mlir.constant (3 : i32) : i32
  %nullptr = llvm.mlir.zero : !llvm.ptr

  sycl.host.handler.set_nd_range %handler -> nd_range %ndrange : !llvm.ptr, !llvm.ptr

  %lambda = llvm.alloca %c1 x !lambda_class : (i32) -> !llvm.ptr
  %some_struct = llvm.alloca %c1 x !llvm.struct<"some_struct", opaque> : (i32) -> !llvm.ptr
  sycl.host.set_captured %lambda[0] = %some_struct : !llvm.ptr, !llvm.ptr
  sycl.host.set_captured %lambda[1] = %some_arg : !llvm.ptr, !llvm.ptr
  sycl.host.set_captured %lambda[2] = %c0 : !llvm.ptr, i32

  %kernel_signatures = llvm.mlir.addressof @_ZN4sycl3_V16detailL17kernel_signaturesE : !llvm.ptr
  %sig_offset = llvm.getelementptr %kernel_signatures[0, 3] : (!llvm.ptr) -> !llvm.ptr, !kernel_signatures_type
  %kernel_num_params_str = llvm.mlir.addressof @kernel_num_params_str : !llvm.ptr
  %kernel_param_desc_str = llvm.mlir.addressof @kernel_param_desc_str : !llvm.ptr
  %num_params = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %param_desc = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
  llvm.store %c3, %num_params : i32, !llvm.ptr
  llvm.store %sig_offset, %param_desc : !llvm.ptr, !llvm.ptr
  "llvm.intr.var.annotation"(%num_params, %kernel_num_params_str, %nullptr, %c0, %nullptr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
  "llvm.intr.var.annotation"(%param_desc, %kernel_param_desc_str, %nullptr, %c0, %nullptr) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
  llvm.br ^bb1

^bb1:
  sycl.host.handler.set_kernel %handler -> @device_functions::@foo : !llvm.ptr

  llvm.return
}
