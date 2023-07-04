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

// CHECK-LABEL: llvm.func @f() -> !llvm.ptr
// CHECK-NEXT:    %[[VAL_0:.*]] = sycl.host.get_kernel @device_functions::@foo : !llvm.ptr
// CHECK-NEXT:    llvm.return %[[VAL_0]] : !llvm.ptr
llvm.func @f() -> !llvm.ptr {
  %kn = llvm.mlir.addressof @kernel_ref : !llvm.ptr
  llvm.return %kn : !llvm.ptr
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
