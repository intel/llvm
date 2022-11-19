// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm -verify-diagnostics %s | FileCheck %s

//===-------------------------------------------------------------------------------------------------===//
// Constructors for sycl::accessor<t, d, m, t, p>::accessor()
//===-------------------------------------------------------------------------------------------------===//

!sycl_accessor_1_i32_read_write_global_buffer = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK: llvm.func @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC2Ev([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::accessor.1",.*]])
func.func private @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC2Ev(memref<?x!sycl_accessor_1_i32_read_write_global_buffer>)

func.func @accessorInt1ReadWriteGlobalBufferFalseCtor(%arg0: memref<?x!sycl_accessor_1_i32_read_write_global_buffer>) {
  // CHECK: llvm.call @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC2Ev({{.*}}) : ([[THIS_PTR_TYPE]]) -> ()
  sycl.constructor(%arg0) {MangledFunctionName = @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC2Ev, TypeName = @accessor} : (memref<?x!sycl_accessor_1_i32_read_write_global_buffer>) -> () 
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for sycl::id<n>::id()
//===-------------------------------------------------------------------------------------------------===//

// CHECK: llvm.func @_ZN2cl4sycl2idILi1EEC2Ev([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.1",.*]])
func.func private @_ZN2cl4sycl2idILi1EEC2Ev(memref<?x!sycl.id<1>>)

func.func @id1Ctor(%arg0: memref<?x!sycl.id<1>>) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi1EEC2Ev({{.*}}) : ([[THIS_PTR_TYPE]]) -> ()
  sycl.constructor(%arg0) {MangledFunctionName = @_ZN2cl4sycl2idILi1EEC2Ev, TypeName = @id} : (memref<?x!sycl.id<1>>) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi2EEC2Ev([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.2",.*]])
func.func private @_ZN2cl4sycl2idILi2EEC2Ev(memref<?x!sycl.id<2>>)

func.func @id2Ctor(%arg0: memref<?x!sycl.id<2>>) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi2EEC2Ev({{.*}}) : ([[THIS_PTR_TYPE]]) -> ()
  sycl.constructor(%arg0) {MangledFunctionName = @_ZN2cl4sycl2idILi2EEC2Ev, TypeName = @id} : (memref<?x!sycl.id<2>>) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi3EEC2Ev([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.3",.*]])
func.func private @_ZN2cl4sycl2idILi3EEC2Ev(memref<?x!sycl.id<3>>)

func.func @id3Ctor(%arg0: memref<?x!sycl.id<3>>) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi3EEC2Ev({{.*}}) : ([[THIS_PTR_TYPE]]) -> ()
  sycl.constructor(%arg0) {MangledFunctionName = @_ZN2cl4sycl2idILi3EEC2Ev, TypeName = @id} : (memref<?x!sycl.id<3>>) -> ()
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for cl::sycl::id<n>::id<n>(std::enable_if<(n)==(n), unsigned long>::type)
//===-------------------------------------------------------------------------------------------------===//

// CHECK: llvm.func @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.1",.*]], i64)
func.func private @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(memref<?x!sycl.id<1>>, i64)

func.func @id1CtorSizeT(%arg0: memref<?x!sycl.id<1>>, %arg1: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE({{.*}}, %arg5) : ([[THIS_PTR_TYPE]], i64) -> ()
  sycl.constructor(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE, TypeName = @id} : (memref<?x!sycl.id<1>>, i64) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.2",.*]], i64)
func.func private @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE(memref<?x!sycl.id<2>>, i64)

func.func @id2CtorSizeT(%arg0: memref<?x!sycl.id<2>>, %arg1: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE({{.*}}, %arg5) : ([[THIS_PTR_TYPE]], i64) -> ()
  sycl.constructor(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE, TypeName = @id} : (memref<?x!sycl.id<2>>, i64) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.3",.*]], i64)
func.func private @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE(memref<?x!sycl.id<3>>, i64)

func.func @id3CtorSizeT(%arg0: memref<?x!sycl.id<3>>, %arg1: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE({{.*}}, %arg5) : ([[THIS_PTR_TYPE]], i64) -> ()
  sycl.constructor(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE, TypeName = @id} : (memref<?x!sycl.id<3>>, i64) -> ()
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for cl::sycl::id<n>::id<n>(std::enable_if<(n)==(n), unsigned long>::type, unsigned long)
//===-------------------------------------------------------------------------------------------------===//

// CHECK: llvm.func @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.1",.*]], i64, i64)
func.func private @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm(memref<?x!sycl.id<1>>, i64, i64)

func.func @id1CtorRange(%arg0: memref<?x!sycl.id<1>>, %arg1: i64, %arg2: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm({{.*}}, %arg5, %arg6) : ([[THIS_PTR_TYPE]], i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2) {MangledFunctionName = @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm, TypeName = @id} : (memref<?x!sycl.id<1>>, i64, i64) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.2",.*]], i64, i64)
func.func private @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(memref<?x!sycl.id<2>>, i64, i64)

func.func @id2CtorRange(%arg0: memref<?x!sycl.id<2>>, %arg1: i64, %arg2: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm({{.*}}, %arg5, %arg6) : ([[THIS_PTR_TYPE]], i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2) {MangledFunctionName = @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm, TypeName = @id} : (memref<?x!sycl.id<2>>, i64, i64) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.3",.*]], i64, i64)
func.func private @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm(memref<?x!sycl.id<3>>, i64, i64)

func.func @id3CtorRange(%arg0: memref<?x!sycl.id<3>>, %arg1: i64, %arg2: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm({{.*}}, %arg5, %arg6) : ([[THIS_PTR_TYPE]], i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2) {MangledFunctionName = @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm, TypeName = @id} : (memref<?x!sycl.id<3>>, i64, i64) -> ()
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for cl::sycl::id<n>::id<n>(std::enable_if<(n)==(n), unsigned long>::type, unsigned long, unsigned long)
//===-------------------------------------------------------------------------------------------------===//

// CHECK: llvm.func @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.1",.*]], i64, i64, i64)
func.func private @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm(memref<?x!sycl.id<1>>, i64, i64, i64)

func.func @id1CtorItem(%arg0: memref<?x!sycl.id<1>>, %arg1: i64, %arg2: i64, %arg3: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm({{.*}}, %arg5, %arg6, %arg7) : ([[THIS_PTR_TYPE]], i64, i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2, %arg3) {MangledFunctionName = @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm, TypeName = @id} : (memref<?x!sycl.id<1>>, i64, i64, i64) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.2",.*]], i64, i64, i64)
func.func private @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm(memref<?x!sycl.id<2>>, i64, i64, i64)

func.func @id2CtorItem(%arg0: memref<?x!sycl.id<2>>, %arg1: i64, %arg2: i64, %arg3: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm({{.*}}, %arg5, %arg6, %arg7) : ([[THIS_PTR_TYPE]], i64, i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2, %arg3) {MangledFunctionName = @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm, TypeName = @id} : (memref<?x!sycl.id<2>>, i64, i64, i64) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.3",.*]], i64, i64, i64)
func.func private @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm(memref<?x!sycl.id<3>>, i64, i64, i64)

func.func @id3CtorItem(%arg0: memref<?x!sycl.id<3>>, %arg1: i64, %arg2: i64, %arg3: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm({{.*}}, %arg5, %arg6, %arg7) : ([[THIS_PTR_TYPE]], i64, i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2, %arg3) {MangledFunctionName = @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm, TypeName = @id} : (memref<?x!sycl.id<3>>, i64, i64, i64) -> ()
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors sycl::id<n>::id(sycl::id<n> const&, sycl::id<n> const&)
//===-------------------------------------------------------------------------------------------------===//

// CHECK: llvm.func @_ZN2cl4sycl2idILi1EEC1ERKS2_([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.1",.*]], [[THIS_PTR_TYPE]])
func.func private @_ZN2cl4sycl2idILi1EEC1ERKS2_(memref<?x!sycl.id<1>>, memref<?x!sycl.id<1>>)

func.func @id1CopyCtor(%arg0: memref<?x!sycl.id<1>>, %arg1: memref<?x!sycl.id<1>>) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi1EEC1ERKS2_({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], [[THIS_PTR_TYPE]]) -> ()  
 "sycl.constructor"(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl2idILi1EEC1ERKS2_, TypeName = @id} : (memref<?x!sycl.id<1>>, memref<?x!sycl.id<1>>) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi2EEC1ERKS2_([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.2",.*]], [[THIS_PTR_TYPE]])
func.func private @_ZN2cl4sycl2idILi2EEC1ERKS2_(memref<?x!sycl.id<2>>, memref<?x!sycl.id<2>>)

func.func @id2CopyCtor(%arg0: memref<?x!sycl.id<2>>, %arg1: memref<?x!sycl.id<2>>) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi2EEC1ERKS2_({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], [[THIS_PTR_TYPE]]) -> ()
 "sycl.constructor"(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl2idILi2EEC1ERKS2_, TypeName = @id} : (memref<?x!sycl.id<2>>, memref<?x!sycl.id<2>>) -> ()  
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi3EEC1ERKS2_([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.3",.*]], [[THIS_PTR_TYPE]])
func.func private @_ZN2cl4sycl2idILi3EEC1ERKS2_(memref<?x!sycl.id<3>>, memref<?x!sycl.id<3>>)

func.func @id3CopyCtor(%arg0: memref<?x!sycl.id<3>>, %arg1: memref<?x!sycl.id<3>>) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi3EEC1ERKS2_({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], [[THIS_PTR_TYPE]]) -> ()
 "sycl.constructor"(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl2idILi3EEC1ERKS2_, TypeName = @id} : (memref<?x!sycl.id<3>>, memref<?x!sycl.id<3>>) -> ()
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for sycl::range<n>::range()
//===-------------------------------------------------------------------------------------------------===//

// CHECK: llvm.func @_ZN2cl4sycl5rangeILi1EEC2Ev([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.1",.*]])
func.func private @_ZN2cl4sycl5rangeILi1EEC2Ev(memref<?x!sycl.range<1>>)

func.func @range1Ctor(%arg0: memref<?x!sycl.range<1>>) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi1EEC2Ev({{.*}}) : ([[THIS_PTR_TYPE]]) -> ()
  sycl.constructor(%arg0) {MangledFunctionName = @_ZN2cl4sycl5rangeILi1EEC2Ev, TypeName = @range} : (memref<?x!sycl.range<1>>) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl5rangeILi2EEC2Ev([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.2",.*]])
func.func private @_ZN2cl4sycl5rangeILi2EEC2Ev(memref<?x!sycl.range<2>>)

func.func @range2Ctor(%arg0: memref<?x!sycl.range<2>>) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi2EEC2Ev({{.*}}) : ([[THIS_PTR_TYPE]]) -> ()
  sycl.constructor(%arg0) {MangledFunctionName = @_ZN2cl4sycl5rangeILi2EEC2Ev, TypeName = @range} : (memref<?x!sycl.range<2>>) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl5rangeILi3EEC2Ev([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.3",.*]])
func.func private @_ZN2cl4sycl5rangeILi3EEC2Ev(memref<?x!sycl.range<3>>)

func.func @range3Ctor(%arg0: memref<?x!sycl.range<3>>) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi3EEC2Ev({{.*}}) : ([[THIS_PTR_TYPE]]) -> ()
  sycl.constructor(%arg0) {MangledFunctionName = @_ZN2cl4sycl5rangeILi3EEC2Ev, TypeName = @range} : (memref<?x!sycl.range<3>>) -> ()
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for cl::sycl::range<n>::range<n>(std::enable_if<(n)==(n), unsigned long>::type)
//===-------------------------------------------------------------------------------------------------===//

// CHECK: llvm.func @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.1",.*]], i64)
func.func private @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(memref<?x!sycl.range<1>>, i64)

func.func @range1CtorSizeT(%arg0: memref<?x!sycl.range<1>>, %arg1: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE({{.*}}, %arg5) : ([[THIS_PTR_TYPE]], i64) -> ()
  sycl.constructor(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE, TypeName = @range} : (memref<?x!sycl.range<1>>, i64) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.2",.*]], i64)
func.func private @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE( memref<?x!sycl.range<2>>, i64)

func.func @range2CtorSizeT(%arg0: memref<?x!sycl.range<2>>, %arg1: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE({{.*}}, %arg5) : ([[THIS_PTR_TYPE]], i64) -> ()
  sycl.constructor(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE, TypeName = @range} : (memref<?x!sycl.range<2>>, i64) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.3",.*]], i64)
func.func private @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE(memref<?x!sycl.range<3>>, i64)

func.func @range3CtorSizeT(%arg0: memref<?x!sycl.range<3>>, %arg1: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE({{.*}}, %arg5) : ([[THIS_PTR_TYPE]], i64) -> ()
  sycl.constructor(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE, TypeName = @range} : (memref<?x!sycl.range<3>>, i64) -> ()
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for cl::sycl::range<n>::range<n>(std::enable_if<(n)==(n), unsigned long>::type, unsigned long)
//===-------------------------------------------------------------------------------------------------===//

// CHECK: llvm.func @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.1",.*]], i64, i64)
func.func private @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm(memref<?x!sycl.range<1>>, i64, i64)

func.func @range1Ctor2SizeT(%arg0: memref<?x!sycl.range<1>>, %arg1: i64, %arg2: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm({{.*}}, %arg5, %arg6) : ([[THIS_PTR_TYPE]], i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2) {MangledFunctionName = @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm, TypeName = @range} : (memref<?x!sycl.range<1>>, i64, i64) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.2",.*]], i64, i64)
func.func private @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(memref<?x!sycl.range<2>>, i64, i64)

func.func @range2Ctor2SizeT(%arg0: memref<?x!sycl.range<2>>, %arg1: i64, %arg2: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm({{.*}}, %arg5, %arg6) : ([[THIS_PTR_TYPE]], i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2) {MangledFunctionName = @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm, TypeName = @range} : (memref<?x!sycl.range<2>>, i64, i64) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.3",.*]], i64, i64)
func.func private @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm(memref<?x!sycl.range<3>>, i64, i64)

func.func @range3Ctor2SizeT(%arg0: memref<?x!sycl.range<3>>, %arg1: i64, %arg2: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm({{.*}}, %arg5, %arg6) : ([[THIS_PTR_TYPE]], i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2) {MangledFunctionName = @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm, TypeName = @range} : (memref<?x!sycl.range<3>>, i64, i64) -> ()
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for cl::sycl::range<n>::range<n>(std::enable_if<(n)==(n), unsigned long>::type, unsigned long, unsigned long)
//===-------------------------------------------------------------------------------------------------===//

// CHECK: llvm.func @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.1",.*]], i64, i64, i64)
func.func private @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm(memref<?x!sycl.range<1>>, i64, i64, i64)

func.func @range1Ctor3SizeT(%arg0: memref<?x!sycl.range<1>>, %arg1: i64, %arg2: i64, %arg3: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm({{.*}}, %arg5, %arg6, %arg7) : ([[THIS_PTR_TYPE]], i64, i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2, %arg3) {MangledFunctionName = @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm, TypeName = @range} : (memref<?x!sycl.range<1>>, i64, i64, i64) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.2",.*]], i64, i64, i64)
func.func private @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm(memref<?x!sycl.range<2>>, i64, i64, i64)

func.func @range2Ctor3SizeT(%arg0: memref<?x!sycl.range<2>>, %arg1: i64, %arg2: i64, %arg3: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm({{.*}}, %arg5, %arg6, %arg7) : ([[THIS_PTR_TYPE]], i64, i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2, %arg3) {MangledFunctionName = @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm, TypeName = @range} : (memref<?x!sycl.range<2>>, i64, i64, i64) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.3",.*]], i64, i64, i64)
func.func private @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm(memref<?x!sycl.range<3>>, i64, i64, i64)

func.func @range3Ctor3SizeT(%arg0: memref<?x!sycl.range<3>>, %arg1: i64, %arg2: i64, %arg3: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm({{.*}}, %arg5, %arg6, %arg7) : ([[THIS_PTR_TYPE]], i64, i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2, %arg3) {MangledFunctionName = @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm, TypeName = @range} : (memref<?x!sycl.range<3>>, i64, i64, i64) -> ()
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors sycl::range<n>::id(sycl::range<n> const&, sycl::range<n> const&)
//===-------------------------------------------------------------------------------------------------===//

// CHECK: llvm.func @_ZN2cl4sycl5rangeILi1EEC1ERKS2_([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.1",.*]], [[THIS_PTR_TYPE]])
func.func private @_ZN2cl4sycl5rangeILi1EEC1ERKS2_(memref<?x!sycl.range<1>>, memref<?x!sycl.range<1>>)

func.func @range1CopyCtor(%arg0: memref<?x!sycl.range<1>>, %arg1: memref<?x!sycl.range<1>>) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi1EEC1ERKS2_({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], [[THIS_PTR_TYPE]]) -> ()
 "sycl.constructor"(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl5rangeILi1EEC1ERKS2_, TypeName = @range} : (memref<?x!sycl.range<1>>, memref<?x!sycl.range<1>>) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl5rangeILi2EEC1ERKS2_([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.2",.*]], [[THIS_PTR_TYPE]])
func.func private @_ZN2cl4sycl5rangeILi2EEC1ERKS2_(memref<?x!sycl.range<2>>, memref<?x!sycl.range<2>>)

func.func @range2CopyCtor(%arg0: memref<?x!sycl.range<2>>, %arg1: memref<?x!sycl.range<2>>) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi2EEC1ERKS2_({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], [[THIS_PTR_TYPE]]) -> ()
 "sycl.constructor"(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl5rangeILi2EEC1ERKS2_, TypeName = @range} : (memref<?x!sycl.range<2>>, memref<?x!sycl.range<2>>) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl5rangeILi3EEC1ERKS2_([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.3",.*]], [[THIS_PTR_TYPE]])
func.func private @_ZN2cl4sycl5rangeILi3EEC1ERKS2_(memref<?x!sycl.range<3>>, memref<?x!sycl.range<3>>)

func.func @range3CopyCtor(%arg0: memref<?x!sycl.range<3>>, %arg1: memref<?x!sycl.range<3>>) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi3EEC1ERKS2_({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], [[THIS_PTR_TYPE]]) -> ()
 "sycl.constructor"(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl5rangeILi3EEC1ERKS2_, TypeName = @range} : (memref<?x!sycl.range<3>>, memref<?x!sycl.range<3>>) -> ()
  return
}
