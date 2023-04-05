// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm='use-opaque-pointers=0' -verify-diagnostics %s | FileCheck %s

//===-------------------------------------------------------------------------------------------------===//
// Constructors for sycl::accessor<t, d, m, t, p>::accessor()
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK: llvm.func @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC2Ev([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::accessor.1",.*]])
func.func private @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC2Ev(memref<?x!sycl_accessor_1_i32_rw_gb>)

func.func @accessorInt1ReadWriteGlobalBufferFalseCtor(%arg0: memref<?x!sycl_accessor_1_i32_rw_gb>) {
  // CHECK: llvm.call @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC2Ev({{.*}}) : ([[THIS_PTR_TYPE]]) -> ()
  sycl.constructor @accessor(%arg0) {MangledFunctionName = @_ZN2cl4sycl8accessorIiLi1ELNS0_6access4modeE1026ELNS2_6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC2Ev} : (memref<?x!sycl_accessor_1_i32_rw_gb>)
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for sycl::id<n>::id()
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl2idILi1EEC2Ev([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.1",.*]])
func.func private @_ZN2cl4sycl2idILi1EEC2Ev(memref<?x!sycl_id_1_>)

func.func @id1Ctor(%arg0: memref<?x!sycl_id_1_>) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi1EEC2Ev({{.*}}) : ([[THIS_PTR_TYPE]]) -> ()
  sycl.constructor @id(%arg0) {MangledFunctionName = @_ZN2cl4sycl2idILi1EEC2Ev} : (memref<?x!sycl_id_1_>)
  return
}

// -----

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl2idILi2EEC2Ev([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.2",.*]])
func.func private @_ZN2cl4sycl2idILi2EEC2Ev(memref<?x!sycl_id_2_>)

func.func @id2Ctor(%arg0: memref<?x!sycl_id_2_>) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi2EEC2Ev({{.*}}) : ([[THIS_PTR_TYPE]]) -> ()
  sycl.constructor @id(%arg0) {MangledFunctionName = @_ZN2cl4sycl2idILi2EEC2Ev} : (memref<?x!sycl_id_2_>)
  return
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl2idILi3EEC2Ev([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.3",.*]])
func.func private @_ZN2cl4sycl2idILi3EEC2Ev(memref<?x!sycl_id_3_>)

func.func @id3Ctor(%arg0: memref<?x!sycl_id_3_>) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi3EEC2Ev({{.*}}) : ([[THIS_PTR_TYPE]]) -> ()
  sycl.constructor @id(%arg0) {MangledFunctionName = @_ZN2cl4sycl2idILi3EEC2Ev} : (memref<?x!sycl_id_3_>)
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for cl::sycl::id<n>::id<n>(std::enable_if<(n)==(n), unsigned long>::type)
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.1",.*]], i64)
func.func private @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(memref<?x!sycl_id_1_>, i64)

func.func @id1CtorSizeT(%arg0: memref<?x!sycl_id_1_>, %arg1: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64) -> ()
  sycl.constructor @id(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE} : (memref<?x!sycl_id_1_>, i64)
  return
}

// -----

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.2",.*]], i64)
func.func private @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE(memref<?x!sycl_id_2_>, i64)

func.func @id2CtorSizeT(%arg0: memref<?x!sycl_id_2_>, %arg1: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64) -> ()
  sycl.constructor @id(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE} : (memref<?x!sycl_id_2_>, i64)
  return
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.3",.*]], i64)
func.func private @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE(memref<?x!sycl_id_3_>, i64)

func.func @id3CtorSizeT(%arg0: memref<?x!sycl_id_3_>, %arg1: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64) -> ()
  sycl.constructor @id(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE} : (memref<?x!sycl_id_3_>, i64)
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for cl::sycl::id<n>::id<n>(std::enable_if<(n)==(n), unsigned long>::type, unsigned long)
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.1",.*]], i64, i64)
func.func private @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm(memref<?x!sycl_id_1_>, i64, i64)

func.func @id1CtorRange(%arg0: memref<?x!sycl_id_1_>, %arg1: i64, %arg2: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm({{.*}}, {{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64, i64) -> ()
  sycl.constructor @id(%arg0, %arg1, %arg2) {MangledFunctionName = @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm} : (memref<?x!sycl_id_1_>, i64, i64)
  return
}

// -----

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.2",.*]], i64, i64)
func.func private @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(memref<?x!sycl_id_2_>, i64, i64)

func.func @id2CtorRange(%arg0: memref<?x!sycl_id_2_>, %arg1: i64, %arg2: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm({{.*}}, {{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64, i64) -> ()
  sycl.constructor @id(%arg0, %arg1, %arg2) {MangledFunctionName = @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm} : (memref<?x!sycl_id_2_>, i64, i64)
  return
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.3",.*]], i64, i64)
func.func private @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm(memref<?x!sycl_id_3_>, i64, i64)

func.func @id3CtorRange(%arg0: memref<?x!sycl_id_3_>, %arg1: i64, %arg2: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm({{.*}}, {{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64, i64) -> ()
  sycl.constructor @id(%arg0, %arg1, %arg2) {MangledFunctionName = @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm} : (memref<?x!sycl_id_3_>, i64, i64)
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for cl::sycl::id<n>::id<n>(std::enable_if<(n)==(n), unsigned long>::type, unsigned long, unsigned long)
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.1",.*]], i64, i64, i64)
func.func private @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm(memref<?x!sycl_id_1_>, i64, i64, i64)

func.func @id1CtorItem(%arg0: memref<?x!sycl_id_1_>, %arg1: i64, %arg2: i64, %arg3: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm({{.*}}, {{.*}}, {{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64, i64, i64) -> ()
  sycl.constructor @id(%arg0, %arg1, %arg2, %arg3) {MangledFunctionName = @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm} : (memref<?x!sycl_id_1_>, i64, i64, i64)
  return
}

// -----

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.2",.*]], i64, i64, i64)
func.func private @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm(memref<?x!sycl_id_2_>, i64, i64, i64)

func.func @id2CtorItem(%arg0: memref<?x!sycl_id_2_>, %arg1: i64, %arg2: i64, %arg3: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm({{.*}}, {{.*}}, {{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64, i64, i64) -> ()
  sycl.constructor @id(%arg0, %arg1, %arg2, %arg3) {MangledFunctionName = @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm} : (memref<?x!sycl_id_2_>, i64, i64, i64)
  return
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.3",.*]], i64, i64, i64)
func.func private @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm(memref<?x!sycl_id_3_>, i64, i64, i64)

func.func @id3CtorItem(%arg0: memref<?x!sycl_id_3_>, %arg1: i64, %arg2: i64, %arg3: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm({{.*}}, {{.*}}, {{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64, i64, i64) -> ()
  sycl.constructor @id(%arg0, %arg1, %arg2, %arg3) {MangledFunctionName = @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm} : (memref<?x!sycl_id_3_>, i64, i64, i64)
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors sycl::id<n>::id(sycl::id<n> const&, sycl::id<n> const&)
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl2idILi1EEC1ERKS2_([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.1",.*]], [[THIS_PTR_TYPE]])
func.func private @_ZN2cl4sycl2idILi1EEC1ERKS2_(memref<?x!sycl_id_1_>, memref<?x!sycl_id_1_>)

func.func @id1CopyCtor(%arg0: memref<?x!sycl_id_1_>, %arg1: memref<?x!sycl_id_1_>) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi1EEC1ERKS2_({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], [[THIS_PTR_TYPE]]) -> ()
 sycl.constructor @id(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl2idILi1EEC1ERKS2_} : (memref<?x!sycl_id_1_>, memref<?x!sycl_id_1_>)
  return
}

// -----

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl2idILi2EEC1ERKS2_([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.2",.*]], [[THIS_PTR_TYPE]])
func.func private @_ZN2cl4sycl2idILi2EEC1ERKS2_(memref<?x!sycl_id_2_>, memref<?x!sycl_id_2_>)

func.func @id2CopyCtor(%arg0: memref<?x!sycl_id_2_>, %arg1: memref<?x!sycl_id_2_>) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi2EEC1ERKS2_({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], [[THIS_PTR_TYPE]]) -> ()
 sycl.constructor @id(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl2idILi2EEC1ERKS2_} : (memref<?x!sycl_id_2_>, memref<?x!sycl_id_2_>)
  return
}

// -----

!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl2idILi3EEC1ERKS2_([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::id.3",.*]], [[THIS_PTR_TYPE]])
func.func private @_ZN2cl4sycl2idILi3EEC1ERKS2_(memref<?x!sycl_id_3_>, memref<?x!sycl_id_3_>)

func.func @id3CopyCtor(%arg0: memref<?x!sycl_id_3_>, %arg1: memref<?x!sycl_id_3_>) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi3EEC1ERKS2_({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], [[THIS_PTR_TYPE]]) -> ()
 sycl.constructor @id(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl2idILi3EEC1ERKS2_} : (memref<?x!sycl_id_3_>, memref<?x!sycl_id_3_>)
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for sycl::range<n>::range()
//===-------------------------------------------------------------------------------------------------===//

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl5rangeILi1EEC2Ev([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.1",.*]])
func.func private @_ZN2cl4sycl5rangeILi1EEC2Ev(memref<?x!sycl_range_1_>)

func.func @range1Ctor(%arg0: memref<?x!sycl_range_1_>) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi1EEC2Ev({{.*}}) : ([[THIS_PTR_TYPE]]) -> ()
  sycl.constructor @range(%arg0) {MangledFunctionName = @_ZN2cl4sycl5rangeILi1EEC2Ev} : (memref<?x!sycl_range_1_>)
  return
}

// -----

!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl5rangeILi2EEC2Ev([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.2",.*]])
func.func private @_ZN2cl4sycl5rangeILi2EEC2Ev(memref<?x!sycl_range_2_>)

func.func @range2Ctor(%arg0: memref<?x!sycl_range_2_>) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi2EEC2Ev({{.*}}) : ([[THIS_PTR_TYPE]]) -> ()
  sycl.constructor @range(%arg0) {MangledFunctionName = @_ZN2cl4sycl5rangeILi2EEC2Ev} : (memref<?x!sycl_range_2_>)
  return
}

// -----

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<2xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl5rangeILi3EEC2Ev([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.3",.*]])
func.func private @_ZN2cl4sycl5rangeILi3EEC2Ev(memref<?x!sycl_range_3_>)

func.func @range3Ctor(%arg0: memref<?x!sycl_range_3_>) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi3EEC2Ev({{.*}}) : ([[THIS_PTR_TYPE]]) -> ()
  sycl.constructor @range(%arg0) {MangledFunctionName = @_ZN2cl4sycl5rangeILi3EEC2Ev} : (memref<?x!sycl_range_3_>)
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for cl::sycl::range<n>::range<n>(std::enable_if<(n)==(n), unsigned long>::type)
//===-------------------------------------------------------------------------------------------------===//

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.1",.*]], i64)
func.func private @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(memref<?x!sycl_range_1_>, i64)

func.func @range1CtorSizeT(%arg0: memref<?x!sycl_range_1_>, %arg1: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64) -> ()
  sycl.constructor @range(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE} : (memref<?x!sycl_range_1_>, i64)
  return
}

// -----

!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.2",.*]], i64)
func.func private @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE( memref<?x!sycl_range_2_>, i64)

func.func @range2CtorSizeT(%arg0: memref<?x!sycl_range_2_>, %arg1: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64) -> ()
  sycl.constructor @range(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE} : (memref<?x!sycl_range_2_>, i64)
  return
}

// -----

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<2xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.3",.*]], i64)
func.func private @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE(memref<?x!sycl_range_3_>, i64)

func.func @range3CtorSizeT(%arg0: memref<?x!sycl_range_3_>, %arg1: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64) -> ()
  sycl.constructor @range(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE} : (memref<?x!sycl_range_3_>, i64)
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for cl::sycl::range<n>::range<n>(std::enable_if<(n)==(n), unsigned long>::type, unsigned long)
//===-------------------------------------------------------------------------------------------------===//

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.1",.*]], i64, i64)
func.func private @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm(memref<?x!sycl_range_1_>, i64, i64)

func.func @range1Ctor2SizeT(%arg0: memref<?x!sycl_range_1_>, %arg1: i64, %arg2: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm({{.*}}, {{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64, i64) -> ()
  sycl.constructor @range(%arg0, %arg1, %arg2) {MangledFunctionName = @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm} : (memref<?x!sycl_range_1_>, i64, i64)
  return
}

// -----

!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.2",.*]], i64, i64)
func.func private @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm(memref<?x!sycl_range_2_>, i64, i64)

func.func @range2Ctor2SizeT(%arg0: memref<?x!sycl_range_2_>, %arg1: i64, %arg2: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm({{.*}}, {{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64, i64) -> ()
  sycl.constructor @range(%arg0, %arg1, %arg2) {MangledFunctionName = @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm} : (memref<?x!sycl_range_2_>, i64, i64)
  return
}

// -----

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<2xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.3",.*]], i64, i64)
func.func private @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm(memref<?x!sycl_range_3_>, i64, i64)

func.func @range3Ctor2SizeT(%arg0: memref<?x!sycl_range_3_>, %arg1: i64, %arg2: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm({{.*}}, {{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64, i64) -> ()
  sycl.constructor @range(%arg0, %arg1, %arg2) {MangledFunctionName = @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm} : (memref<?x!sycl_range_3_>, i64, i64)
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for cl::sycl::range<n>::range<n>(std::enable_if<(n)==(n), unsigned long>::type, unsigned long, unsigned long)
//===-------------------------------------------------------------------------------------------------===//

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.1",.*]], i64, i64, i64)
func.func private @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm(memref<?x!sycl_range_1_>, i64, i64, i64)

func.func @range1Ctor3SizeT(%arg0: memref<?x!sycl_range_1_>, %arg1: i64, %arg2: i64, %arg3: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm({{.*}}, {{.*}}, {{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64, i64, i64) -> ()
  sycl.constructor @range(%arg0, %arg1, %arg2, %arg3) {MangledFunctionName = @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm} : (memref<?x!sycl_range_1_>, i64, i64, i64)
  return
}

// -----

!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.2",.*]], i64, i64, i64)
func.func private @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm(memref<?x!sycl_range_2_>, i64, i64, i64)

func.func @range2Ctor3SizeT(%arg0: memref<?x!sycl_range_2_>, %arg1: i64, %arg2: i64, %arg3: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm({{.*}}, {{.*}}, {{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64, i64, i64) -> ()
  sycl.constructor @range(%arg0, %arg1, %arg2, %arg3) {MangledFunctionName = @_ZN2cl4sycl5rangeILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm} : (memref<?x!sycl_range_2_>, i64, i64, i64)
  return
}

// -----

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<2xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.3",.*]], i64, i64, i64)
func.func private @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm(memref<?x!sycl_range_3_>, i64, i64, i64)

func.func @range3Ctor3SizeT(%arg0: memref<?x!sycl_range_3_>, %arg1: i64, %arg2: i64, %arg3: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm({{.*}}, {{.*}}, {{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], i64, i64, i64) -> ()
  sycl.constructor @range(%arg0, %arg1, %arg2, %arg3) {MangledFunctionName = @_ZN2cl4sycl5rangeILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm} : (memref<?x!sycl_range_3_>, i64, i64, i64)
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors sycl::range<n>::id(sycl::range<n> const&, sycl::range<n> const&)
//===-------------------------------------------------------------------------------------------------===//

!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl5rangeILi1EEC1ERKS2_([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.1",.*]], [[THIS_PTR_TYPE]])
func.func private @_ZN2cl4sycl5rangeILi1EEC1ERKS2_(memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>)

func.func @range1CopyCtor(%arg0: memref<?x!sycl_range_1_>, %arg1: memref<?x!sycl_range_1_>) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi1EEC1ERKS2_({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], [[THIS_PTR_TYPE]]) -> ()
 sycl.constructor @range(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl5rangeILi1EEC1ERKS2_} : (memref<?x!sycl_range_1_>, memref<?x!sycl_range_1_>)
  return
}

// -----

!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl5rangeILi2EEC1ERKS2_([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.2",.*]], [[THIS_PTR_TYPE]])
func.func private @_ZN2cl4sycl5rangeILi2EEC1ERKS2_(memref<?x!sycl_range_2_>, memref<?x!sycl_range_2_>)

func.func @range2CopyCtor(%arg0: memref<?x!sycl_range_2_>, %arg1: memref<?x!sycl_range_2_>) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi2EEC1ERKS2_({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], [[THIS_PTR_TYPE]]) -> ()
 sycl.constructor @range(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl5rangeILi2EEC1ERKS2_} : (memref<?x!sycl_range_2_>, memref<?x!sycl_range_2_>)
  return
}

// -----

!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<2xi64, 4>)>)>
// CHECK: llvm.func @_ZN2cl4sycl5rangeILi3EEC1ERKS2_([[THIS_PTR_TYPE:!llvm.ptr<struct<"class.sycl::_V1::range.3",.*]], [[THIS_PTR_TYPE]])
func.func private @_ZN2cl4sycl5rangeILi3EEC1ERKS2_(memref<?x!sycl_range_3_>, memref<?x!sycl_range_3_>)

func.func @range3CopyCtor(%arg0: memref<?x!sycl_range_3_>, %arg1: memref<?x!sycl_range_3_>) {
  // CHECK: llvm.call @_ZN2cl4sycl5rangeILi3EEC1ERKS2_({{.*}}, {{.*}}) : ([[THIS_PTR_TYPE]], [[THIS_PTR_TYPE]]) -> ()
 sycl.constructor @range(%arg0, %arg1) {MangledFunctionName = @_ZN2cl4sycl5rangeILi3EEC1ERKS2_} : (memref<?x!sycl_range_3_>, memref<?x!sycl_range_3_>)
  return
}
