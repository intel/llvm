// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm -verify-diagnostics %s | FileCheck %s

//===-------------------------------------------------------------------------------------------------===//
// Constructors for sycl::id<n>::id()
//===-------------------------------------------------------------------------------------------------===//

// CHECK: llvm.func @_ZN2cl4sycl2idILi1EEC2Ev([[THIS_PTR_TYPE:!llvm.struct<\(ptr<struct<"class.cl::sycl::id.1",.*]])
func.func @id1Ctor(%arg0: memref<?x!sycl.id<1>>) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi1EEC2Ev({{.*}}) : ([[THIS_PTR_TYPE]]) -> ()
  sycl.constructor(%arg0) {Type = @id} : (memref<?x!sycl.id<1>>) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi2EEC2Ev([[THIS_PTR_TYPE:!llvm.struct<\(ptr<struct<"class.cl::sycl::id.2",.*]])
func.func @id2Ctor(%arg0: memref<?x!sycl.id<2>>) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi2EEC2Ev({{.*}}) : ([[THIS_PTR_TYPE]]) -> ()
  sycl.constructor(%arg0) {Type = @id} : (memref<?x!sycl.id<2>>) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi3EEC2Ev([[THIS_PTR_TYPE:!llvm.struct<\(ptr<struct<"class.cl::sycl::id.3",.*]])
func.func @id3Ctor(%arg0: memref<?x!sycl.id<3>>) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi3EEC2Ev({{.*}}) : ([[THIS_PTR_TYPE]]) -> ()
  sycl.constructor(%arg0) {Type = @id} : (memref<?x!sycl.id<3>>) -> ()
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for cl::sycl::id<n>::id<n>(std::enable_if<(n)==(n), unsigned long>::type)
//===-------------------------------------------------------------------------------------------------===//

// CHECK: llvm.func @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE([[THIS_PTR_TYPE:!llvm.struct<\(ptr<struct<"class.cl::sycl::id.1",.*]], i64)
func.func @id1CtorSizeT(%arg0: memref<?x!sycl.id<1>>, %arg1: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE({{.*}}, %arg5) : ([[THIS_PTR_TYPE]], i64) -> ()
  sycl.constructor(%arg0, %arg1) {Type = @id} : (memref<?x!sycl.id<1>>, i64) -> ()
  return
}


// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE([[THIS_PTR_TYPE:!llvm.struct<\(ptr<struct<"class.cl::sycl::id.2",.*]], i64)
func.func @id2CtorSizeT(%arg0: memref<?x!sycl.id<2>>, %arg1: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeE({{.*}}, %arg5) : ([[THIS_PTR_TYPE]], i64) -> ()
  sycl.constructor(%arg0, %arg1) {Type = @id} : (memref<?x!sycl.id<2>>, i64) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE([[THIS_PTR_TYPE:!llvm.struct<\(ptr<struct<"class.cl::sycl::id.3",.*]], i64)
func.func @id3CtorSizeT(%arg0: memref<?x!sycl.id<3>>, %arg1: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeE({{.*}}, %arg5) : ([[THIS_PTR_TYPE]], i64) -> ()
  sycl.constructor(%arg0, %arg1) {Type = @id} : (memref<?x!sycl.id<3>>, i64) -> ()
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for cl::sycl::id<n>::id<n>(std::enable_if<(n)==(n), unsigned long>::type, unsigned long)
//===-------------------------------------------------------------------------------------------------===//

// CHECK: llvm.func @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm([[THIS_PTR_TYPE:!llvm.struct<\(ptr<struct<"class.cl::sycl::id.1",.*]], i64, i64)
func.func @id1CtorRange(%arg0: memref<?x!sycl.id<1>>, %arg1: i64, %arg2: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEm({{.*}}, %arg5, %arg6) : ([[THIS_PTR_TYPE]], i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2) {Type = @id} : (memref<?x!sycl.id<1>>, i64, i64) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm([[THIS_PTR_TYPE:!llvm.struct<\(ptr<struct<"class.cl::sycl::id.2",.*]], i64, i64)
func.func @id2CtorRange(%arg0: memref<?x!sycl.id<2>>, %arg1: i64, %arg2: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEm({{.*}}, %arg5, %arg6) : ([[THIS_PTR_TYPE]], i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2) {Type = @id} : (memref<?x!sycl.id<2>>, i64, i64) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm([[THIS_PTR_TYPE:!llvm.struct<\(ptr<struct<"class.cl::sycl::id.3",.*]], i64, i64)
func.func @id3CtorRange(%arg0: memref<?x!sycl.id<3>>, %arg1: i64, %arg2: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEm({{.*}}, %arg5, %arg6) : ([[THIS_PTR_TYPE]], i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2) {Type = @id} : (memref<?x!sycl.id<3>>, i64, i64) -> ()
  return
}

// -----

//===-------------------------------------------------------------------------------------------------===//
// Constructors for cl::sycl::id<n>::id<n>(std::enable_if<(n)==(n), unsigned long>::type, unsigned long, unsigned long)
//===-------------------------------------------------------------------------------------------------===//

// CHECK: llvm.func @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm([[THIS_PTR_TYPE:!llvm.struct<\(ptr<struct<"class.cl::sycl::id.1",.*]], i64, i64, i64)
func.func @id1CtorItem(%arg0: memref<?x!sycl.id<1>>, %arg1: i64, %arg2: i64, %arg3: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeEmm({{.*}}, %arg5, %arg6, %arg7) : ([[THIS_PTR_TYPE]], i64, i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2, %arg3) {Type = @id} : (memref<?x!sycl.id<1>>, i64, i64, i64) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm([[THIS_PTR_TYPE:!llvm.struct<\(ptr<struct<"class.cl::sycl::id.2",.*]], i64, i64, i64)
func.func @id2CtorItem(%arg0: memref<?x!sycl.id<2>>, %arg1: i64, %arg2: i64, %arg3: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi2EEC2ILi2EEENSt9enable_ifIXeqT_Li2EEmE4typeEmm({{.*}}, %arg5, %arg6, %arg7) : ([[THIS_PTR_TYPE]], i64, i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2, %arg3) {Type = @id} : (memref<?x!sycl.id<2>>, i64, i64, i64) -> ()
  return
}

// -----

// CHECK: llvm.func @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm([[THIS_PTR_TYPE:!llvm.struct<\(ptr<struct<"class.cl::sycl::id.3",.*]], i64, i64, i64)
func.func @id3CtorItem(%arg0: memref<?x!sycl.id<3>>, %arg1: i64, %arg2: i64, %arg3: i64) {
  // CHECK: llvm.call @_ZN2cl4sycl2idILi3EEC2ILi3EEENSt9enable_ifIXeqT_Li3EEmE4typeEmm({{.*}}, %arg5, %arg6, %arg7) : ([[THIS_PTR_TYPE]], i64, i64, i64) -> ()
  sycl.constructor(%arg0, %arg1, %arg2, %arg3) {Type = @id} : (memref<?x!sycl.id<3>>, i64, i64, i64) -> ()
  return
}

// -----
