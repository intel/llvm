// RUN: polygeist-opt --arg-promotion --split-input-file %s | FileCheck %s

#map = affine_map<(s0) -> (s0 - 1)>  
gpu.module @device_func {
  // COM: This function is not a candidate because it is not defined.
  func.func private @callee1(%arg0: memref<?x!llvm.struct<(i32, i64)>>) -> ()
    //CHECK-LABEL: func.func private @callee1
  gpu.func @caller1() kernel {  
    // CHECK-LABEL: gpu.func @caller1() kernel
    // CHECK:         func.call @callee1({{.*}}) : (memref<?x!llvm.struct<(i32, i64)>>) -> ()
    %alloca = memref.alloca() : memref<1x!llvm.struct<(i32, i64)>>
    %cast = memref.cast %alloca : memref<1x!llvm.struct<(i32, i64)>> to memref<?x!llvm.struct<(i32, i64)>>    
    func.call @callee1(%cast) : (memref<?x!llvm.struct<(i32, i64)>>) -> ()
    gpu.return    
  }

  // COM: This function is not a candidate because it doesn't have any argument.
  func.func private @callee2() -> () {
    // CHECK-LABEL: func.func private @callee2
    func.return 
  }
  gpu.func @caller2() kernel {
    // CHECK-LABEL: gpu.func @caller2() kernel
    // CHECK:         func.call @callee2() : () -> ()
    func.call @callee2() : () -> ()
    gpu.return
  }

  // COM: This function is not a candidate because the argument doesn't have the
  // COM: expected type.
  func.func private @callee3(%arg0: memref<?xi32>) -> () {
    // CHECK-LABEL: func.func private @callee3
    // CHECK-SAME:    (%arg0: memref<?xi32>) {
    func.return
  }
  gpu.func @caller3() kernel {
    // CHECK-LABEL: gpu.func @caller3() kernel
    // CHECK:         func.call @callee3({{.*}}) : (memref<?xi32>) -> ()
    %alloca = memref.alloca() : memref<1xi32>
    %cast = memref.cast %alloca : memref<1xi32> to memref<?xi32>
    func.call @callee3(%cast) : (memref<?xi32>) -> ()
    gpu.return
  }

  // COM: Should not peel the argument because it is a multidimentional memref.
  func.func private @callee4(%arg0: memref<?x1x!llvm.struct<(i32)>>) {
    // CHECK-LABEL: func.func private @callee4
    // CHECK-SAME:    (%arg0: memref<?x1x!llvm.struct<(i32)>>) {
    func.return
  }
  gpu.func @caller4() kernel {
    // CHECK-LABEL: gpu.func @caller4() kernel
    // CHECK:         func.call @callee4({{.*}}) : (memref<?x1x!llvm.struct<(i32)>>) -> ()
    %alloca = memref.alloca() : memref<1x1x!llvm.struct<(i32)>>
    %cast = memref.cast %alloca : memref<1x1x!llvm.struct<(i32)>> to memref<?x1x!llvm.struct<(i32)>>
    func.call @callee4(%cast) : (memref<?x1x!llvm.struct<(i32)>>) -> ()
    gpu.return
  }

  // COM: Should not peel the argument because it is memref with non-identity
  // COM: layout.
  func.func private @callee5(%arg0: memref<?x!llvm.struct<(i32)>, #map>) {
    // CHECK-LABEL: func.func private @callee5
    // CHECK-SAME:    (%arg0: memref<?x!llvm.struct<(i32)>, #map>) {
    func.return
  }
  gpu.func @caller5() kernel {
    // CHECK-LABEL: gpu.func @caller5() kernel
    // CHECK:         func.call @callee5({{.*}}) : (memref<?x!llvm.struct<(i32)>, #map>) -> ()
    %alloca = memref.alloca() : memref<8x!llvm.struct<(i32)>, #map>
    %cast = memref.cast %alloca : memref<8x!llvm.struct<(i32)>, #map> to memref<?x!llvm.struct<(i32)>, #map>
    func.call @callee5(%cast) : (memref<?x!llvm.struct<(i32)>, #map>) -> ()
    gpu.return
  }

  // COM: Test that an argument that is a memref with element type
  // COM: 'struct<(struct<(...)>)>' is not peeled.
  func.func private @callee6(%arg0: memref<?x!llvm.struct<(struct<(i32)>)>>) {
    // CHECK-LABEL: func.func private @callee6
    // CHECK-SAME:    (%arg0: memref<?x!llvm.struct<(struct<(i32)>)>>) {
    func.return
  }
  gpu.func @caller6() kernel {
    // CHECK-LABEL: gpu.func @caller6() kernel
    // CHECK:         func.call @callee6({{.*}}) : (memref<?x!llvm.struct<(struct<(i32)>)>>) -> ()
    %alloca = memref.alloca() : memref<3x!llvm.struct<(struct<(i32)>)>>
    %cast = memref.cast %alloca : memref<3x!llvm.struct<(struct<(i32)>)>> to memref<?x!llvm.struct<(struct<(i32)>)>>
    func.call @callee6(%cast) : (memref<?x!llvm.struct<(struct<(i32)>)>>) -> ()
    gpu.return
  }

  // COM: This function is not a candidate because one call site uses the
  // COM: argument after the call.
  func.func private @callee7(%arg0: memref<?x!llvm.struct<(i32, i64)>>) {  
    // CHECK-LABEL: func.func private @callee7
    // CHECK-SAME:    (%arg0: memref<?x!llvm.struct<(i32, i64)>>) {      
    func.return
  }  
  gpu.func @caller7() kernel {
    // CHECK-LABEL: gpu.func @caller7() kernel
    // CHECK:         func.call @callee7([[ARG0:%.*]]) : (memref<?x!llvm.struct<(i32, i64)>>) -> ()
    %alloca = memref.alloca() : memref<1x!llvm.struct<(i32, i64)>>
    %cast = memref.cast %alloca : memref<1x!llvm.struct<(i32, i64)>> to memref<?x!llvm.struct<(i32, i64)>>
    func.call @callee7(%cast) : (memref<?x!llvm.struct<(i32, i64)>>) -> ()
    %i = arith.constant 0 : index
    %0 = memref.load %cast[%i] : memref<?x!llvm.struct<(i32, i64)>>
    gpu.return
  }
}
