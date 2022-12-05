// RUN: cgeist %s --function=* -S | FileCheck %s
// RUN: cgeist %s --function=* -S -emit-llvm | FileCheck %s --check-prefix=LLVM

#include <cstddef>

typedef size_t size_t_vec __attribute__((ext_vector_type(3)));

size_t evt() {
  size_t_vec stv;
  return stv.x;
}

extern "C" const size_t_vec stv;
size_t evt2() {
  return stv.x;
}


// CHECK: memref.global constant @stv : memref<vector<3xi64>> {alignment = 32 : i64}

// CHECK-LABEL:   func.func @_Z3evtv() -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %0 = llvm.mlir.undef : vector<3xi64>
// CHECK-NEXT:    %1 = vector.extract %0[0] : vector<3xi64>
// CHECK-NEXT:    return %1 : i64
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_Z4evt2v() -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = memref.get_global @stv : memref<vector<3xi64>>
// CHECK-NEXT:     %alloca = memref.alloca() : memref<1xindex>
// CHECK-NEXT:     %reshape = memref.reshape %0(%alloca) : (memref<vector<3xi64>>, memref<1xindex>) -> memref<1xvector<3xi64>>
// CHECK-NEXT:     %1 = affine.load %reshape[0] : memref<1xvector<3xi64>>
// CHECK-NEXT:     %2 = vector.extract %1[0] : vector<3xi64>
// CHECK-NEXT:     return %2 : i64
// CHECK-NEXT:     }

// LLVM:       @stv = external constant <3 x i64>, align 32

// LLVM-LABEL: define i64 @_Z3evtv() {
// LLVM-NEXT:   ret i64 undef

// LLVM-LABEL: define i64 @_Z4evt2v() {
// LLVM-NEXT:   %1 = load <3 x i64>, <3 x i64>* @stv, align 32
// LLVM-NEXT:   %2 = extractelement <3 x i64> %1, i64 0
// LLVM-NEXT:   ret i64 %2
