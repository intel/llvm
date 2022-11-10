// RUN: cgeist %s --function=* -S | FileCheck %s

typedef size_t size_t_vec __attribute__((ext_vector_type(3)));

size_t evt() {
  size_t_vec stv;
  return stv.x;
}

extern "C" const size_t_vec stv;
size_t evt2() {
  return stv.x;
}

// CHECK:   func.func @_Z3evtv() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:    %alloca = memref.alloca() : memref<1xvector<3xi32>>
// CHECK-NEXT:    %0 = affine.load %alloca[0] : memref<1xvector<3xi32>>
// CHECK-NEXT:    %1 = llvm.extractelement %0[%c0_i64 : i64] : vector<3xi32>
// CHECK-NEXT:    return %1 : i32
// CHECK-NEXT:    }
// CHECK:   func.func @_Z4evt2v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:     %0 = memref.get_global @stv : memref<vector<3xi32>>
// CHECK-NEXT:     %alloca = memref.alloca() : memref<1xindex>
// CHECK-NEXT:     %reshape = memref.reshape %0(%alloca) : (memref<vector<3xi32>>, memref<1xindex>) -> memref<1xvector<3xi32>>
// CHECK-NEXT:     %1 = affine.load %reshape[0] : memref<1xvector<3xi32>>
// CHECK-NEXT:     %2 = llvm.extractelement %1[%c0_i64 : i64] : vector<3xi32>
// CHECK-NEXT:     return %2 : i32
// CHECK-NEXT:     }
