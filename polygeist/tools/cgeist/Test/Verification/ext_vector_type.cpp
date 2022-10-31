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
// CHECK-NEXT:    %alloca = memref.alloca() : memref<1x3xi32>
// CHECK-NEXT:    %0 = affine.load %alloca[0, 0] : memref<1x3xi32>
// CHECK-NEXT:    return %0 : i32
// CHECK-NEXT:    }
// CHECK:   func.func @_Z4evt2v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = memref.get_global @stv : memref<3xi32>
// CHECK-NEXT:     %1 = affine.load %0[0] : memref<3xi32>
// CHECK-NEXT:     return %1 : i32
// CHECK-NEXT:     }
