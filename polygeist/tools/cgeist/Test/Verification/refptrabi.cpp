// RUN: cgeist %s -O2 --function=ll -S | FileCheck %s

struct alignas(2) Half {
  unsigned short x;

  Half() = default;
};

extern "C" {

float thing(Half);

float ll(void* data) {
    return thing(*(Half*)data);
}

}

// CHECK:   func @ll(%arg0: !llvm.ptr<i8>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %alloca = memref.alloca() : memref<1x1xi16>
// CHECK-NEXT:     %alloca_0 = memref.alloca() : memref<1x1xi16>
// CHECK-NEXT:     %cast = memref.cast %alloca_0 : memref<1x1xi16> to memref<?x1xi16>
// CHECK-NEXT:     %0 = "polygeist.pointer2memref"(%arg0) : (!llvm.ptr<i8>) -> memref<?x1xi16>
// CHECK-NEXT:     call @_ZN4HalfC1ERKS_(%cast, %0) : (memref<?x1xi16>, memref<?x1xi16>) -> ()
// CHECK-NEXT:     %1 = affine.load %alloca_0[0, 0] : memref<1x1xi16>
// CHECK-NEXT:     affine.store %1, %alloca[0, 0] : memref<1x1xi16>
// CHECK-NEXT:     %cast_1 = memref.cast %alloca : memref<1x1xi16> to memref<?x1xi16>
// CHECK-NEXT:     %2 = call @thing(%cast_1) : (memref<?x1xi16>) -> f32
// CHECK-NEXT:     return %2 : f32
// CHECK-NEXT:   }
