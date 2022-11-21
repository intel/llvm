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
// CHECK-NEXT:     %0 = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<i16>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<i16>
// CHECK-NEXT:     affine.store %1, %alloca[0, 0] : memref<1x1xi16>
// CHECK-NEXT:     %cast = memref.cast %alloca : memref<1x1xi16> to memref<?x1xi16>
// CHECK-NEXT:     %2 = call @thing(%cast) : (memref<?x1xi16>) -> f32
// CHECK-NEXT:     return %2 : f32
// CHECK-NEXT:   }
