// RUN: cgeist %s --function=ll -S | FileCheck %s

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
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x1xi16>
// CHECK-NEXT:     %1 = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<i16>
// CHECK-NEXT:     %2 = llvm.load %1 : !llvm.ptr<i16>
// CHECK-NEXT:     affine.store %2, %0[0, 0] : memref<1x1xi16>
// CHECK-NEXT:     %3 = memref.cast %0 : memref<1x1xi16> to memref<?x1xi16>
// CHECK-NEXT:     %4 = call @thing(%3) : (memref<?x1xi16>) -> f32
// CHECK-NEXT:     return %4 : f32
// CHECK-NEXT:   }
