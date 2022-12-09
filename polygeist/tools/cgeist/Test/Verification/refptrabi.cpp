// RUN: cgeist -O0 -w %s  --function=ll -S | FileCheck %s

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
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(i16)> : (i64) -> !llvm.ptr<struct<(i16)>>
// CHECK-NEXT:     %1 = llvm.bitcast %arg0 : !llvm.ptr<i8> to !llvm.ptr<struct<(i16)>>
// CHECK-NEXT:     call @_ZN4HalfC1ERKS_(%0, %1) : (!llvm.ptr<struct<(i16)>>, !llvm.ptr<struct<(i16)>>) -> ()
// CHECK-NEXT:     %2 = llvm.load %0 : !llvm.ptr<struct<(i16)>>
// CHECK-NEXT:     %3 = call @thing(%2) : (!llvm.struct<(i16)>) -> f32
// CHECK-NEXT:     return %3 : f32
// CHECK-NEXT:   }
