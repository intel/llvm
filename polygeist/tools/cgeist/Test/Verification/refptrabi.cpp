// RUN: cgeist  -O0 -w %s  --function=ll -S | FileCheck %s

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

// CHECK-LABEL:   func.func @ll(
// CHECK-SAME:                  %[[VAL_0:.*]]: !llvm.ptr) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(i16)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      call @_ZN4HalfC1ERKS_(%[[VAL_2]], %[[VAL_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.struct<(i16)>
// CHECK-NEXT:      %[[VAL_4:.*]] = call @thing(%[[VAL_3]]) : (!llvm.struct<(i16)>) -> f32
// CHECK-NEXT:      return %[[VAL_4]] : f32
// CHECK-NEXT:    }
