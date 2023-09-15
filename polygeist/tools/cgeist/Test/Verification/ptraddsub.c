// RUN: cgeist %s -O2 --function=* -S | FileCheck %s

int sub() {
    int data[10];
    int* start = &data[0];
    int* end = &data[7];
    return end - start;
}

int* add (int* in) {
	return &in[7];
}

// CHECK-LABEL:   func.func @sub() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant 4 : i64
// CHECK-NEXT:      %[[VAL_1:.*]] = memref.alloca() : memref<10xi32>
// CHECK-NEXT:      %[[VAL_2:.*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<10xi32>) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_2]][7] : (!llvm.ptr) -> !llvm.ptr, i32
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.ptrtoint %[[VAL_3]] : !llvm.ptr to i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.ptrtoint %[[VAL_2]] : !llvm.ptr to i64
// CHECK-NEXT:      %[[VAL_6:.*]] = arith.subi %[[VAL_4]], %[[VAL_5]] : i64
// CHECK-NEXT:      %[[VAL_7:.*]] = arith.divsi %[[VAL_6]], %[[VAL_0]] : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = arith.trunci %[[VAL_7]] : i64 to i32
// CHECK-NEXT:      return %[[VAL_8]] : i32
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @add(
// CHECK-SAME:                   %[[VAL_0:.*]]: memref<?xi32>) -> memref<?xi32> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 7 : index
// CHECK-NEXT:      %[[VAL_2:.*]] = "polygeist.subindex"(%[[VAL_0]], %[[VAL_1]]) : (memref<?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:      return %[[VAL_2]] : memref<?xi32>
// CHECK-NEXT:    }
