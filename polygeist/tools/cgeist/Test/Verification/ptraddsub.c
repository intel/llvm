// RUN: cgeist %s --function=* -S | FileCheck %s

int sub() {
    int data[10];
    int* start = &data[0];
    int* end = &data[7];
    return end - start;
}

int* add (int* in) {
	return &in[7];
}

// CHECK-LABEL:   func.func @sub() -> i32
// CHECK-NEXT:      %[[VAL_0:.*]] = arith.constant 4 : i64
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 7 : index
// CHECK-NEXT:      %[[VAL_2:.*]] = memref.alloca() : memref<10xi32>
// CHECK-NEXT:      %[[VAL_3:.*]] = "polygeist.subindex"(%[[VAL_2]], %[[VAL_1]]) : (memref<10xi32>, index) -> memref<?xi32>
// CHECK-NEXT:      %[[VAL_4:.*]] = "polygeist.memref2pointer"(%[[VAL_3]]) : (memref<?xi32>) -> !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_5:.*]] = "polygeist.memref2pointer"(%[[VAL_2]]) : (memref<10xi32>) -> !llvm.ptr<i32>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.ptrtoint %[[VAL_5]] : !llvm.ptr<i32> to i64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.ptrtoint %[[VAL_4]] : !llvm.ptr<i32> to i64
// CHECK-NEXT:      %[[VAL_8:.*]] = arith.subi %[[VAL_7]], %[[VAL_6]] : i64
// CHECK-NEXT:      %[[VAL_9:.*]] = arith.divsi %[[VAL_8]], %[[VAL_0]] : i64
// CHECK-NEXT:      %[[VAL_10:.*]] = arith.trunci %[[VAL_9]] : i64 to i32
// CHECK-NEXT:      return %[[VAL_10]] : i32
// CHECK-NEXT:    }

// CHECK:   func @add(%arg0: memref<?xi32>) -> memref<?xi32> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c7 = arith.constant 7 : index
// CHECK-NEXT:     %0 = "polygeist.subindex"(%arg0, %c7) : (memref<?xi32>, index) -> memref<?xi32>
// CHECK-NEXT:     return %0 : memref<?xi32>
// CHECK-NEXT:   }
