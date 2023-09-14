// RUN: cgeist %s --function=* -S | FileCheck %s

int c2i(char x) {
    return x;
}

unsigned int c2ui(char x) {
    return x;
}

int uc2i(unsigned char x) {
    return x;
}

unsigned int uc2ui(unsigned char x) {
    return x;
}

// CHECK:   func @c2i(%arg0: i8) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = arith.extsi %arg0 : i8 to i32
// CHECK-NEXT:     return %0 : i32
// CHECK-NEXT:   }
// CHECK:   func @c2ui(%arg0: i8) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = arith.extsi %arg0 : i8 to i32
// CHECK-NEXT:     return %0 : i32
// CHECK-NEXT:   }
// CHECK:   func @uc2i(%arg0: i8) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = arith.extui %arg0 : i8 to i32
// CHECK-NEXT:     return %0 : i32
// CHECK-NEXT:   }
// CHECK:   func @uc2ui(%arg0: i8) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = arith.extui %arg0 : i8 to i32
// CHECK-NEXT:     return %0 : i32
// CHECK-NEXT:   }
