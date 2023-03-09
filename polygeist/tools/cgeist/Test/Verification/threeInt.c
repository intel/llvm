// RUN: cgeist -O0 -w %s --function=struct_pass_all_same -S | FileCheck %s

typedef struct {
  int a, b, c;
} threeInt;

int struct_pass_all_same(threeInt* a) {
  return a->b;
}

// CHECK:  func @struct_pass_all_same(%arg0: !llvm.ptr<struct<(i32, i32, i32)>>) -> i32
// CHECK-NEXT:    %0 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr<struct<(i32, i32, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:    %1 = llvm.load %0 : !llvm.ptr<i32>
// CHECK-NEXT:    return %1 : i32
// CHECK-NEXT:  }
