// RUN: cgeist -O0 -w %s --function=* -S | FileCheck %s

extern "C" {

struct pair {
    int x, y;
};
void sub0(pair& a);
void sub(pair& a) {
    a.x++;
}

void kernel_deriche() {
    pair a;
    a.x = 32;;
    pair &b = a;
    sub0(b);
}

}

// CHECK:   func @sub(%arg0: !llvm.ptr<struct<(i32, i32)>>)
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %0 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr<struct<(i32, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<i32>
// CHECK-NEXT:     %2 = arith.addi %1, %c1_i32 : i32
// CHECK-NEXT:     llvm.store %2, %0 : !llvm.ptr<i32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func @kernel_deriche()
// CHECK-DAG:      %c32_i32 = arith.constant 32 : i32
// CHECK-DAG:      %c1_i64 = arith.constant 1 : i64
// CHECK:          %0 = llvm.alloca %c1_i64 x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr<struct<(i32, i32)>>
// CHECK-NEXT:     %1 = llvm.getelementptr inbounds %0[0, 0] : (!llvm.ptr<struct<(i32, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %c32_i32, %1 : !llvm.ptr<i32>
// CHECK-NEXT:     call @sub0(%0) : (!llvm.ptr<struct<(i32, i32)>>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
