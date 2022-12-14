// RUN: cgeist -O0 %s --function=* -S | FileCheck %s

typedef float float8 __attribute__((ext_vector_type(8)));

struct structvec {
  float8 v;
};

// CHECK-LABEL:  func.func @_Z14test_structvec9structvecii(%arg0: !llvm.ptr<struct<(vector<8xf32>)>>, %arg1: i32, %arg2: i32) -> !llvm.struct<(vector<8xf32>)>
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(vector<8xf32>)> : (i64) -> !llvm.ptr<struct<(vector<8xf32>)>>
// CHECK-NEXT:     call @_ZN9structvecC1EOS_(%0, %arg0) : (!llvm.ptr<struct<(vector<8xf32>)>>, !llvm.ptr<struct<(vector<8xf32>)>>) -> ()
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<struct<(vector<8xf32>)>>
// CHECK-NEXT:     return %1 : !llvm.struct<(vector<8xf32>)>
// CHECK-NEXT:   }

// CHECK-LABEL:  func.func @_ZN9structvecC1EOS_(%arg0: !llvm.ptr<struct<(vector<8xf32>)>>, %arg1: !llvm.ptr<struct<(vector<8xf32>)>>)
// CHECK-NEXT:     %0 = llvm.getelementptr %arg1[0, 0] : (!llvm.ptr<struct<(vector<8xf32>)>>) -> !llvm.ptr<vector<8xf32>>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<vector<8xf32>>
// CHECK-NEXT:     %2 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr<struct<(vector<8xf32>)>>) -> !llvm.ptr<vector<8xf32>>
// CHECK-NEXT:     llvm.store %1, %2 : !llvm.ptr<vector<8xf32>>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

structvec test_structvec(structvec sv, int idx, int el) {
  sv.v[idx] = el;
  return sv;
}
