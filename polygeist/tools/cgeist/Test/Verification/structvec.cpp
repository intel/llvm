// RUN: cgeist -O0 %s --function=* -S | FileCheck %s

#include <initializer_list>

struct structvec {
  using char2 = char __attribute__((ext_vector_type(2)));
  char2 v;

  structvec(std::initializer_list<char> l) {
    for (unsigned I = 0; I < 2; ++I) {
      v[I] = *(l.begin() + I) ? -1 : 0;
    }
  }
};

// CHECK-LABEL:  func.func @_Z10test_store9structvecic(%arg0: !llvm.struct<(vector<2xi8>)>, %arg1: i32, %arg2: i8)
// CHECK-NEXT:     %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK-NEXT:     %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(vector<2xi8>)> : (i64) -> !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:     %[[VAL_2:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(vector<2xi8>)> : (i64) -> !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:     llvm.store %[[VAL_3:.*]], %[[VAL_2]] : !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:     call @_ZN9structvecC1EOS_(%[[VAL_1]], %[[VAL_2]]) : (!llvm.ptr<struct<(vector<2xi8>)>>, !llvm.ptr<struct<(vector<2xi8>)>>) -> ()
// CHECK-NEXT:     %[[VAL_4:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:     return %[[VAL_4]] : !llvm.struct<(vector<2xi8>)>
// CHECK-NEXT:   }

// CHECK-LABEL:  func.func @_ZN9structvecC1EOS_(%arg0: !llvm.ptr<struct<(vector<2xi8>)>>, %arg1: !llvm.ptr<struct<(vector<2xi8>)>>)
// CHECK-NEXT:     %0 = llvm.getelementptr %arg1[0, 0] : (!llvm.ptr<struct<(vector<2xi8>)>>) -> !llvm.ptr<vector<2xi8>>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<vector<2xi8>>
// CHECK-NEXT:     %2 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr<struct<(vector<2xi8>)>>) -> !llvm.ptr<vector<2xi8>>
// CHECK-NEXT:     llvm.store %1, %2 : !llvm.ptr<vector<2xi8>>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

structvec test_store(structvec sv, int idx, char el) {
  sv.v[idx] = el;
  return sv;
}

// CHECK-LABEL: func.func @_Z9test_initv() -> !llvm.struct<(vector<2xi8>)>
// CHECK-DAG:     %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK-DAG:     %[[VAL_1:.*]] = arith.constant 1 : i8
// CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 0 : i8
// CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 1 : i64
// CHECK-DAG:     %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<(vector<2xi8>)> : (i64) -> !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-DAG:     %[[VAL_5:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<(vector<2xi8>)> : (i64) -> !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-DAG:     %[[VAL_6:.*]] = llvm.alloca %[[VAL_3]] x !llvm.array<2 x i8> : (i64) -> !llvm.ptr<array<2 x i8>>
// CHECK-DAG:     %[[VAL_7:.*]] = llvm.alloca %[[VAL_3]] x !llvm.array<2 x i8> : (i64) -> !llvm.ptr<array<2 x i8>>
// CHECK-DAG:     %[[VAL_8:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<(vector<2xi8>)> : (i64) -> !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:    %[[VAL_9:.*]] = llvm.getelementptr %[[VAL_7]][0, 0] : (!llvm.ptr<array<2 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:    llvm.store %[[VAL_2]], %[[VAL_9]] : !llvm.ptr<i8>
// CHECK-NEXT:    %[[VAL_10:.*]] = llvm.getelementptr %[[VAL_7]][0, 1] : (!llvm.ptr<array<2 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:    llvm.store %[[VAL_1]], %[[VAL_10]] : !llvm.ptr<i8>
// CHECK-NEXT:    %[[VAL_11:.*]] = llvm.load %[[VAL_7]] : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:    llvm.store %[[VAL_11]], %[[VAL_6]] : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:    %[[VAL_12:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<i8>, i64)>
// CHECK-NEXT:    %[[VAL_13:.*]] = llvm.getelementptr %[[VAL_6]][0, 0] : (!llvm.ptr<array<2 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:    %[[VAL_14:.*]] = llvm.insertvalue %[[VAL_13]], %[[VAL_12]][0] : !llvm.struct<(ptr<i8>, i64)>
// CHECK-NEXT:    %[[VAL_15:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_14]][1] : !llvm.struct<(ptr<i8>, i64)>
// CHECK-NEXT:    call @_ZN9structvecC1ESt16initializer_listIcE(%[[VAL_8]], %[[VAL_15]]) : (!llvm.ptr<struct<(vector<2xi8>)>>, !llvm.struct<(ptr<i8>, i64)>) -> ()
// CHECK-NEXT:    %[[VAL_16:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:    llvm.store %[[VAL_16]], %[[VAL_5]] : !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:    call @_ZN9structvecC1EOS_(%[[VAL_4]], %[[VAL_5]]) : (!llvm.ptr<struct<(vector<2xi8>)>>, !llvm.ptr<struct<(vector<2xi8>)>>) -> ()
// CHECK-NEXT:    %[[VAL_17:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:    return %[[VAL_17]] : !llvm.struct<(vector<2xi8>)>
// CHECK-NEXT:   }

structvec test_init() {
  structvec sv{0, 1};
  return sv;
}
