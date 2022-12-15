// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s
// XFAIL: *

#include <initializer_list>
#include <sycl/sycl.hpp>

struct structvec {
  using char2 = char __attribute__((ext_vector_type(2)));
  char2 v;

  structvec(std::initializer_list<char> l) {
    for (unsigned I = 0; I < 2; ++I) {
      v[I] = *(l.begin() + I) ? -1 : 0;
    }
  }
};

// CHECK-LABEL: func.func @_Z10test_store9structvecic(%arg0: !llvm.ptr<struct<(vector<2xi8>)>> {llvm.align = 2 : i64, llvm.byval = !llvm.struct<(vector<2xi8>)>, llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i8 {llvm.noundef, llvm.signext}) -> !llvm.struct<(vector<2xi8>)>
// CHECK-NEXT:    %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:    %0 = llvm.alloca %c1_i64 x !llvm.struct<(vector<2xi8>)> : (i64) -> !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:    %1 = llvm.addrspacecast %0 : !llvm.ptr<struct<(vector<2xi8>)>> to !llvm.ptr<struct<(vector<2xi8>)>, 4>
// CHECK-NEXT:    %2 = llvm.addrspacecast %arg0 : !llvm.ptr<struct<(vector<2xi8>)>> to !llvm.ptr<struct<(vector<2xi8>)>, 4>
// CHECK-NEXT:    call @_ZN9structvecC1EOS_(%1, %2) : (!llvm.ptr<struct<(vector<2xi8>)>, 4>, !llvm.ptr<struct<(vector<2xi8>)>, 4>) -> ()
// CHECK-NEXT:    %3 = llvm.load %0 : !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:    return %3 : !llvm.struct<(vector<2xi8>)>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func @_ZN9structvecC1EOS_(%arg0: !llvm.ptr<struct<(vector<2xi8>)>, 4> {llvm.align = 2 : i64, llvm.dereferenceable_or_null = 2 : i64, llvm.noundef}, %arg1: !llvm.ptr<struct<(vector<2xi8>)>, 4> {llvm.align = 2 : i64, llvm.dereferenceable = 2 : i64, llvm.noundef})
// CHECK-NEXT:    %0 = llvm.getelementptr %arg1[0, 0] : (!llvm.ptr<struct<(vector<2xi8>)>, 4>) -> !llvm.ptr<vector<2xi8>, 4>
// CHECK-NEXT:    %1 = llvm.load %0 : !llvm.ptr<vector<2xi8>, 4>
// CHECK-NEXT:    %2 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr<struct<(vector<2xi8>)>, 4>) -> !llvm.ptr<vector<2xi8>, 4>
// CHECK-NEXT:    llvm.store %1, %2 : !llvm.ptr<vector<2xi8>, 4>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

SYCL_EXTERNAL structvec test_store(structvec sv, int idx, char el) {
  sv.v[idx] = el;
  return sv;
}

// CHECK-LABEL: func.func @_Z9test_initv() -> !llvm.struct<(vector<2xi8>)>
// CHECK-DAG:     %c2_i64 = arith.constant 2 : i64
// CHECK-DAG:     %c1_i8 = arith.constant 1 : i8
// CHECK-DAG:     %c0_i8 = arith.constant 0 : i8
// CHECK-DAG:     %c1_i64 = arith.constant 1 : i64
// CHECK-DAG:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(vector<2xi8>)> : (i64) -> !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-DAG:     %1 = llvm.alloca %c1_i64 x !llvm.struct<(vector<2xi8>)> : (i64) -> !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-DAG:     %2 = llvm.alloca %c1_i64 x !llvm.struct<(ptr<i8, 4>, i64)> : (i64) -> !llvm.ptr<struct<(ptr<i8, 4>, i64)>>
// CHECK-DAG:     %3 = llvm.alloca %c1_i64 x !llvm.array<2 x i8> : (i64) -> !llvm.ptr<array<2 x i8>>
// CHECK-DAG:     %4 = llvm.alloca %c1_i64 x !llvm.array<2 x i8> : (i64) -> !llvm.ptr<array<2 x i8>>
// CHECK-DAG:     %5 = llvm.alloca %c1_i64 x !llvm.struct<(vector<2xi8>)> : (i64) -> !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:    %6 = llvm.getelementptr %4[0, 0] : (!llvm.ptr<array<2 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:    llvm.store %c0_i8, %6 : !llvm.ptr<i8>
// CHECK-NEXT:    %7 = llvm.getelementptr %4[0, 1] : (!llvm.ptr<array<2 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:    llvm.store %c1_i8, %7 : !llvm.ptr<i8>
// CHECK-NEXT:    %8 = llvm.addrspacecast %3 : !llvm.ptr<array<2 x i8>> to !llvm.ptr<array<2 x i8>, 4>
// CHECK-NEXT:    %9 = llvm.load %4 : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:    llvm.store %9, %8 : !llvm.ptr<array<2 x i8>, 4>
// CHECK-NEXT:    %10 = llvm.mlir.undef : !llvm.struct<(ptr<i8, 4>, i64)>
// CHECK-NEXT:    %11 = llvm.getelementptr %8[0, 0] : (!llvm.ptr<array<2 x i8>, 4>) -> !llvm.ptr<i8, 4>
// CHECK-NEXT:    %12 = llvm.insertvalue %11, %10[0] : !llvm.struct<(ptr<i8, 4>, i64)>
// CHECK-NEXT:    %13 = llvm.insertvalue %c2_i64, %12[1] : !llvm.struct<(ptr<i8, 4>, i64)>
// CHECK-NEXT:    %14 = llvm.addrspacecast %5 : !llvm.ptr<struct<(vector<2xi8>)>> to !llvm.ptr<struct<(vector<2xi8>)>, 4>
// CHECK-NEXT:    llvm.store %13, %2 : !llvm.ptr<struct<(ptr<i8, 4>, i64)>>
// CHECK-NEXT:    call @_ZN9structvecC1ESt16initializer_listIcE(%14, %2) : (!llvm.ptr<struct<(vector<2xi8>)>, 4>, !llvm.ptr<struct<(ptr<i8, 4>, i64)>>) -> ()
// CHECK-NEXT:    %15 = llvm.load %5 : !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:    llvm.store %15, %1 : !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:    %16 = llvm.addrspacecast %0 : !llvm.ptr<struct<(vector<2xi8>)>> to !llvm.ptr<struct<(vector<2xi8>)>, 4>
// CHECK-NEXT:    %17 = llvm.addrspacecast %1 : !llvm.ptr<struct<(vector<2xi8>)>> to !llvm.ptr<struct<(vector<2xi8>)>, 4>
// CHECK-NEXT:    call @_ZN9structvecC1EOS_(%16, %17) : (!llvm.ptr<struct<(vector<2xi8>)>, 4>, !llvm.ptr<struct<(vector<2xi8>)>, 4>) -> ()
// CHECK-NEXT:    %18 = llvm.load %0 : !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:    return %18 : !llvm.struct<(vector<2xi8>)>
// CHECK-NEXT:   }

SYCL_EXTERNAL structvec test_init() {
  structvec sv{0, 1};
  return sv;
}
