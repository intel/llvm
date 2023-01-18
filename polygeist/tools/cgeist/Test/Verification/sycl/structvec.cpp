// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s

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
// CHECK-NEXT:    %1 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr<struct<(vector<2xi8>)>>) -> !llvm.ptr<vector<2xi8>>
// CHECK-NEXT:    %2 = llvm.load %1 : !llvm.ptr<vector<2xi8>>
// CHECK-NEXT:    %3 = vector.insertelement %arg2, %2[%arg1 : i32] : vector<2xi8>
// CHECK-NEXT:    llvm.store %3, %1 : !llvm.ptr<vector<2xi8>>
// CHECK-NEXT:    %4 = llvm.addrspacecast %0 : !llvm.ptr<struct<(vector<2xi8>)>> to !llvm.ptr<struct<(vector<2xi8>)>, 4>
// CHECK-NEXT:    %5 = llvm.addrspacecast %arg0 : !llvm.ptr<struct<(vector<2xi8>)>> to !llvm.ptr<struct<(vector<2xi8>)>, 4>
// CHECK-NEXT:    call @_ZN9structvecC1EOS_(%4, %5) : (!llvm.ptr<struct<(vector<2xi8>)>, 4>, !llvm.ptr<struct<(vector<2xi8>)>, 4>) -> ()
// CHECK-NEXT:    %6 = llvm.load %0 : !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:    return %6 : !llvm.struct<(vector<2xi8>)>
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
// CHECK-DAG:     %2 = llvm.alloca %c1_i64 x !llvm.struct<(memref<?xi8, 4>, i64)> : (i64) -> !llvm.ptr<!llvm.struct<(memref<?xi8, 4>, i64)>>
// CHECK-DAG:     %3 = llvm.alloca %c1_i64 x !llvm.struct<(memref<?xi8, 4>, i64)> : (i64) -> !llvm.ptr<!llvm.struct<(memref<?xi8, 4>, i64)>>
// CHECK-DAG:     %4 = llvm.alloca %c1_i64 x !llvm.array<2 x i8> : (i64) -> !llvm.ptr<array<2 x i8>>
// CHECK-DAG:     %5 = llvm.alloca %c1_i64 x !llvm.array<2 x i8> : (i64) -> !llvm.ptr<array<2 x i8>>
// CHECK-DAG:     %6 = llvm.alloca %c1_i64 x !llvm.struct<(vector<2xi8>)> : (i64) -> !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:    %7 = llvm.getelementptr %5[0, 0] : (!llvm.ptr<array<2 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:    llvm.store %c0_i8, %7 : !llvm.ptr<i8>
// CHECK-NEXT:    %8 = llvm.getelementptr %5[0, 1] : (!llvm.ptr<array<2 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:    llvm.store %c1_i8, %8 : !llvm.ptr<i8>
// CHECK-NEXT:    %9 = llvm.addrspacecast %4 : !llvm.ptr<array<2 x i8>> to !llvm.ptr<array<2 x i8>, 4>
// CHECK-NEXT:    %10 = llvm.load %5 : !llvm.ptr<array<2 x i8>>
// CHECK-NEXT:    llvm.store %10, %9 : !llvm.ptr<array<2 x i8>, 4>
// CHECK-NEXT:    %11 = "polygeist.pointer2memref"(%9) : (!llvm.ptr<array<2 x i8>, 4>) -> memref<?xi8, 4 : i32>
// CHECK-NEXT:    %12 = llvm.getelementptr %3[0, 0] : (!llvm.ptr<!llvm.struct<(memref<?xi8, 4>, i64)>>) -> !llvm.ptr<memref<?xi8, 4>>
// CHECK-NEXT:    llvm.store %11, %12 : !llvm.ptr<memref<?xi8, 4>>
// CHECK-NEXT:    %13 = llvm.getelementptr %3[0, 1] : (!llvm.ptr<!llvm.struct<(memref<?xi8, 4>, i64)>>) -> !llvm.ptr<i64>
// CHECK-NEXT:    llvm.store %c2_i64, %13 : !llvm.ptr<i64>
// CHECK-NEXT:    %14 = llvm.load %3 : !llvm.ptr<!llvm.struct<(memref<?xi8, 4>, i64)>>
// CHECK-NEXT:    %15 = llvm.addrspacecast %6 : !llvm.ptr<struct<(vector<2xi8>)>> to !llvm.ptr<struct<(vector<2xi8>)>, 4>
// CHECK-NEXT:    llvm.store %14, %2 : !llvm.ptr<!llvm.struct<(memref<?xi8, 4>, i64)>>
// CHECK-NEXT:    call @_ZN9structvecC1ESt16initializer_listIcE(%15, %2) : (!llvm.ptr<struct<(vector<2xi8>)>, 4>, !llvm.ptr<!llvm.struct<(memref<?xi8, 4>, i64)>>) -> ()
// CHECK-NEXT:    %16 = llvm.load %6 : !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:    llvm.store %16, %1 : !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:    %17 = llvm.addrspacecast %0 : !llvm.ptr<struct<(vector<2xi8>)>> to !llvm.ptr<struct<(vector<2xi8>)>, 4>
// CHECK-NEXT:    %18 = llvm.addrspacecast %1 : !llvm.ptr<struct<(vector<2xi8>)>> to !llvm.ptr<struct<(vector<2xi8>)>, 4>
// CHECK-NEXT:    call @_ZN9structvecC1EOS_(%17, %18) : (!llvm.ptr<struct<(vector<2xi8>)>, 4>, !llvm.ptr<struct<(vector<2xi8>)>, 4>) -> ()
// CHECK-NEXT:    %19 = llvm.load %0 : !llvm.ptr<struct<(vector<2xi8>)>>
// CHECK-NEXT:    return %19 : !llvm.struct<(vector<2xi8>)>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func @_ZN9structvecC1ESt16initializer_listIcE(%arg0: !llvm.ptr<struct<(vector<2xi8>)>, 4> {llvm.align = 2 : i64, llvm.dereferenceable_or_null = 2 : i64, llvm.noundef}, %arg1: !llvm.ptr<!llvm.struct<(memref<?xi8, 4>, i64)>> {llvm.align = 8 : i64, llvm.byval = !llvm.struct<(memref<?xi8, 4>, i64)>, llvm.noundef})
// CHECK-DAG:     %c2 = arith.constant 2 : index
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c0_i8 = arith.constant 0 : i8
// CHECK-NEXT:    scf.for %arg2 = %c0 to %c2 step %c1 {
// CHECK-NEXT:      %0 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:      %1 = llvm.addrspacecast %arg1 : !llvm.ptr<!llvm.struct<(memref<?xi8, 4>, i64)>> to !llvm.ptr<!llvm.struct<(memref<?xi8, 4>, i64)>, 4>
// CHECK-NEXT:      %2 = func.call @_ZNKSt16initializer_listIcE5beginEv(%1) : (!llvm.ptr<!llvm.struct<(memref<?xi8, 4>, i64)>, 4>) -> memref<?xi8, 4>
// CHECK-NEXT:      %3 = arith.index_castui %0 : i32 to index
// CHECK-NEXT:      %4 = memref.load %2[%3] : memref<?xi8, 4>
// CHECK-NEXT:      %5 = arith.cmpi ne, %4, %c0_i8 : i8
// CHECK-NEXT:      %6 = arith.extui %5 : i1 to i32
// CHECK-NEXT:      %7 = arith.trunci %6 : i32 to i8
// CHECK-NEXT:      %8 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr<struct<(vector<2xi8>)>, 4>) -> !llvm.ptr<vector<2xi8>, 4>
// CHECK-NEXT:      %9 = llvm.load %8 : !llvm.ptr<vector<2xi8>, 4>
// CHECK-NEXT:      %10 = vector.insertelement %7, %9[%0 : i32] : vector<2xi8>
// CHECK-NEXT:      llvm.store %10, %8 : !llvm.ptr<vector<2xi8>, 4>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

SYCL_EXTERNAL structvec test_init() {
  structvec sv{0, 1};
  return sv;
}
