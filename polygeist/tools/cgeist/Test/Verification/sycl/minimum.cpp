// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s

// CHECK-LABEL:    func.func @_ZNKSt4lessIvEclIRiS2_EEDTltclsr3stdE7forwardIT_Efp_Eclsr3stdE7forwardIT0_Efp0_EEOS3_OS4_(%arg0: !llvm.ptr<struct<(i8)>, 4> {llvm.align = 1 : i64, llvm.dereferenceable_or_null = 1 : i64, llvm.noundef}, %arg1: memref<?xi32, 4> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.noundef}, %arg2: memref<?xi32, 4> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.noundef}) -> (i1 {llvm.noundef, llvm.zeroext})
// CHECK-DAG:        %c1_i64 = arith.constant 1 : i64
// CHECK-DAG:        %0 = llvm.alloca %c1_i64 x !llvm.struct<(i8)> : (i64) -> !llvm.ptr<struct<(i8)>>
// CHECK-DAG:        %1 = llvm.alloca %c1_i64 x !llvm.struct<(i8)> : (i64) -> !llvm.ptr<struct<(i8)>>
// CHECK-DAG:        %2 = llvm.alloca %c1_i64 x !llvm.struct<(i8)> : (i64) -> !llvm.ptr<struct<(i8)>>
// CHECK-DAG:        %3 = llvm.alloca %c1_i64 x !llvm.struct<(i8)> : (i64) -> !llvm.ptr<struct<(i8)>>
// CHECK-NEXT:       %4 = llvm.addrspacecast %1 : !llvm.ptr<struct<(i8)>> to !llvm.ptr<struct<(i8)>, 4>
// CHECK-NEXT:       %5 = llvm.load %2 : !llvm.ptr<struct<(i8)>>
// CHECK-NEXT:       llvm.store %5, %4 : !llvm.ptr<struct<(i8)>, 4>
// CHECK-NEXT:       %6 = llvm.mlir.null : !llvm.ptr<struct<(i8)>, 4>
// CHECK-NEXT:       %7 = llvm.icmp "ne" %4, %6 : !llvm.ptr<struct<(i8)>, 4>
// CHECK-NEXT:       %8 = arith.select %7, %4, %6 : !llvm.ptr<struct<(i8)>, 4>
// CHECK-NEXT:       %9 = llvm.addrspacecast %3 : !llvm.ptr<struct<(i8)>> to !llvm.ptr<struct<(i8)>, 4>
// CHECK-NEXT:       call @_ZNSt17integral_constantIbLb0EEC1EOS0_(%9, %8) : (!llvm.ptr<struct<(i8)>, 4>, !llvm.ptr<struct<(i8)>, 4>) -> ()
// CHECK-NEXT:       %10 = llvm.load %3 : !llvm.ptr<struct<(i8)>>
// CHECK-NEXT:       llvm.store %10, %0 : !llvm.ptr<struct<(i8)>>
// CHECK-NEXT:       %11 = call @_ZNSt4lessIvE6_S_cmpIRiS2_EEDcOT_OT0_St17integral_constantIbLb0EE(%arg1, %arg2, %0) : (memref<?xi32, 4>, memref<?xi32, 4>, !llvm.ptr<struct<(i8)>>) -> i1
// CHECK-NEXT:       return %11 : i1
// CHECK-NEXT:     }

#include <sycl/sycl.hpp>
using namespace sycl;
static constexpr unsigned N = 8;

void parallel_for_nd_item(std::array<int, N> &A, queue q) {
  nd_range<1> ndRange(N /*globalSize*/, 2 /*localSize*/);

  {
    auto buf = buffer<int, 1>{A.data(), N};
    q.submit([&](handler &cgh) {
      auto A = buf.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class kernel_parallel_for_nd_item>(ndRange, [=](nd_item<1> NdItem) {
        ext::oneapi::sub_group sg = NdItem.get_sub_group();
        A[0] = ext::oneapi::exclusive_scan(sg, 0, 0, ext::oneapi::minimum<>());
      });
    });
  }
}
