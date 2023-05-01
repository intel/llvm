// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s

// CHECK-LABEL:     func.func @_ZNKSt4lessIvEclIRiS2_EEDTltclsr3stdE7forwardIT_Efp_Eclsr3stdE7forwardIT0_Efp0_EEOS3_OS4_(
// CHECK-SAME:        %[[VAL_735:.*]]: !llvm.ptr<4> {llvm.align = 1 : i64, llvm.dereferenceable_or_null = 1 : i64, llvm.noundef},
// CHECK-SAME:        %[[VAL_736:.*]]: memref<?xi32, 4> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.noundef}, 
// CHECK-SAME:        %[[VAL_737:.*]]: memref<?xi32, 4> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.noundef})
// CHECK-SAME:          -> (i1 {llvm.noundef, llvm.zeroext})
// CHECK-DAG:         %[[VAL_738:.*]] = arith.constant 1 : i64
// CHECK-DAG:         %[[VAL_739:.*]] = llvm.alloca %[[VAL_738]] x !llvm.struct<(i8)> : (i64) -> !llvm.ptr
// CHECK-DAG:         %[[VAL_740:.*]] = llvm.alloca %[[VAL_738]] x !llvm.struct<(i8)> : (i64) -> !llvm.ptr
// CHECK-DAG:         %[[VAL_741:.*]] = llvm.alloca %[[VAL_738]] x !llvm.struct<(i8)> : (i64) -> !llvm.ptr
// CHECK-DAG:         %[[VAL_742:.*]] = llvm.alloca %[[VAL_738]] x !llvm.struct<(i8)> : (i64) -> !llvm.ptr
// CHECK-NEXT:        %[[VAL_743:.*]] = llvm.addrspacecast %[[VAL_740]] : !llvm.ptr to !llvm.ptr<4>
// CHECK-NEXT:        %[[VAL_744:.*]] = llvm.load %[[VAL_741]] : !llvm.ptr -> !llvm.struct<(i8)>
// CHECK-NEXT:        llvm.store %[[VAL_744]], %[[VAL_743]] : !llvm.struct<(i8)>, !llvm.ptr<4>
// CHECK-NEXT:        %[[VAL_745:.*]] = llvm.mlir.null : !llvm.ptr<4>
// CHECK-NEXT:        %[[VAL_746:.*]] = llvm.icmp "ne" %[[VAL_743]], %[[VAL_745]] : !llvm.ptr<4>
// CHECK-NEXT:        %[[VAL_747:.*]] = arith.select %[[VAL_746]], %[[VAL_743]], %[[VAL_745]] : !llvm.ptr<4>
// CHECK-NEXT:        %[[VAL_748:.*]] = llvm.addrspacecast %[[VAL_742]] : !llvm.ptr to !llvm.ptr<4>
// CHECK-NEXT:        call @_ZNSt17integral_constantIbLb0EEC1EOS0_(%[[VAL_748]], %[[VAL_747]]) : (!llvm.ptr<4>, !llvm.ptr<4>) -> ()
// CHECK-NEXT:        %[[VAL_749:.*]] = llvm.load %[[VAL_742]] : !llvm.ptr -> !llvm.struct<(i8)>
// CHECK-NEXT:        llvm.store %[[VAL_749]], %[[VAL_739]] : !llvm.struct<(i8)>, !llvm.ptr
// CHECK-NEXT:        %[[VAL_750:.*]] = call @_ZNSt4lessIvE6_S_cmpIRiS2_EEDcOT_OT0_St17integral_constantIbLb0EE(%[[VAL_736]], %[[VAL_737]], %[[VAL_739]]) : (memref<?xi32, 4>, memref<?xi32, 4>, !llvm.ptr) -> i1
// CHECK-NEXT:        return %[[VAL_750]] : i1
// CHECK-NEXT:      }

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
