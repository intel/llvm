// RUN: clang++ -O1 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -w | FileCheck %s
// RUN: clang++ -O2 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -w | FileCheck %s
// RUN: clang++ -O3 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -w | FileCheck %s

#include <sycl/sycl.hpp>

template <typename... Args>
void keep(Args&&...);

template <int Dimensions>
void nd_range(sycl::range<Dimensions> globalSize,
              sycl::range<Dimensions> localSize) {
  sycl::nd_range<Dimensions> ndr(globalSize, localSize);
  keep(ndr);
}

template <int Dimensions>
void nd_range_offset(sycl::range<Dimensions> globalSize,
                     sycl::range<Dimensions> localSize,
                     sycl::id<Dimensions> offset) {
  sycl::nd_range<Dimensions> ndr(globalSize, localSize, offset);
  keep(ndr);
}

template <int Dimensions>
void nd_range_copy(const sycl::nd_range<Dimensions> &other) {
  sycl::nd_range<Dimensions> ndr(other);
  keep(ndr);
}

template <int Dimensions>
void nd_range_move(sycl::nd_range<Dimensions> &&other) {
  sycl::nd_range<Dimensions> ndr(std::move(other));
  keep(ndr);
}

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z8nd_rangeILi1EEvN4sycl3_V15rangeIXT_EEES3_(
// CHECK-SAME:                                                                     %[[VAL_0:.*]]: i64,
// CHECK-SAME:                                                                     %[[VAL_1:.*]]: i64)
// CHECK-DAG:       %[[VAL_2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:       %[[VAL_3:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<[[RANGE1:.*]]> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<[[RANGE1]]> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<[[ID1:.*]]> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<[[ND1:.*]]> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 24, %[[VAL_8]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_5]], %[[VAL_0]]) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_6]], %[[VAL_1]]) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_7]], %[[VAL_3]]) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_8]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]]) {type = !sycl_nd_range_1_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V18nd_rangeILi1EEEEEvDpOT_(%[[VAL_8]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 24, %[[VAL_8]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void nd_range(sycl::range<1>, sycl::range<1>);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z8nd_rangeILi2EEvN4sycl3_V15rangeIXT_EEES3_(
// CHECK-SAME:                                                                     %[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64)
// CHECK-DAG:       %[[VAL_4:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:       %[[VAL_5:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK-DAG:       %[[VAL_6:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.alloca %[[VAL_4]] x !llvm.struct<[[RANGE2:.*]]> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.alloca %[[VAL_4]] x !llvm.struct<[[RANGE2]]> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.alloca %[[VAL_4]] x !llvm.struct<[[ND2:.*]]> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 48, %[[VAL_10]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_8]], %[[VAL_0]], %[[VAL_1]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_9]], %[[VAL_2]], %[[VAL_3]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_10]], %[[VAL_8]], %[[VAL_9]]) {type = !sycl_nd_range_2_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      %[[VAL_11:.*]] = llvm.getelementptr inbounds %[[VAL_10]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<[[ND2]]>
// CHECK-NEXT:      "llvm.intr.memset"(%[[VAL_11]], %[[VAL_5]], %[[VAL_6]]) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V18nd_rangeILi2EEEEEvDpOT_(%[[VAL_10]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 48, %[[VAL_10]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void nd_range(sycl::range<2>, sycl::range<2>);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z8nd_rangeILi3EEvN4sycl3_V15rangeIXT_EEES3_(
// CHECK-SAME:                                                                     %[[VAL_0:.*]]: !llvm.ptr {{{.*}}},
// CHECK-SAME:                                                                     %[[VAL_1:.*]]: !llvm.ptr {{{.*}}})
// CHECK-DAG:       %[[VAL_2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-DAG:       %[[VAL_3:.*]] = llvm.mlir.constant(24 : i64) : i64
// CHECK-DAG:       %[[VAL_5:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<[[ND3:.*]]> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 72, %[[VAL_6]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_6]], %[[VAL_0]], %[[VAL_1]]) {type = !sycl_nd_range_3_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.getelementptr inbounds %[[VAL_6]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<[[ND3]]>
// CHECK-NEXT:      "llvm.intr.memset"(%[[VAL_7]], %[[VAL_5]], %[[VAL_3]]) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V18nd_rangeILi3EEEEEvDpOT_(%[[VAL_6]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 72, %[[VAL_6]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void nd_range(sycl::range<3>, sycl::range<3>);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z15nd_range_offsetILi1EEvN4sycl3_V15rangeIXT_EEES3_NS1_2idIXT_EEE(
// CHECK-SAME:                                                                                           %[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64)
// CHECK-DAG:       %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<[[RANGE1]]> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<[[RANGE1]]> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<[[ID1]]> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<[[ND1]]> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 24, %[[VAL_8]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_5]], %[[VAL_0]]) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_6]], %[[VAL_1]]) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_7]], %[[VAL_2]]) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_8]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]]) {type = !sycl_nd_range_1_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V18nd_rangeILi1EEEEEvDpOT_(%[[VAL_8]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 24, %[[VAL_8]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void nd_range_offset(sycl::range<1>, sycl::range<1>, sycl::id<1>);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z15nd_range_offsetILi2EEvN4sycl3_V15rangeIXT_EEES3_NS1_2idIXT_EEE(
// CHECK-SAME:                                                                                           %[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: i64, %[[VAL_5:.*]]: i64)
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<[[RANGE2]]> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<[[RANGE2]]> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<[[ID2:.*]]> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.alloca %[[VAL_6]] x !llvm.struct<[[ND2]]> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 48, %[[VAL_10]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_7]], %[[VAL_0]], %[[VAL_1]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_8]], %[[VAL_2]], %[[VAL_3]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_9]], %[[VAL_4]], %[[VAL_5]]) {type = !sycl_id_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_10]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]]) {type = !sycl_nd_range_2_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V18nd_rangeILi2EEEEEvDpOT_(%[[VAL_10]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 48, %[[VAL_10]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void nd_range_offset(sycl::range<2>, sycl::range<2>, sycl::id<2>);


// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z15nd_range_offsetILi3EEvN4sycl3_V15rangeIXT_EEES3_NS1_2idIXT_EEE(
// CHECK-SAME:                                                                                           %[[VAL_0:.*]]: !llvm.ptr {{{.*}}}, %[[VAL_1:.*]]: !llvm.ptr {{{.*}}}, %[[VAL_2:.*]]: !llvm.ptr {{{.*}}})
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<[[ND3]]> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 72, %[[VAL_4]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_4]], %[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {type = !sycl_nd_range_3_} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V18nd_rangeILi3EEEEEvDpOT_(%[[VAL_4]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 72, %[[VAL_4]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void nd_range_offset(sycl::range<3>, sycl::range<3>, sycl::id<3>);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z13nd_range_copyILi1EEvRKN4sycl3_V18nd_rangeIXT_EEE(
// CHECK-SAME:                                                                             %[[VAL_0:.*]]: !llvm.ptr {{{.*}}})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<[[ND1]]> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 24, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_0]]) {type = !sycl_nd_range_1_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V18nd_rangeILi1EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 24, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void nd_range_copy(const sycl::nd_range<1> &);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z13nd_range_copyILi2EEvRKN4sycl3_V18nd_rangeIXT_EEE(
// CHECK-SAME:                                                                             %[[VAL_0:.*]]: !llvm.ptr {{{.*}}})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<[[ND2]]> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 48, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_0]]) {type = !sycl_nd_range_2_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V18nd_rangeILi2EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 48, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void nd_range_copy(const sycl::nd_range<2> &);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z13nd_range_copyILi3EEvRKN4sycl3_V18nd_rangeIXT_EEE(
// CHECK-SAME:                                                                             %[[VAL_0:.*]]: !llvm.ptr {{{.*}}})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<[[ND3]]> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 72, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_0]]) {type = !sycl_nd_range_3_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V18nd_rangeILi3EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 72, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void nd_range_copy(const sycl::nd_range<3> &);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z13nd_range_moveILi1EEvON4sycl3_V18nd_rangeIXT_EEE(
// CHECK-SAME:                                                                            %[[VAL_0:.*]]: !llvm.ptr {{{.*}}})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<[[ND1]]> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 24, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_0]]) {type = !sycl_nd_range_1_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V18nd_rangeILi1EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 24, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void nd_range_move(sycl::nd_range<1> &&);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z13nd_range_moveILi2EEvON4sycl3_V18nd_rangeIXT_EEE(
// CHECK-SAME:                                                                            %[[VAL_0:.*]]: !llvm.ptr {{{.*}}})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<[[ND2]]> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 48, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_0]]) {type = !sycl_nd_range_2_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V18nd_rangeILi2EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 48, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void nd_range_move(sycl::nd_range<2> &&);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z13nd_range_moveILi3EEvON4sycl3_V18nd_rangeIXT_EEE(
// CHECK-SAME:                                                                            %[[VAL_0:.*]]: !llvm.ptr {{{.*}}})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<[[ND3]]> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 72, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_0]]) {type = !sycl_nd_range_3_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V18nd_rangeILi3EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 72, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void nd_range_move(sycl::nd_range<3> &&);
