// RUN: clang++ -O1 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xcgeist -no-early-drop-host-code -w | FileCheck %s
// RUN: clang++ -O2 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xcgeist -no-early-drop-host-code -w | FileCheck %s
// RUN: clang++ -O3 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -Xcgeist -no-early-drop-host-code -w | FileCheck %s

#include <sycl/sycl.hpp>

template <typename... Args>
void keep(Args&&...);

// CHECK-LABEL:   llvm.func local_unnamed_addr @_Z5rangem(
// CHECK-SAME:                         %[[VAL_0:.*]]: i64 {llvm.noundef})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 8, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_0]]) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V15rangeILi1EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 8, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
void range(std::size_t i) {
  sycl::range<1> range(i);
  keep(range);
}

// CHECK-LABEL:   llvm.func local_unnamed_addr @_Z5rangemm(
// CHECK-SAME:                          %[[VAL_0:.*]]: i64 {llvm.noundef},
// CHECK-SAME:                          %[[VAL_1:.*]]: i64 {llvm.noundef})
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"class.sycl::_V1::range.0", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 16, %[[VAL_3]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_3]], %[[VAL_0]], %[[VAL_1]]) {type = !sycl_range_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V15rangeILi2EEEEEvDpOT_(%[[VAL_3]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 16, %[[VAL_3]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
void range(std::size_t i, std::size_t j) {
  sycl::range<2> range(i, j);
  keep(range);
}

// CHECK-LABEL:   llvm.func local_unnamed_addr @_Z5rangemmm(
// CHECK-SAME:                           %[[VAL_0:.*]]: i64 {llvm.noundef}, %[[VAL_1:.*]]: i64 {llvm.noundef}, %[[VAL_2:.*]]: i64 {llvm.noundef})
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 24, %[[VAL_4]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_4]], %[[VAL_0]], %[[VAL_1]], %[[VAL_1]]) {type = !sycl_range_3_} : (!llvm.ptr, i64, i64, i64) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V15rangeILi3EEEEEvDpOT_(%[[VAL_4]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 24, %[[VAL_4]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
void range(std::size_t i, std::size_t j, std::size_t k) {
  sycl::range<3> range(i, j, j);
  keep(range);
}

template <int Dimensions>
void range(const sycl::range<Dimensions> &other) {
  sycl::range<Dimensions> range(other);
  keep(range);
}

template <int Dimensions>
void range(sycl::range<Dimensions> &&other) {
  sycl::range<Dimensions> range(std::move(other));
  keep(range);
}

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z5rangeILi1EEvRKN4sycl3_V15rangeIXT_EEE(
// CHECK-SAME:                                                                 %[[VAL_0:.*]]: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 8, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_0]] {alignment = 8 : i64} : !llvm.ptr -> i64
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_3]]) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V15rangeILi1EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 8, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void range<1>(const sycl::range<1> &);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z5rangeILi2EEvRKN4sycl3_V15rangeIXT_EEE(
// CHECK-SAME:                                                                 %[[VAL_0:.*]]: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::range.0", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 16, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_0]]) {type = !sycl_range_2_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V15rangeILi2EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 16, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void range<2>(const sycl::range<2> &);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z5rangeILi3EEvRKN4sycl3_V15rangeIXT_EEE(
// CHECK-SAME:                                                                 %[[VAL_0:.*]]: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 24, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_0]]) {type = !sycl_range_3_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V15rangeILi3EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 24, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void range<3>(const sycl::range<3> &);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z5rangeILi1EEvON4sycl3_V15rangeIXT_EEE(
// CHECK-SAME:                                                                %[[VAL_0:.*]]: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::range", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 8, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_0]] {alignment = 8 : i64} : !llvm.ptr -> i64
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_3]]) {type = !sycl_range_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V15rangeILi1EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 8, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void range<1>(sycl::range<1> &&);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z5rangeILi2EEvON4sycl3_V15rangeIXT_EEE(
// CHECK-SAME:                                                                %[[VAL_0:.*]]: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::range.0", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 16, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_0]]) {type = !sycl_range_2_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V15rangeILi2EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 16, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void range<2>(sycl::range<2> &&);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z5rangeILi3EEvON4sycl3_V15rangeIXT_EEE(
// CHECK-SAME:                                                                %[[VAL_0:.*]]: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::range.2", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 24, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_0]]) {type = !sycl_range_3_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V15rangeILi3EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 24, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void range<3>(sycl::range<3> &&);
