// RUN: clang++ -O1 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -w | FileCheck %s
// RUN: clang++ -O2 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -w | FileCheck %s
// RUN: clang++ -O3 %s -S -emit-mlir -o - -fsycl -fsycl-raise-host -w | FileCheck %s

#include <sycl/sycl.hpp>

template <typename... Args>
void keep(Args&&...);

template <int Dimensions>
void id() {
  sycl::id<Dimensions> id;
  keep(id);
}

// CHECK-LABEL:   llvm.func local_unnamed_addr @_Z2idm(
// CHECK-SAME:                      %[[VAL_0:.*]]: i64 {llvm.noundef})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:           llvm.intr.lifetime.start 8, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:           sycl.host.constructor(%[[VAL_2]], %[[VAL_0]]) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:           llvm.call @_Z4keepIJRN4sycl3_V12idILi1EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:           llvm.intr.lifetime.end 8, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:           llvm.return
// CHECK-NEXT:         }
void id(std::size_t i) {
  sycl::id<1> id(i);
  keep(id);
}

// CHECK-LABEL:   llvm.func local_unnamed_addr @_Z2idmm(
// CHECK-SAME:                       %[[VAL_0:.*]]: i64 {llvm.noundef},
// CHECK-SAME:                       %[[VAL_1:.*]]: i64 {llvm.noundef})
// CHECK-NEXT:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<"class.sycl::_V1::id.0", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:           llvm.intr.lifetime.start 16, %[[VAL_3]] : !llvm.ptr
// CHECK-NEXT:           sycl.host.constructor(%[[VAL_3]], %[[VAL_0]], %[[VAL_1]]) {type = !sycl_id_2_} : (!llvm.ptr, i64, i64) -> ()
// CHECK-NEXT:           llvm.call @_Z4keepIJRN4sycl3_V12idILi2EEEEEvDpOT_(%[[VAL_3]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:           llvm.intr.lifetime.end 16, %[[VAL_3]] : !llvm.ptr
// CHECK-NEXT:           llvm.return
// CHECK-NEXT:         }
void id(std::size_t i, std::size_t j) {
  sycl::id<2> id(i, j);
  keep(id);
}

// CHECK-LABEL:   llvm.func local_unnamed_addr @_Z2idmmm(
// CHECK-SAME:                        %[[VAL_0:.*]]: i64 {llvm.noundef}, %[[VAL_1:.*]]: i64 {llvm.noundef}, %[[VAL_2:.*]]: i64 {llvm.noundef})
// CHECK-NEXT:           %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:           llvm.intr.lifetime.start 24, %[[VAL_4]] : !llvm.ptr
// CHECK-NEXT:           sycl.host.constructor(%[[VAL_4]], %[[VAL_0]], %[[VAL_1]], %[[VAL_1]]) {type = !sycl_id_3_} : (!llvm.ptr, i64, i64, i64) -> ()
// CHECK-NEXT:           llvm.call @_Z4keepIJRN4sycl3_V12idILi3EEEEEvDpOT_(%[[VAL_4]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:           llvm.intr.lifetime.end 24, %[[VAL_4]] : !llvm.ptr
// CHECK-NEXT:           llvm.return
// CHECK-NEXT:         }
void id(std::size_t i, std::size_t j, std::size_t k) {
  sycl::id<3> id(i, j, j);
  keep(id);
}

template <int Dimensions>
void id(const sycl::id<Dimensions> &other) {
  sycl::id<Dimensions> id(other);
  keep(id);
}

template <int Dimensions>
void id(sycl::id<Dimensions> &&other) {
  sycl::id<Dimensions> id(std::move(other));
  keep(id);
}

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z2idILi1EEvv()
// CHECK-NEXT:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:           llvm.intr.lifetime.start 8, %[[VAL_1]] : !llvm.ptr
// CHECK-NEXT:           sycl.host.constructor(%[[VAL_1]]) {type = !sycl_id_1_} : (!llvm.ptr) -> ()
// CHECK-NEXT:           llvm.call @_Z4keepIJRN4sycl3_V12idILi1EEEEEvDpOT_(%[[VAL_1]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:           llvm.intr.lifetime.end 8, %[[VAL_1]] : !llvm.ptr
// CHECK-NEXT:           llvm.return
// CHECK-NEXT:         }
template void id<1>();

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z2idILi2EEvv()
// CHECK-NEXT:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::id.0", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:           llvm.intr.lifetime.start 16, %[[VAL_1]] : !llvm.ptr
// CHECK-NEXT:           sycl.host.constructor(%[[VAL_1]]) {type = !sycl_id_2_} : (!llvm.ptr) -> ()
// CHECK-NEXT:           llvm.call @_Z4keepIJRN4sycl3_V12idILi2EEEEEvDpOT_(%[[VAL_1]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:           llvm.intr.lifetime.end 16, %[[VAL_1]] : !llvm.ptr
// CHECK-NEXT:           llvm.return
// CHECK-NEXT:         }
template void id<2>();

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z2idILi3EEvv()
// CHECK-NEXT:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:           llvm.intr.lifetime.start 24, %[[VAL_1]] : !llvm.ptr
// CHECK-NEXT:           sycl.host.constructor(%[[VAL_1]]) {type = !sycl_id_3_} : (!llvm.ptr) -> ()
// CHECK-NEXT:           llvm.call @_Z4keepIJRN4sycl3_V12idILi3EEEEEvDpOT_(%[[VAL_1]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:           llvm.intr.lifetime.end 24, %[[VAL_1]] : !llvm.ptr
// CHECK-NEXT:           llvm.return
// CHECK-NEXT:         }
template void id<3>();

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z2idILi1EEvRKN4sycl3_V12idIXT_EEE(
// CHECK-SAME:                                                           %[[VAL_0:.*]]: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
// CHECK-NEXT:           %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:           llvm.intr.lifetime.start 8, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:           %[[VAL_3:.*]] = llvm.load %[[VAL_0]] {alignment = 8 : i64} : !llvm.ptr -> i64
// CHECK-NEXT:           sycl.host.constructor(%[[VAL_2]], %[[VAL_3]]) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:           llvm.call @_Z4keepIJRN4sycl3_V12idILi1EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:           llvm.intr.lifetime.end 8, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:           llvm.return
// CHECK-NEXT:         }
template void id<1>(const sycl::id<1> &);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z2idILi2EEvRKN4sycl3_V12idIXT_EEE(
// CHECK-SAME:                                                           %[[VAL_0:.*]]: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::id.0", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 16, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_0]]) {type = !sycl_id_2_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V12idILi2EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 16, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void id<2>(const sycl::id<2> &);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z2idILi3EEvRKN4sycl3_V12idIXT_EEE(
// CHECK-SAME:                                                           %[[VAL_0:.*]]: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 24, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_0]]) {type = !sycl_id_3_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V12idILi3EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 24, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void id<3>(const sycl::id<3> &);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z2idILi1EEvON4sycl3_V12idIXT_EEE(
// CHECK-SAME:                                                          %[[VAL_0:.*]]: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef})
// CHECK-NEXT:           %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:           llvm.intr.lifetime.start 8, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:           %[[VAL_3:.*]] = llvm.load %[[VAL_0]] {alignment = 8 : i64} : !llvm.ptr -> i64
// CHECK-NEXT:           sycl.host.constructor(%[[VAL_2]], %[[VAL_3]]) {type = !sycl_id_1_} : (!llvm.ptr, i64) -> ()
// CHECK-NEXT:           llvm.call @_Z4keepIJRN4sycl3_V12idILi1EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:           llvm.intr.lifetime.end 8, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:           llvm.return
// CHECK-NEXT:         }
template void id<1>(sycl::id<1> &&);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z2idILi2EEvON4sycl3_V12idIXT_EEE(
// CHECK-SAME:                                                          %[[VAL_0:.*]]: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::id.0", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 16, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_0]]) {type = !sycl_id_2_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V12idILi2EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 16, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void id<2>(sycl::id<2> &&);

// CHECK-LABEL:   llvm.func weak_odr local_unnamed_addr @_Z2idILi3EEvON4sycl3_V12idIXT_EEE(
// CHECK-SAME:                                                          %[[VAL_0:.*]]: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 24 : i64, llvm.nonnull, llvm.noundef})
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<"class.sycl::_V1::id.2", (struct<"class.sycl::_V1::detail::array.3", (array<3 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:      llvm.intr.lifetime.start 24, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      sycl.host.constructor(%[[VAL_2]], %[[VAL_0]]) {type = !sycl_id_3_} : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      llvm.call @_Z4keepIJRN4sycl3_V12idILi3EEEEEvDpOT_(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      llvm.intr.lifetime.end 24, %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
template void id<3>(sycl::id<3> &&);
