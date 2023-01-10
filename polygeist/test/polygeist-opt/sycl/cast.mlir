// RUN: polygeist-opt --convert-polygeist-to-llvm="use-bare-ptr-memref-call-conv" --split-input-file %s | FileCheck %s

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>

// CHECK-LABEL:   llvm.func @test1(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>) -> !llvm.ptr<struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>> {
// CHECK:           %[[VAL_1:.*]] = llvm.bitcast %[[VAL_0]] : !llvm.ptr<struct<"class.sycl::_V1::range.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>> to !llvm.ptr<struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>>
// CHECK:           llvm.return %[[VAL_1]] : !llvm.ptr<struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>>
// CHECK:         }

func.func @test1(%arg0: memref<?x!sycl_range_1_>) -> memref<?x!sycl_array_1_> {
  %0 = "sycl.cast"(%arg0) : (memref<?x!sycl_range_1_>) -> memref<?x!sycl_array_1_>
  func.return %0 : memref<?x!sycl_array_1_>
}

// -----

// CHECK-LABEL:   llvm.func @test2(
// CHECK-SAME:                     %[[VAL_0:.*]]: !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>>) -> !llvm.ptr<struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>> {
// CHECK:           %[[VAL_1:.*]] = llvm.bitcast %[[VAL_0]] : !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>> to !llvm.ptr<struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>>
// CHECK:           llvm.return %[[VAL_1]] : !llvm.ptr<struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>>
// CHECK:         }

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
func.func @test2(%arg0: memref<?x!sycl_id_1_>) -> memref<?x!sycl_array_1_> {
  %0 = "sycl.cast"(%arg0) : (memref<?x!sycl_id_1_>) -> memref<?x!sycl_array_1_>
  func.return %0: memref<?x!sycl_array_1_>
}

// -----

// CHECK-LABEL:   llvm.func @test_addrspaces(
// CHECK-SAME:                               %[[VAL_0:.*]]: !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>, 4>) -> !llvm.ptr<struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>, 4> {
// CHECK:           %[[VAL_1:.*]] = llvm.bitcast %[[VAL_0]] : !llvm.ptr<struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>)>, 4> to !llvm.ptr<struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>, 4>
// CHECK:           llvm.return %[[VAL_1]] : !llvm.ptr<struct<"class.sycl::_V1::detail::array.1", (array<1 x i64>)>, 4>
// CHECK:         }

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
func.func @test_addrspaces(%arg0: memref<?x!sycl_id_1_, 4>) -> memref<?x!sycl_array_1_, 4> {
  %0 = "sycl.cast"(%arg0) : (memref<?x!sycl_id_1_, 4>) -> memref<?x!sycl_array_1_, 4>
  func.return %0: memref<?x!sycl_array_1_, 4>
}
