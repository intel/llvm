// RUN: sycl-mlir-opt -convert-sycl-to-llvm %s | FileCheck %s

!sycl_half = !sycl.half<(f16)>
!sycl_vec_i32_4_ = !sycl.vec<[i32, 4], (vector<4xi32>)>
!sycl_vec_sycl_half_2_ = !sycl.vec<[!sycl_half, 2], (vector<2xf16>)>

// CHECK-LABEL:   llvm.func @test_half(
// CHECK-SAME:                         %[[VAL_0:.*]]: !llvm.struct<"class.sycl::_V1::half", (f16)>, %[[VAL_1:.*]]: !llvm.struct<"class.sycl::_V1::half", (f16)>) -> !llvm.struct<"class.sycl::_V1::half", (f16)> {
// CHECK:           %[[VAL_2:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<"class.sycl::_V1::half", (f16)>
// CHECK:           %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_1]][0] : !llvm.struct<"class.sycl::_V1::half", (f16)>
// CHECK:           %[[VAL_4:.*]] = llvm.fadd %[[VAL_2]], %[[VAL_3]]  : f16
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.undef : !llvm.struct<"class.sycl::_V1::half", (f16)>
// CHECK:           %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_5]][0] : !llvm.struct<"class.sycl::_V1::half", (f16)>
// CHECK:           llvm.return %[[VAL_6]] : !llvm.struct<"class.sycl::_V1::half", (f16)>
// CHECK:         }
func.func @test_half(%arg0 : !sycl_half, %arg1 : !sycl_half) -> !sycl_half {
  %0 = sycl.mlir.unwrap %arg0 : !sycl_half to f16
  %1 = sycl.mlir.unwrap %arg1 : !sycl_half to f16
  %2 = arith.addf %0, %1 : f16
  %3 = sycl.mlir.wrap %2 : f16 to !sycl_half
  return %3 : !sycl_half
}

// CHECK-LABEL:   llvm.func @test_vector_of_native(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !llvm.struct<"class.sycl::_V1::vec", (vector<4xi32>)>, %[[VAL_1:.*]]: !llvm.struct<"class.sycl::_V1::vec", (vector<4xi32>)>) -> !llvm.struct<"class.sycl::_V1::vec", (vector<4xi32>)> {
// CHECK:           %[[VAL_2:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<"class.sycl::_V1::vec", (vector<4xi32>)>
// CHECK:           %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_1]][0] : !llvm.struct<"class.sycl::_V1::vec", (vector<4xi32>)>
// CHECK:           %[[VAL_4:.*]] = llvm.add %[[VAL_2]], %[[VAL_3]]  : vector<4xi32>
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.undef : !llvm.struct<"class.sycl::_V1::vec", (vector<4xi32>)>
// CHECK:           %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_5]][0] : !llvm.struct<"class.sycl::_V1::vec", (vector<4xi32>)>
// CHECK:           llvm.return %[[VAL_6]] : !llvm.struct<"class.sycl::_V1::vec", (vector<4xi32>)>
// CHECK:         }
func.func @test_vector_of_native(%arg0 : !sycl_vec_i32_4_, %arg1 : !sycl_vec_i32_4_) -> !sycl_vec_i32_4_ {
  %0 = sycl.mlir.unwrap %arg0 : !sycl_vec_i32_4_ to vector<4xi32>
  %1 = sycl.mlir.unwrap %arg1 : !sycl_vec_i32_4_ to vector<4xi32>
  %2 = arith.addi %0, %1 : vector<4xi32>
  %3 = sycl.mlir.wrap %2 : vector<4xi32> to !sycl_vec_i32_4_
  return %3 : !sycl_vec_i32_4_
}

// CHECK-LABEL:   llvm.func @test_vector_of_half(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !llvm.struct<"class.sycl::_V1::vec.1", (vector<2xf16>)>, %[[VAL_1:.*]]: !llvm.struct<"class.sycl::_V1::vec.1", (vector<2xf16>)>) -> !llvm.struct<"class.sycl::_V1::vec.1", (vector<2xf16>)> {
// CHECK:           %[[VAL_2:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<"class.sycl::_V1::vec.1", (vector<2xf16>)>
// CHECK:           %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_1]][0] : !llvm.struct<"class.sycl::_V1::vec.1", (vector<2xf16>)>
// CHECK:           %[[VAL_4:.*]] = llvm.fadd %[[VAL_2]], %[[VAL_3]]  : vector<2xf16>
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.undef : !llvm.struct<"class.sycl::_V1::vec.1", (vector<2xf16>)>
// CHECK:           %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_5]][0] : !llvm.struct<"class.sycl::_V1::vec.1", (vector<2xf16>)>
// CHECK:           llvm.return %[[VAL_6]] : !llvm.struct<"class.sycl::_V1::vec.1", (vector<2xf16>)>
// CHECK:         }
func.func @test_vector_of_half(%arg0 : !sycl_vec_sycl_half_2_, %arg1 : !sycl_vec_sycl_half_2_) -> !sycl_vec_sycl_half_2_ {
  %0 = sycl.mlir.unwrap %arg0 : !sycl_vec_sycl_half_2_ to vector<2xf16>
  %1 = sycl.mlir.unwrap %arg1 : !sycl_vec_sycl_half_2_ to vector<2xf16>
  %2 = arith.addf %0, %1 : vector<2xf16>
  %3 = sycl.mlir.wrap %2 : vector<2xf16> to !sycl_vec_sycl_half_2_
  return %3 : !sycl_vec_sycl_half_2_
}
