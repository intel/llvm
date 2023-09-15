// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm %s | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>

// Check that the lowering of sycl_id_1_ doesn't clash with "class.sycl::_V1::id.1"
// CHECK: llvm.func @test_id(%arg0: !llvm.struct<"class.sycl::_V1::id.1.1", (struct<"class.sycl::_V1::detail::array.1.1", (array<1 x
func.func @test_id(%arg0: !sycl_id_1_) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::id", (struct<"class.sycl::_V1::detail::array", (array<1 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.alloca %0 x !llvm.struct<"class.sycl::_V1::id.1", (struct<"class.sycl::_V1::detail::array.1", (array<2 x i64>)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  return
}
