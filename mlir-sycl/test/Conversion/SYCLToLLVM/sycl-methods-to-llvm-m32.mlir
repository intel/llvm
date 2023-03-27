// RUN: sycl-mlir-opt -split-input-file -convert-sycl-to-llvm="index-bitwidth=32" %s | FileCheck %s

//===-------------------------------------------------------------------------------------------------===//
// sycl.accessor.get_pointer
//===-------------------------------------------------------------------------------------------------===//

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi32>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi32>)>)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_i32_rw_gb = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>

// CHECK-LABEL:   llvm.func @test(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr<[[ACCESSOR1:.*]]>) -> !llvm.ptr<i32, 1> {
// CHECK-NEXT:      %0 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %1 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:      %2 = llvm.getelementptr inbounds %arg0[0, 0, 2, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<i32>
// CHECK-NEXT:      %3 = llvm.load %2 : !llvm.ptr<i32>
// CHECK-NEXT:      %4 = llvm.getelementptr inbounds %arg0[0, 0, 0, 0, 0, 0] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<i32>
// CHECK-NEXT:      %5 = llvm.load %4 : !llvm.ptr<i32>
// CHECK-NEXT:      %6 = llvm.mul %1, %3  : i32
// CHECK-NEXT:      %7 = llvm.add %6, %5  : i32
// CHECK-NEXT:      %8 = llvm.sub %0, %7  : i32
// CHECK-NEXT:      %9 = llvm.getelementptr inbounds %arg0[0, 1, 0] : (!llvm.ptr<[[ACCESSOR1]]>) -> !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %10 = llvm.load %9 : !llvm.ptr<ptr<i32, 1>>
// CHECK-NEXT:      %11 = llvm.getelementptr inbounds %10[%8] : (!llvm.ptr<i32, 1>, i32) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:      llvm.return %11 : !llvm.ptr<i32, 1>
// CHECK-NEXT:    }
func.func @test(%acc: memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?xi32, 1> {
  %0 = sycl.accessor.get_pointer(%acc) { ArgumentTypes = [memref<?x!sycl_accessor_1_i32_rw_gb>], FunctionName = @"get_pointer", MangledFunctionName = @"get_pointer", TypeName = @"accessor" }  : (memref<?x!sycl_accessor_1_i32_rw_gb>) -> memref<?xi32, 1>
  return %0 : memref<?xi32, 1>
}
