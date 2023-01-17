// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>

// CHECK-MLIR: gpu.module @device_functions

// CHECK-MLIR-LABEL: gpu.func @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZ4testRNS0_5queueEEUlNS0_2idILi1EEEE_EE
// CHECK-MLIR-SAME:     (%arg0: memref<?x!sycl_range_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_range_1_, llvm.noundef}, 
// CHECK-MLIR-SAME:      %arg1: !llvm.ptr<!llvm.struct<(memref<?xi32, 1>)>> {llvm.align = 8 : i64, llvm.byval = !llvm.struct<(memref<?xi32, 1>)>, llvm.noundef}) 
// CHECK-MLIR-SAME:      kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>

// CHECK-MLIR: sycl.constructor(%4, %7) {MangledFunctionName = @_ZN4sycl3_V15rangeILi1EEC1ERKS2_, TypeName = @range} : (memref<?x!sycl_range_1_, 4>, memref<?x!sycl_range_1_, 4>) -> ()
// CHECK-MLIR: %8 = affine.load %alloca_0[0] : memref<1x!sycl_range_1_>
// CHECK-MLIR: affine.store %8, %1[0] : memref<?x!sycl_range_1_>
// CHECK-MLIR: %9 = "polygeist.subindex"(%cast_2, %c1) : (memref<?x!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>>, index) -> memref<?x!llvm.struct<(memref<?xi32, 4>)>>
// CHECK-MLIR: %10 = llvm.bitcast %arg1 : !llvm.ptr<!llvm.struct<(memref<?xi32, 1>)>> to !llvm.ptr<!llvm.struct<(memref<?xi32, 4>)>>
// CHECK-MLIR: %11 = llvm.addrspacecast %0 : !llvm.ptr<!llvm.struct<(memref<?xi32, 4>)>> to !llvm.ptr<!llvm.struct<(memref<?xi32, 4>)>, 4>
// CHECK-MLIR: %12 = llvm.addrspacecast %10 : !llvm.ptr<!llvm.struct<(memref<?xi32, 4>)>> to !llvm.ptr<!llvm.struct<(memref<?xi32, 4>)>, 4>
// CHECK-MLIR: func.call @_ZZ4testRN4sycl3_V15queueEENUlNS0_2idILi1EEEE_C1ERKS5_(%11, %12) : (!llvm.ptr<!llvm.struct<(memref<?xi32, 4>)>, 4>, !llvm.ptr<!llvm.struct<(memref<?xi32, 4>)>, 4>) -> ()
// CHECK-MLIR: %13 = llvm.load %0 : !llvm.ptr<!llvm.struct<(memref<?xi32, 4>)>>
// CHECK-MLIR: affine.store %13, %9[0] : memref<?x!llvm.struct<(memref<?xi32, 4>)>>
// CHECK-MLIR: %14 = sycl.call() {FunctionName = @declptr, MangledFunctionName = @_ZN4sycl3_V16detail7declptrINS0_4itemILi1ELb1EEEEEPT_v} : () -> memref<?x!sycl_item_1_, 4>
// CHECK-MLIR: %15 = sycl.call(%14) {FunctionName = @getElement, MangledFunctionName = @_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE, TypeName = @Builder} : (memref<?x!sycl_item_1_, 4>) -> !sycl_item_1_
// CHECK-MLIR: %16 = "polygeist.memref2pointer"(%alloca_1) : (memref<1x!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>>) -> !llvm.ptr<!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>>
// CHECK-MLIR: %17 = llvm.addrspacecast %16 : !llvm.ptr<!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>> to !llvm.ptr<!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>, 4>
// CHECK-MLIR: %18 = "polygeist.pointer2memref"(%17) : (!llvm.ptr<!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>, 4>) -> memref<?x!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>, 4>
// CHECK-MLIR: affine.store %15, %alloca[0] : memref<1x!sycl_item_1_>
// CHECK-MLIR: sycl.call(%18, %cast) {FunctionName = @"operator()", MangledFunctionName = @_ZNK4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZ4testRNS0_5queueEEUlNS0_2idILi1EEEE_EclES4_, TypeName = @RoundedRangeKernel} : (memref<?x!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>, 4>, memref<?x!sycl_item_1_>) -> ()
// CHECK-MLIR: gpu.return

// CHECK-LLVM-LABEL: define weak_odr spir_kernel void @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZ4testRNS0_5queueEEUlNS0_2idILi1EEEE_EE
// CHECK-LLVM-SAME:     (%"class.sycl::_V1::range.1"* noundef byval(%"class.sycl::_V1::range.1") align 8 %0, 
// CHECK-LLVM-SAME:     { i32 addrspace(1)* }* noundef byval({ i32 addrspace(1)* }) align 8 %1) #0 {
// CHECK-LLVM-NEXT:  %3 = alloca %"class.sycl::_V1::item.1.true", align 8
// CHECK-LLVM-NEXT:  %4 = alloca { i32 addrspace(4)* }, i64 1, align 8
// CHECK-LLVM-NEXT:  %5 = alloca %"class.sycl::_V1::range.1", align 8
// CHECK-LLVM-NEXT:  %6 = alloca { %"class.sycl::_V1::range.1", { i32 addrspace(4)* } }, align 8
// CHECK-LLVM-NEXT:  %7 = getelementptr { %"class.sycl::_V1::range.1", { i32 addrspace(4)* } }, { %"class.sycl::_V1::range.1", { i32 addrspace(4)* } }* %6, i32 0, i32 0
// CHECK-LLVM-NEXT:  %8 = addrspacecast %"class.sycl::_V1::range.1"* %5 to %"class.sycl::_V1::range.1" addrspace(4)*
// CHECK-LLVM-NEXT:  %9 = addrspacecast %"class.sycl::_V1::range.1"* %0 to %"class.sycl::_V1::range.1" addrspace(4)*
// CHECK-LLVM-NEXT:  call spir_func void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_(%"class.sycl::_V1::range.1" addrspace(4)* %8, %"class.sycl::_V1::range.1" addrspace(4)* %9)
// CHECK-LLVM-NEXT:  %10 = load %"class.sycl::_V1::range.1", %"class.sycl::_V1::range.1"* %5, align 8
// CHECK-LLVM-NEXT:  store %"class.sycl::_V1::range.1" %10, %"class.sycl::_V1::range.1"* %7, align 8
// CHECK-LLVM-NEXT:  %11 = getelementptr { %"class.sycl::_V1::range.1", { i32 addrspace(4)* } }, { %"class.sycl::_V1::range.1", { i32 addrspace(4)* } }* %6, i32 0, i32 1
// CHECK-LLVM-NEXT:  %12 = bitcast { i32 addrspace(1)* }* %1 to { i32 addrspace(4)* }*
// CHECK-LLVM-NEXT:  %13 = addrspacecast { i32 addrspace(4)* }* %4 to { i32 addrspace(4)* } addrspace(4)*
// CHECK-LLVM-NEXT:  %14 = addrspacecast { i32 addrspace(4)* }* %12 to { i32 addrspace(4)* } addrspace(4)*
// CHECK-LLVM-NEXT:  call spir_func void @_ZZ4testRN4sycl3_V15queueEENUlNS0_2idILi1EEEE_C1ERKS5_({ i32 addrspace(4)* } addrspace(4)* %13, { i32 addrspace(4)* } addrspace(4)* %14)
// CHECK-LLVM-NEXT:  %15 = load { i32 addrspace(4)* }, { i32 addrspace(4)* }* %4, align 8
// CHECK-LLVM-NEXT:  store { i32 addrspace(4)* } %15, { i32 addrspace(4)* }* %11, align 8
// CHECK-LLVM-NEXT:  %16 = call spir_func %"class.sycl::_V1::item.1.true" addrspace(4)* @_ZN4sycl3_V16detail7declptrINS0_4itemILi1ELb1EEEEEPT_v()
// CHECK-LLVM-NEXT:  %17 = call spir_func %"class.sycl::_V1::item.1.true" @_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE(%"class.sycl::_V1::item.1.true" addrspace(4)* %16)
// CHECK-LLVM-NEXT:  %18 = addrspacecast { %"class.sycl::_V1::range.1", { i32 addrspace(4)* } }* %6 to { %"class.sycl::_V1::range.1", { i32 addrspace(4)* } } addrspace(4)*
// CHECK-LLVM-NEXT:  store %"class.sycl::_V1::item.1.true" %17, %"class.sycl::_V1::item.1.true"* %3, align 8
// CHECK-LLVM-NEXT:  call spir_func void @_ZNK4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZ4testRNS0_5queueEEUlNS0_2idILi1EEEE_EclES4_({ %"class.sycl::_V1::range.1", { i32 addrspace(4)* } } addrspace(4)* %18, %"class.sycl::_V1::item.1.true"* %3)
// CHECK-LLVM-NEXT:  ret void

int test(sycl::queue &q) {
  int *x = sycl::malloc_device<int>(10, q);
  q.parallel_for(sycl::range(10), [=](sycl::id<1> id) { x[id] = id; });
  return *x;
}
