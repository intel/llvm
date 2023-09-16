// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>

// CHECK-MLIR: gpu.module @device_functions

// CHECK-MLIR-LABEL: gpu.func @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZ4testRNS0_5queueEEUlNS0_2idILi1EEEE_EE
// CHECK-MLIR-SAME:     (%arg0: memref<?x!sycl_range_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_range_1_, llvm.noundef}, 
// CHECK-MLIR-SAME:      %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.byval = !llvm.struct<(memref<?xi32, 1>)>, llvm.noundef}) 
// CHECK-MLIR-SAME:      kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>

// CHECK-MLIR:         sycl.constructor @range(%[[VAL_11:.*]], %[[VAL_12:.*]]) {MangledFunctionName = @_ZN4sycl3_V15rangeILi1EEC1ERKS2_} : (memref<?x!sycl_range_1_, 4>, memref<?x!sycl_range_1_, 4>)
// CHECK-MLIR:         %[[VAL_14:.*]] = affine.load %[[VAL_6:.*]][0] : memref<1x!sycl_range_1_>
// CHECK-MLIR:         affine.store %[[VAL_14]], %[[VAL_10:.*]][0] : memref<?x!sycl_range_1_>
// CHECK-MLIR:         %[[VAL_15:.*]] = "polygeist.subindex"(%[[VAL_9:.*]], %[[VAL_1:.*]]) : (memref<?x!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>>, index) -> memref<?x!llvm.struct<(memref<?xi32, 4>)>>
// CHECK-MLIR:         %[[VAL_16:.*]] = llvm.addrspacecast %[[VAL_5:.*]] : !llvm.ptr to !llvm.ptr<4>
// CHECK-MLIR:         %[[VAL_17:.*]] = llvm.addrspacecast %[[VAL_18:.*]] : !llvm.ptr to !llvm.ptr<4>
// CHECK-MLIR:         func.call @_ZZ4testRN4sycl3_V15queueEENUlNS0_2idILi1EEEE_C1ERKS5_(%[[VAL_16]], %[[VAL_17]]) : (!llvm.ptr<4>, !llvm.ptr<4>) -> ()
// CHECK-MLIR:         %[[VAL_19:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> !llvm.struct<(memref<?xi32, 4>)>
// CHECK-MLIR:         affine.store %[[VAL_19]], %[[VAL_15]][0] : memref<?x!llvm.struct<(memref<?xi32, 4>)>>
// CHECK-MLIR:         %[[VAL_20:.*]] = sycl.call @declptr() {MangledFunctionName = @_ZN4sycl3_V16detail7declptrINS0_4itemILi1ELb1EEEEEPT_v} : () -> memref<?x!sycl_item_1_, 4>
// CHECK-MLIR:         %[[VAL_21:.*]] = sycl.call @getElement(%[[VAL_20]]) {MangledFunctionName = @_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE, TypeName = @Builder} : (memref<?x!sycl_item_1_, 4>) -> !sycl_item_1_
// CHECK-MLIR:         %[[VAL_22:.*]] = memref.memory_space_cast %[[VAL_9]] : memref<?x!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>> to memref<?x!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>, 4>
// CHECK-MLIR:         affine.store %[[VAL_21]], %[[VAL_3:.*]][0] : memref<1x!sycl_item_1_>
// CHECK-MLIR:         sycl.call @"operator()"(%[[VAL_22]], %[[VAL_4:.*]]) {MangledFunctionName = @_ZNK4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZ4testRNS0_5queueEEUlNS0_2idILi1EEEE_EclES4_, TypeName = @RoundedRangeKernel} : (memref<?x!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>, 4>, memref<?x!sycl_item_1_>) -> ()
// CHECK-MLIR:         gpu.return

// CHECK-LLVM-LABEL: define weak_odr spir_kernel void @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZ4testRNS0_5queueEEUlNS0_2idILi1EEEE_EE(
// CHECK-LLVM-SAME:        ptr noundef byval(%"class.sycl::_V1::range.1") align 8 %0, 
// CHECK-LLVM-SAME:        ptr noundef byval({ ptr addrspace(1) }) align 8 %1)
// CHECK-LLVM-NEXT:    call spir_func void @__itt_offload_wi_start_wrapper()
// CHECK-LLVM-NEXT:    %3 = alloca %"class.sycl::_V1::item.1.true", align 8
// CHECK-LLVM-NEXT:    %4 = alloca { ptr addrspace(4) }, i64 1, align 8
// CHECK-LLVM-NEXT:    %5 = alloca %"class.sycl::_V1::range.1", align 8
// CHECK-LLVM-NEXT:    %6 = alloca { %"class.sycl::_V1::range.1", { ptr addrspace(4) } }, align 8
// CHECK-LLVM-NEXT:    %7 = getelementptr { %"class.sycl::_V1::range.1", { ptr addrspace(4) } }, ptr %6, i32 0, i32 0
// CHECK-LLVM-NEXT:    %8 = addrspacecast ptr %5 to ptr addrspace(4)
// CHECK-LLVM-NEXT:    %9 = addrspacecast ptr %0 to ptr addrspace(4)
// CHECK-LLVM-NEXT:    call spir_func void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_(ptr addrspace(4) %8, ptr addrspace(4) %9)
// CHECK-LLVM-NEXT:    %10 = load %"class.sycl::_V1::range.1", ptr %5, align 8
// CHECK-LLVM-NEXT:    store %"class.sycl::_V1::range.1" %10, ptr %7, align 8
// CHECK-LLVM-NEXT:    %11 = getelementptr { %"class.sycl::_V1::range.1", { ptr addrspace(4) } }, ptr %6, i32 0, i32 1
// CHECK-LLVM-NEXT:    %12 = addrspacecast ptr %4 to ptr addrspace(4)
// CHECK-LLVM-NEXT:    %13 = addrspacecast ptr %1 to ptr addrspace(4)
// CHECK-LLVM-NEXT:    call spir_func void @_ZZ4testRN4sycl3_V15queueEENUlNS0_2idILi1EEEE_C1ERKS5_(ptr addrspace(4) %12, ptr addrspace(4) %13)
// CHECK-LLVM-NEXT:    %14 = load { ptr addrspace(4) }, ptr %4, align 8
// CHECK-LLVM-NEXT:    store { ptr addrspace(4) } %14, ptr %11, align 8
// CHECK-LLVM-NEXT:    %15 = call spir_func ptr addrspace(4) @_ZN4sycl3_V16detail7declptrINS0_4itemILi1ELb1EEEEEPT_v()
// CHECK-LLVM-NEXT:    %16 = call spir_func %"class.sycl::_V1::item.1.true" @_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE(ptr addrspace(4) %15)
// CHECK-LLVM-NEXT:    %17 = addrspacecast ptr %6 to ptr addrspace(4)
// CHECK-LLVM-NEXT:    store %"class.sycl::_V1::item.1.true" %16, ptr %3, align 8
// CHECK-LLVM-NEXT:    call spir_func void @_ZNK4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZ4testRNS0_5queueEEUlNS0_2idILi1EEEE_EclES4_(ptr addrspace(4) %17, ptr %3)
// CHECK-LLVM-NEXT:    call spir_func void @__itt_offload_wi_finish_wrapper()
// CHECK-LLVM-NEXT:    ret void

int test(sycl::queue &q) {
  int *x = sycl::malloc_device<int>(10, q);
  q.parallel_for(sycl::range(10), [=](sycl::id<1> id) { x[id] = id; });
  return *x;
}
