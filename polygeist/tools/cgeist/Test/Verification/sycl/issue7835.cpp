// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>

// CHECK-MLIR: gpu.module @device_functions

// CHECK-MLIR-LABEL: gpu.func @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZ4testRNS0_5queueEEUlNS0_2idILi1EEEE_EE
// CHECK-MLIR-SAME:        %[[VAL_151:.*]]: memref<?x!sycl_range_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_range_1_, llvm.noundef},
// CHECK-MLIR-SAME:        %[[VAL_152:.*]]: !llvm.ptr {llvm.align = 8 : i64, llvm.byval = !llvm.struct<(memref<?xi32, 1>)>, llvm.noundef}
// CHECK-MLIR:             %[[SIZE:.*]] = arith.constant 8 : i64
// CHECK-MLIR:             %[[VAL_153:.*]] = arith.constant 0 : index
// CHECK-MLIR:             %[[VAL_154:.*]] = arith.constant 1 : index
// CHECK-MLIR:             %[[VAL_155:.*]] = arith.constant 1 : i64
// CHECK-MLIR:             %[[VAL_156:.*]] = memref.alloca() : memref<1x!sycl_item_1_>
// CHECK-MLIR:             %[[VAL_157:.*]] = memref.cast %[[VAL_156]] : memref<1x!sycl_item_1_> to memref<?x!sycl_item_1_>
// CHECK-MLIR:             %[[VAL_158:.*]] = llvm.alloca %[[VAL_155]] x !llvm.struct<(memref<?xi32, 4>)> : (i64) -> !llvm.ptr
// CHECK-MLIR:             %[[VAL_159:.*]] = memref.alloca() : memref<1x!sycl_range_1_>
// CHECK-MLIR:             %[[VAL_160:.*]] = memref.cast %[[VAL_159]] : memref<1x!sycl_range_1_> to memref<?x!sycl_range_1_>
// CHECK-MLIR:             %[[VAL_161:.*]] = memref.alloca() : memref<1x!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>>
// CHECK-MLIR:             %[[VAL_162:.*]] = memref.cast %[[VAL_161]] : memref<1x!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>> to memref<?x!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>>
// CHECK-MLIR:             %[[VAL_163:.*]] = "polygeist.subindex"(%[[VAL_162]], %[[VAL_153]]) : (memref<?x!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>>, index) -> memref<?x!sycl_range_1_>
// CHECK-MLIR:             %[[VAL_164:.*]] = memref.memory_space_cast %[[VAL_160]] : memref<?x!sycl_range_1_> to memref<?x!sycl_range_1_, 4>
// CHECK-MLIR:             %[[VAL_165:.*]] = memref.memory_space_cast %[[VAL_151]] : memref<?x!sycl_range_1_> to memref<?x!sycl_range_1_, 4>
// CHECK-MLIR:             sycl.constructor @range(%[[VAL_164]], %[[VAL_165]]) {MangledFunctionName = @_ZN4sycl3_V15rangeILi1EEC1ERKS2_} : (memref<?x!sycl_range_1_, 4>, memref<?x!sycl_range_1_, 4>)
// CHECK-MLIR:             %[[VAL_166:.*]] = affine.load %[[VAL_159]][0] : memref<1x!sycl_range_1_>
// CHECK-MLIR:             affine.store %[[VAL_166]], %[[VAL_163]][0] : memref<?x!sycl_range_1_>
// CHECK-MLIR:             %[[VAL_167:.*]] = "polygeist.subindex"(%[[VAL_162]], %[[VAL_154]]) : (memref<?x!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>>, index) -> memref<?x!llvm.struct<(memref<?xi32, 4>)>>
// CHECK-MLIR:             %[[VAL_168:.*]] = llvm.addrspacecast %[[VAL_152]] : !llvm.ptr to !llvm.ptr<4>
// CHECK-MLIR:             "llvm.intr.memcpy"(%[[VAL_158]], %[[VAL_168]], %[[SIZE]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr<4>, i64) -> ()
// CHECK-MLIR:             %[[VAL_170:.*]] = llvm.load %[[VAL_158]] : !llvm.ptr -> !llvm.struct<(memref<?xi32, 4>)>
// CHECK-MLIR:             affine.store %[[VAL_170]], %[[VAL_167]][0] : memref<?x!llvm.struct<(memref<?xi32, 4>)>>
// CHECK-MLIR:             %[[VAL_171:.*]] = sycl.call @declptr() {MangledFunctionName = @_ZN4sycl3_V16detail7declptrINS0_4itemILi1ELb1EEEEEPT_v} : () -> memref<?x!sycl_item_1_, 4>
// CHECK-MLIR:             %[[VAL_172:.*]] = sycl.call @getElement(%[[VAL_171]]) {MangledFunctionName = @_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE, TypeName = @Builder} : (memref<?x!sycl_item_1_, 4>) -> !sycl_item_1_
// CHECK-MLIR:             %[[VAL_173:.*]] = memref.memory_space_cast %[[VAL_162]] : memref<?x!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>> to memref<?x!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>, 4>
// CHECK-MLIR:             affine.store %[[VAL_172]], %[[VAL_156]][0] : memref<1x!sycl_item_1_>
// CHECK-MLIR:             sycl.call @"operator()"(%[[VAL_173]], %[[VAL_157]]) {MangledFunctionName = @_ZNK4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZ4testRNS0_5queueEEUlNS0_2idILi1EEEE_EclES4_, TypeName = @RoundedRangeKernel} : (memref<?x!llvm.struct<(!sycl_range_1_, !llvm.struct<(memref<?xi32, 4>)>)>, 4>, memref<?x!sycl_item_1_>) -> ()
// CHECK-MLIR:             gpu.return

// CHECK-LLVM-LABEL: define weak_odr spir_kernel void @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZ4testRNS0_5queueEEUlNS0_2idILi1EEEE_EE(
// CHECK-LLVM-SAME:        ptr noundef byval(%"class.sycl::_V1::range.1") align 8 %0,
// CHECK-LLVM-SAME:        ptr noundef byval({ ptr addrspace(1) }) align 8 %1
// CHECK-LLVM-NEXT:    call spir_func void @__itt_offload_wi_start_wrapper()
// CHECK-LLVM-NEXT:    %3 = alloca %"class.sycl::_V1::item.1.true", i64 1, align 8
// CHECK-LLVM-NEXT:    %4 = alloca { ptr addrspace(4) }, i64 1, align 8
// CHECK-LLVM-NEXT:    %5 = alloca %"class.sycl::_V1::range.1", i64 1, align 8
// CHECK-LLVM-NEXT:    %6 = alloca { %"class.sycl::_V1::range.1", { ptr addrspace(4) } }, i64 1, align 8
// CHECK-LLVM-NEXT:    %7 = getelementptr { %"class.sycl::_V1::range.1", { ptr addrspace(4) } }, ptr %6, i32 0, i32 0
// CHECK-LLVM-NEXT:    %8 = addrspacecast ptr %5 to ptr addrspace(4)
// CHECK-LLVM-NEXT:    %9 = addrspacecast ptr %0 to ptr addrspace(4)
// CHECK-LLVM-NEXT:    call spir_func void @_ZN4sycl3_V15rangeILi1EEC1ERKS2_(ptr addrspace(4) %8, ptr addrspace(4) %9)
// CHECK-LLVM-NEXT:    %10 = load %"class.sycl::_V1::range.1", ptr %5, align 8
// CHECK-LLVM-NEXT:    store %"class.sycl::_V1::range.1" %10, ptr %7, align 8
// CHECK-LLVM-NEXT:    %11 = getelementptr { %"class.sycl::_V1::range.1", { ptr addrspace(4) } }, ptr %6, i32 0, i32 1
// CHECK-LLVM-NEXT:    %12 = addrspacecast ptr %1 to ptr addrspace(4)
// CHECK-LLVM-NEXT:    call void @llvm.memcpy.p0.p4.i64(ptr %4, ptr addrspace(4) %12, i64 8, i1 false)
// CHECK-LLVM-NEXT:    %13 = load { ptr addrspace(4) }, ptr %4, align 8
// CHECK-LLVM-NEXT:    store { ptr addrspace(4) } %13, ptr %11, align 8
// CHECK-LLVM-NEXT:    %14 = call spir_func ptr addrspace(4) @_ZN4sycl3_V16detail7declptrINS0_4itemILi1ELb1EEEEEPT_v()
// CHECK-LLVM-NEXT:    %15 = call spir_func %"class.sycl::_V1::item.1.true" @_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE(ptr addrspace(4) %14)
// CHECK-LLVM-NEXT:    %16 = addrspacecast ptr %6 to ptr addrspace(4)
// CHECK-LLVM-NEXT:    store %"class.sycl::_V1::item.1.true" %15, ptr %3, align 8
// CHECK-LLVM-NEXT:    call spir_func void @_ZNK4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZ4testRNS0_5queueEEUlNS0_2idILi1EEEE_EclES4_(ptr addrspace(4) %16, ptr %3)
// CHECK-LLVM-NEXT:    call spir_func void @__itt_offload_wi_finish_wrapper()
// CHECK-LLVM-NEXT:    ret void

int test(sycl::queue &q) {
  int *x = sycl::malloc_device<int>(10, q);
  q.parallel_for(sycl::range(10), [=](sycl::id<1> id) { x[id] = id; });
  return *x;
}
