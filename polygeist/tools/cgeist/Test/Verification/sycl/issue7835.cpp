// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s -o - | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>

// CHECK-MLIR: gpu.module @device_functions

// CHECK-MLIR-LABEL:     gpu.func @_ZTSZ4testRN4sycl3_V15queueEEUlNS0_2idILi1EEEE_(
// CHECK-MLIR-SAME:          %[[VAL_151:.*]]: memref<?xi32, 1> {llvm.align = 4 : i64, llvm.noundef})
// CHECK-MLIR:             %[[VAL_152:.*]] = arith.constant 8 : i64
// CHECK-MLIR:             %[[VAL_153:.*]] = arith.constant 1 : i64
// CHECK-MLIR:             %[[VAL_154:.*]] = memref.alloca() : memref<1x!sycl_id_1_>
// CHECK-MLIR:             %[[VAL_155:.*]] = memref.cast %[[VAL_154]] : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
// CHECK-MLIR:             %[[VAL_156:.*]] = memref.alloca() : memref<1x!sycl_item_1_>
// CHECK-MLIR:             %[[VAL_157:.*]] = memref.cast %[[VAL_156]] : memref<1x!sycl_item_1_> to memref<?x!sycl_item_1_>
// CHECK-MLIR:             %[[VAL_158:.*]] = memref.alloca() : memref<1x!sycl_id_1_>
// CHECK-MLIR:             %[[VAL_159:.*]] = memref.cast %[[VAL_158]] : memref<1x!sycl_id_1_> to memref<?x!sycl_id_1_>
// CHECK-MLIR:             %[[VAL_160:.*]] = llvm.alloca %[[VAL_153]] x !llvm.struct<(memref<?xi32, 4>)> : (i64) -> !llvm.ptr
// CHECK-MLIR:             %[[VAL_161:.*]] = llvm.getelementptr inbounds %[[VAL_160]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(memref<?xi32, 4>)>
// CHECK-MLIR:             %[[VAL_162:.*]] = memref.memory_space_cast %[[VAL_151]] : memref<?xi32, 1> to memref<?xi32, 4>
// CHECK-MLIR:             llvm.store %[[VAL_162]], %[[VAL_161]] : memref<?xi32, 4>, !llvm.ptr
// CHECK-MLIR:             %[[VAL_163:.*]] = sycl.call @declptr() {MangledFunctionName = @_ZN4sycl3_V16detail7declptrINS0_4itemILi1ELb1EEEEEPT_v} : () -> memref<?x!sycl_item_1_, 4>
// CHECK-MLIR:             %[[VAL_164:.*]] = sycl.call @getElement(%[[VAL_163]]) {MangledFunctionName = @_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE, TypeName = @Builder} : (memref<?x!sycl_item_1_, 4>) -> !sycl_item_1_
// CHECK-MLIR:             %[[VAL_165:.*]] = memref.memory_space_cast %[[VAL_157]] : memref<?x!sycl_item_1_> to memref<?x!sycl_item_1_, 4>
// CHECK-MLIR:             affine.store %[[VAL_164]], %[[VAL_165]][0] : memref<?x!sycl_item_1_, 4>
// CHECK-MLIR:             %[[VAL_166:.*]] = memref.memory_space_cast %[[VAL_159]] : memref<?x!sycl_id_1_> to memref<?x!sycl_id_1_, 4>
// CHECK-MLIR:             %[[VAL_167:.*]] = sycl.id.constructor(%[[VAL_156]]) : (memref<1x!sycl_item_1_>) -> memref<?x!sycl_id_1_, 4>
// CHECK-MLIR:             %[[VAL_168:.*]] = "polygeist.memref2pointer"(%[[VAL_166]]) : (memref<?x!sycl_id_1_, 4>) -> !llvm.ptr<4>
// CHECK-MLIR:             %[[VAL_169:.*]] = "polygeist.memref2pointer"(%[[VAL_167]]) : (memref<?x!sycl_id_1_, 4>) -> !llvm.ptr<4>
// CHECK-MLIR:             "llvm.intr.memcpy"(%[[VAL_168]], %[[VAL_169]], %[[VAL_152]]) <{isVolatile = false}> : (!llvm.ptr<4>, !llvm.ptr<4>, i64) -> ()
// CHECK-MLIR:             %[[VAL_170:.*]] = llvm.addrspacecast %[[VAL_160]] : !llvm.ptr to !llvm.ptr<4>
// CHECK-MLIR:             %[[VAL_171:.*]] = affine.load %[[VAL_158]][0] : memref<1x!sycl_id_1_>
// CHECK-MLIR:             affine.store %[[VAL_171]], %[[VAL_154]][0] : memref<1x!sycl_id_1_>
// CHECK-MLIR:             func.call @_ZZ4testRN4sycl3_V15queueEENKUlNS0_2idILi1EEEE_clES4_(%[[VAL_170]], %[[VAL_155]]) : (!llvm.ptr<4>, memref<?x!sycl_id_1_>) -> ()
// CHECK-MLIR:             gpu.return

// CHECK-LLVM-LABEL:  define weak_odr spir_kernel void @_ZTSZ4testRN4sycl3_V15queueEEUlNS0_2idILi1EEEE_(
// CHECK-LLVM-SAME:       ptr addrspace(1) noundef align 4 %[[ARG0:.*]])
// CHECK-LLVM:          call spir_func void @__itt_offload_wi_start_wrapper()
// CHECK-LLVM:          %[[VAL0:.*]] = alloca %"class.sycl::_V1::id.1", i64 1, align 8
// CHECK-LLVM:          %[[VAL1:.*]] = alloca %"class.sycl::_V1::item.1.true", i64 1, align 8
// CHECK-LLVM:          %[[VAL2:.*]] = alloca %"class.sycl::_V1::id.1", i64 1, align 8
// CHECK-LLVM:          %[[VAL3:.*]] = alloca { ptr addrspace(4) }, i64 1, align 8
// CHECK-LLVM:          %[[VAL4:.*]] = getelementptr inbounds { ptr addrspace(4) }, ptr %[[VAL3]], i32 0, i32 0
// CHECK-LLVM:          %[[VAL5:.*]] = addrspacecast ptr addrspace(1) %[[ARG0]] to ptr addrspace(4)
// CHECK-LLVM:          store ptr addrspace(4) %[[VAL5]], ptr %[[VAL4]], align 8
// CHECK-LLVM:          %[[VAL6:.*]] = call spir_func noundef ptr addrspace(4) @_ZN4sycl3_V16detail7declptrINS0_4itemILi1ELb1EEEEEPT_v()
// CHECK-LLVM:          %[[VAL8:.*]] = call spir_func %"class.sycl::_V1::item.1.true" @_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE(ptr addrspace(4) noundef %[[VAL6]])
// CHECK-LLVM:          %[[VAL9:.*]] = addrspacecast ptr %[[VAL1]] to ptr addrspace(4)
// CHECK-LLVM:          store %"class.sycl::_V1::item.1.true" %[[VAL8]], ptr addrspace(4) %[[VAL9]], align 8
// CHECK-LLVM:          %[[VAL10:.*]] = addrspacecast ptr %[[VAL2]] to ptr addrspace(4)
// CHECK-LLVM:          %[[VAL11:.*]] = alloca %"class.sycl::_V1::id.1", align 8
// CHECK-LLVM:          %[[VAL12:.*]] = addrspacecast ptr %[[VAL11]] to ptr addrspace(4)
// CHECK-LLVM:          %[[VAL13:.*]] = getelementptr inbounds %"class.sycl::_V1::item.1.true", ptr %[[VAL1]], i32 0, i32 0, i32 1, i32 0, i32 0, i32 0
// CHECK-LLVM:          %[[VAL14:.*]] = load i64, ptr %[[VAL13]], align 8
// CHECK-LLVM:          %[[VAL15:.*]] = getelementptr inbounds %"class.sycl::_V1::id.1", ptr %[[VAL11]], i32 0, i32 0, i32 0, i32 0
// CHECK-LLVM:          store i64 %[[VAL14]], ptr %[[VAL15]], align 8
// CHECK-LLVM:          call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) %[[VAL10]], ptr addrspace(4) %[[VAL12]], i64 8, i1 false)
// CHECK-LLVM:          %[[VAL16:.*]] = addrspacecast ptr %[[VAL3]] to ptr addrspace(4)
// CHECK-LLVM:          %[[VAL17:.*]] = load %"class.sycl::_V1::id.1", ptr %[[VAL2]], align 8
// CHECK-LLVM:          store %"class.sycl::_V1::id.1" %[[VAL17]], ptr %[[VAL0]], align 8
// CHECK-LLVM:          call spir_func void @_ZZ4testRN4sycl3_V15queueEENKUlNS0_2idILi1EEEE_clES4_(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %[[VAL16]], ptr noundef byval(%"class.sycl::_V1::id.1") align 8 %[[VAL0]])
// CHECK-LLVM:          call spir_func void @__itt_offload_wi_finish_wrapper()
// CHECK-LLVM:          ret void

int test(sycl::queue &q) {
  int *x = sycl::malloc_device<int>(10, q);
  q.parallel_for(sycl::range(10), [=](sycl::id<1> id) { x[id] = id; });
  return *x;
}
