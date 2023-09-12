; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_vector_compute --spirv-allow-unknown-intrinsics
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis %t.bc -o %t.ll
; RUN: FileCheck %s --input-file %t.spt -check-prefix=SPV
; RUN: FileCheck %s --input-file %t.ll  -check-prefix=LLVM

target triple = "spir"

; LLVM-DAG: declare i32 @llvm.some.unknown.intrinsic.i32.p1.buffer_ro_t(ptr addrspace(1))
; LLVM-DAG: declare i32 @llvm.some.unknown.intrinsic.i32.p1.buffer_wo_t(ptr addrspace(1))
; LLVM-DAG: declare i32 @llvm.some.unknown.intrinsic.i32.p1.buffer_rw_t(ptr addrspace(1))
; LLVM-DAG: declare i32 @llvm.some.unknown.intrinsic.i32.p1.image1d_rw_t(ptr addrspace(1))
; LLVM-DAG: declare i32 @llvm.some.unknown.intrinsic.i32.p1.image1d_buffer_wo_t(ptr addrspace(1))
; LLVM-DAG: declare i32 @llvm.some.unknown.intrinsic.i32.p1.image2d_wo_t(ptr addrspace(1))
; LLVM-DAG: declare i32 @llvm.some.unknown.intrinsic.i32.p1.image3d_ro_t(ptr addrspace(1))

declare i32 @llvm.some.unknown.intrinsic.i32.p1.buffer_ro_t(target("spirv.BufferSurfaceINTEL", 0))
declare i32 @llvm.some.unknown.intrinsic.i32.p1.buffer_wo_t(target("spirv.BufferSurfaceINTEL", 1))
declare i32 @llvm.some.unknown.intrinsic.i32.p1.buffer_rw_t(target("spirv.BufferSurfaceINTEL", 2))
declare i32 @llvm.some.unknown.intrinsic.i32.p1.image1d_rw_t(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 2))
declare i32 @llvm.some.unknown.intrinsic.i32.p1.image1d_buffer_wo_t(target("spirv.Image", void, 5, 0, 0, 0, 0, 0, 1))
declare i32 @llvm.some.unknown.intrinsic.i32.p1.image2d_wo_t(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1))
declare i32 @llvm.some.unknown.intrinsic.i32.p1.image3d_ro_t(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0))

; SPV-DAG: TypeInt [[INT:[0-9]+]] 32 0
; SPV-DAG: TypeVoid [[VOID:[0-9]+]]
; SPV-DAG: TypeBufferSurfaceINTEL [[BUFRO:[0-9]+]] 0{{(^|[^0-9])}}
; SPV-DAG: TypeBufferSurfaceINTEL [[BUFWO:[0-9]+]] 1{{(^|[^0-9])}}
; SPV-DAG: TypeBufferSurfaceINTEL [[BUFRW:[0-9]+]] 2{{(^|[^0-9])}}
; SPV-DAG: TypeImage [[IMAGE1D_RW:[0-9]+]] {{[0-9]+}} 0 0 0 0 0 0 2
; SPV-DAG: TypeImage [[IMAGE1D_BUF_WO:[0-9]+]] {{[0-9]+}} 5 0 0 0 0 0 1
; SPV-DAG: TypeImage [[IMAGE2D_WO:[0-9]+]] {{[0-9]+}} 1 0 0 0 0 0 1
; SPV-DAG: TypeImage [[IMAGE3D_RO:[0-9]+]] {{[0-9]+}} 2 0 0 0 0 0 0
; SPV-DAG: TypeFunction [[BUFINTR_RO:[0-9]+]] [[INT]] [[BUFRO]]{{(^|[^0-9])}}
; SPV-DAG: TypeFunction [[BUFINTR_WO:[0-9]+]] [[INT]] [[BUFWO]]{{(^|[^0-9])}}
; SPV-DAG: TypeFunction [[BUFINTR_RW:[0-9]+]] [[INT]] [[BUFRW]]{{(^|[^0-9])}}
; SPV-DAG: TypeFunction {{[0-9]+}} [[INT]] [[IMAGE1D_RW]]{{(^|[^0-9])}}
; SPV-DAG: TypeFunction {{[0-9]+}} [[INT]] [[IMAGE1D_BUF_WO]]{{(^|[^0-9])}}
; SPV-DAG: TypeFunction {{[0-9]+}} [[INT]] [[IMAGE2D_WO]]{{(^|[^0-9])}}
; SPV-DAG: TypeFunction {{[0-9]+}} [[INT]] [[IMAGE3D_RO]]{{(^|[^0-9])}}
; SPV-DAG: TypeFunction {{[0-9]+}} [[VOID]] [[BUFRO]] [[BUFWO]] [[BUFRW]] [[IMAGE1D_RW]] [[IMAGE1D_BUF_WO]] [[IMAGE2D_WO]] [[IMAGE3D_RO]]{{(^|[^0-9])}}

; LLVM-DAG: define spir_kernel void @test(ptr addrspace(1) %buf_ro, ptr addrspace(1) %buf_wo, ptr addrspace(1) %buf_rw, ptr addrspace(1) %im1d, ptr addrspace(1) %im1db, ptr addrspace(1) %im2d, ptr addrspace(1) %im3d)
define spir_kernel void @test(target("spirv.BufferSurfaceINTEL", 0) %buf_ro, target("spirv.BufferSurfaceINTEL", 1) %buf_wo, target("spirv.BufferSurfaceINTEL", 2) %buf_rw, target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 2) %im1d, target("spirv.Image", void, 5, 0, 0, 0, 0, 0, 1) %im1db, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %im2d, target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %im3d) #0 {
entry:
; LLVM: %0 = call i32 @llvm.some.unknown.intrinsic.i32.p1.buffer_ro_t(ptr addrspace(1) %buf_ro)
; LLVM: %1 = call i32 @llvm.some.unknown.intrinsic.i32.p1.buffer_wo_t(ptr addrspace(1) %buf_wo)
; LLVM: %2 = call i32 @llvm.some.unknown.intrinsic.i32.p1.buffer_rw_t(ptr addrspace(1) %buf_rw)
; LLVM: %3 = call i32 @llvm.some.unknown.intrinsic.i32.p1.image1d_rw_t(ptr addrspace(1) %im1d)
; LLVM: %4 = call i32 @llvm.some.unknown.intrinsic.i32.p1.image1d_buffer_wo_t(ptr addrspace(1) %im1db)
; LLVM: %5 = call i32 @llvm.some.unknown.intrinsic.i32.p1.image2d_wo_t(ptr addrspace(1) %im2d)
; LLVM: %6 = call i32 @llvm.some.unknown.intrinsic.i32.p1.image3d_ro_t(ptr addrspace(1) %im3d)
; LLVM: ret void
  %0 = call i32 @llvm.some.unknown.intrinsic.i32.p1.buffer_ro_t(target("spirv.BufferSurfaceINTEL", 0) %buf_ro)
  %1 = call i32 @llvm.some.unknown.intrinsic.i32.p1.buffer_wo_t(target("spirv.BufferSurfaceINTEL", 1) %buf_wo)
  %2 = call i32 @llvm.some.unknown.intrinsic.i32.p1.buffer_rw_t(target("spirv.BufferSurfaceINTEL", 2) %buf_rw)
  %3 = call i32 @llvm.some.unknown.intrinsic.i32.p1.image1d_rw_t(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 2) %im1d)
  %4 = call i32 @llvm.some.unknown.intrinsic.i32.p1.image1d_buffer_wo_t(target("spirv.Image", void, 5, 0, 0, 0, 0, 0, 1) %im1db)
  %5 = call i32 @llvm.some.unknown.intrinsic.i32.p1.image2d_wo_t(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %im2d)
  %6 = call i32 @llvm.some.unknown.intrinsic.i32.p1.image3d_ro_t(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %im3d)
  ret void
}


attributes #0 = { "VCFunction" }
