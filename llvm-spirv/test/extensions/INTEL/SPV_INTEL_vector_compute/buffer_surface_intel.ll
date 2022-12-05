; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_vector_compute --spirv-allow-unknown-intrinsics
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis %t.bc -o %t.ll
; RUN: FileCheck %s --input-file %t.spt -check-prefix=SPV
; RUN: FileCheck %s --input-file %t.ll  -check-prefix=LLVM

target triple = "spir"

; LLVM-DAG: %intel.buffer_ro_t = type opaque
; LLVM-DAG: %intel.buffer_wo_t = type opaque
; LLVM-DAG: %intel.buffer_rw_t = type opaque
; LLVM-DAG: %opencl.image1d_rw_t = type opaque
; LLVM-DAG: %opencl.image1d_buffer_wo_t = type opaque
; LLVM-DAG: %opencl.image2d_wo_t = type opaque
; LLVM-DAG: %opencl.image3d_ro_t = type opaque
%intel.buffer_ro_t = type opaque
%intel.buffer_wo_t = type opaque
%intel.buffer_rw_t = type opaque
%opencl.image1d_rw_t = type opaque
%opencl.image1d_buffer_wo_t = type opaque
%opencl.image2d_wo_t = type opaque
%opencl.image3d_ro_t = type opaque

; LLVM-DAG: declare i32 @llvm.some.unknown.intrinsic.i32.p1intel.buffer_ro_t(%intel.buffer_ro_t addrspace(1)*)
; LLVM-DAG: declare i32 @llvm.some.unknown.intrinsic.i32.p1intel.buffer_wo_t(%intel.buffer_wo_t addrspace(1)*)
; LLVM-DAG: declare i32 @llvm.some.unknown.intrinsic.i32.p1intel.buffer_rw_t(%intel.buffer_rw_t addrspace(1)*)
; LLVM-DAG: declare i32 @llvm.some.unknown.intrinsic.i32.p1opencl.image1d_rw_t(%opencl.image1d_rw_t addrspace(1)*)
; LLVM-DAG: declare i32 @llvm.some.unknown.intrinsic.i32.p1opencl.image1d_buffer_wo_t(%opencl.image1d_buffer_wo_t addrspace(1)*)
; LLVM-DAG: declare i32 @llvm.some.unknown.intrinsic.i32.p1opencl.image2d_wo_t(%opencl.image2d_wo_t addrspace(1)*)
; LLVM-DAG: declare i32 @llvm.some.unknown.intrinsic.i32.p1opencl.image3d_ro_t(%opencl.image3d_ro_t addrspace(1)*)

declare i32 @llvm.some.unknown.intrinsic.i32.p1intel.buffer_ro_t(%intel.buffer_ro_t addrspace(1)*)
declare i32 @llvm.some.unknown.intrinsic.i32.p1intel.buffer_wo_t(%intel.buffer_wo_t addrspace(1)*)
declare i32 @llvm.some.unknown.intrinsic.i32.p1intel.buffer_rw_t(%intel.buffer_rw_t addrspace(1)*)
declare i32 @llvm.some.unknown.intrinsic.i32.p1opencl.image1d_rw_t(%opencl.image1d_rw_t addrspace(1)*)
declare i32 @llvm.some.unknown.intrinsic.i32.p1opencl.image1d_buffer_wo_t(%opencl.image1d_buffer_wo_t addrspace(1)*)
declare i32 @llvm.some.unknown.intrinsic.i32.p1opencl.image2d_wo_t(%opencl.image2d_wo_t addrspace(1)*)
declare i32 @llvm.some.unknown.intrinsic.i32.p1opencl.image3d_ro_t(%opencl.image3d_ro_t addrspace(1)*)

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

; LLVM-DAG: define spir_kernel void @test(%intel.buffer_ro_t addrspace(1)* %buf_ro, %intel.buffer_wo_t addrspace(1)* %buf_wo, %intel.buffer_rw_t addrspace(1)* %buf_rw, %opencl.image1d_rw_t addrspace(1)* %im1d, %opencl.image1d_buffer_wo_t addrspace(1)* %im1db, %opencl.image2d_wo_t addrspace(1)* %im2d, %opencl.image3d_ro_t addrspace(1)* %im3d)
define spir_kernel void @test(%intel.buffer_ro_t addrspace(1)* %buf_ro, %intel.buffer_wo_t addrspace(1)* %buf_wo, %intel.buffer_rw_t addrspace(1)* %buf_rw, %opencl.image1d_rw_t addrspace(1)* %im1d, %opencl.image1d_buffer_wo_t addrspace(1)* %im1db, %opencl.image2d_wo_t addrspace(1)* %im2d, %opencl.image3d_ro_t addrspace(1)* %im3d) #0 {
entry:
; LLVM: %0 = call i32 @llvm.some.unknown.intrinsic.i32.p1intel.buffer_ro_t(%intel.buffer_ro_t addrspace(1)* %buf_ro)
; LLVM: %1 = call i32 @llvm.some.unknown.intrinsic.i32.p1intel.buffer_wo_t(%intel.buffer_wo_t addrspace(1)* %buf_wo)
; LLVM: %2 = call i32 @llvm.some.unknown.intrinsic.i32.p1intel.buffer_rw_t(%intel.buffer_rw_t addrspace(1)* %buf_rw)
; LLVM: %3 = call i32 @llvm.some.unknown.intrinsic.i32.p1opencl.image1d_rw_t(%opencl.image1d_rw_t addrspace(1)* %im1d)
; LLVM: %4 = call i32 @llvm.some.unknown.intrinsic.i32.p1opencl.image1d_buffer_wo_t(%opencl.image1d_buffer_wo_t addrspace(1)* %im1db)
; LLVM: %5 = call i32 @llvm.some.unknown.intrinsic.i32.p1opencl.image2d_wo_t(%opencl.image2d_wo_t addrspace(1)* %im2d)
; LLVM: %6 = call i32 @llvm.some.unknown.intrinsic.i32.p1opencl.image3d_ro_t(%opencl.image3d_ro_t addrspace(1)* %im3d)
; LLVM: ret void
  %0 = call i32 @llvm.some.unknown.intrinsic.i32.p1intel.buffer_ro_t(%intel.buffer_ro_t addrspace(1)* %buf_ro)
  %1 = call i32 @llvm.some.unknown.intrinsic.i32.p1intel.buffer_wo_t(%intel.buffer_wo_t addrspace(1)* %buf_wo)
  %2 = call i32 @llvm.some.unknown.intrinsic.i32.p1intel.buffer_rw_t(%intel.buffer_rw_t addrspace(1)* %buf_rw)
  %3 = call i32 @llvm.some.unknown.intrinsic.i32.p1opencl.image1d_rw_t(%opencl.image1d_rw_t addrspace(1)* %im1d)
  %4 = call i32 @llvm.some.unknown.intrinsic.i32.p1opencl.image1d_buffer_wo_t(%opencl.image1d_buffer_wo_t addrspace(1)* %im1db)
  %5 = call i32 @llvm.some.unknown.intrinsic.i32.p1opencl.image2d_wo_t(%opencl.image2d_wo_t addrspace(1)* %im2d)
  %6 = call i32 @llvm.some.unknown.intrinsic.i32.p1opencl.image3d_ro_t(%opencl.image3d_ro_t addrspace(1)* %im3d)
  ret void
}


attributes #0 = { "VCFunction" }
