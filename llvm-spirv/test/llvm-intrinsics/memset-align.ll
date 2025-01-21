; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

define internal spir_func void @_ZN4sycl3_V16detail26builtin_delegate_to_scalarIZNS0_8upsampleINS0_3vecIiLi2EEENS4_IjLi2EEEEENSt9enable_ifIXsr6detailE17enable_upsample_vIT_T0_EENS1_15change_elementsINS1_8map_typeIJNS1_13get_elem_typeIS8_E4typeEashtsitjiljmEE4typeES8_E4typeEE4typeES8_S9_EUlDpT_E_JS5_S6_EEEDaS8_DpRKT0_() {
entry:
  %r.sroa.0 = alloca [2 x i64], i32 1, align 16
  %r.sroa.0.0.r.ascast.sroa_cast1 = addrspacecast ptr %r.sroa.0 to ptr addrspace(4)

; CHECK-SPIRV: Decorate [[#]] Alignment 16
; CHECK-SPIRV: Decorate [[#SrcVar:]] Alignment [[#SrcAlignment:]]
; CHECK-SPIRV: Bitcast [[#]] [[#]] [[#]]
; CHECK-SPIRV: Bitcast [[#]] [[#Src:]] [[#SrcVar]]
; CHECK-SPIRV: CopyMemorySized [[#]] [[#Src]] [[#]] 2 [[#SrcAlignment]]
; CHECK-LLVM: [[SrcVar:@[0-9]+]] = internal unnamed_addr addrspace(2) constant [16 x i8] zeroinitializer, align [[#SrcAlignment:]]
; CHECK-LLVM: [[SrcOp:%[0-9]+]] = bitcast ptr addrspace(2) [[SrcVar]] to ptr addrspace(2)
; CHECK-LLVM: call void @llvm.memcpy.p4.p2.i64(ptr addrspace(4) align 16 %0, ptr addrspace(2) align [[#SrcAlignment]] [[SrcOp]], i64 16, i1 false)
  call void @llvm.memset.p4.i64(ptr addrspace(4) align 16 %r.sroa.0.0.r.ascast.sroa_cast1, i8 0, i64 16, i1 false)
  ret void
}

define spir_func void @_ZN4sycl3_V18upsampleINS0_3vecIiLi2EEENS2_IjLi2EEEEENSt9enable_ifIXsr6detailE17enable_upsample_vIT_T0_EENS0_6detail15change_elementsINS8_8map_typeIJNS8_13get_elem_typeIS6_E4typeEashtsitjiljmEE4typeES6_E4typeEE4typeES6_S7_() {
entry:
  call spir_func void @_ZN4sycl3_V16detail26builtin_delegate_to_scalarIZNS0_8upsampleINS0_3vecIiLi2EEENS4_IjLi2EEEEENSt9enable_ifIXsr6detailE17enable_upsample_vIT_T0_EENS1_15change_elementsINS1_8map_typeIJNS1_13get_elem_typeIS8_E4typeEashtsitjiljmEE4typeES8_E4typeEE4typeES8_S9_EUlDpT_E_JS5_S6_EEEDaS8_DpRKT0_()
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p4.i64(ptr addrspace(4) nocapture writeonly, i8, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: write) }
