; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_vector_compute --spirv-allow-unknown-intrinsics=llvm.genx
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: llvm-spirv -r -emit-opaque-pointers %t.spv -o %t.bc
; RUN: llvm-dis %t.bc -o %t.ll
; RUN: FileCheck %s --input-file %t.spt -check-prefix=SPV
; RUN: FileCheck %s --input-file %t.ll  -check-prefix=LLVM

; ModuleID = 'linear_genx.cpp'
source_filename = "linear_genx.cpp"
target datalayout = "e-p:64:64-i64:64-n8:16:32"
target triple = "spir64"


@global_var = external global i32** #2

; SPV-DAG: Name [[def:[0-9]+]] "_Z24__cm_intrinsic_impl_sdivu2CMvb1_cS_"
; SPV-DAG: Name [[a:[0-9]+]] "a"
; SPV-DAG: Name [[b:[0-9]+]] "b"
; SPV-DAG: Name [[glob:[0-9]+]] "global_var"
; SPV-DAG: Decorate [[def]] SingleElementVectorINTEL
; SPV-DAG: Decorate [[a]] SingleElementVectorINTEL
; SPV-DAG: Decorate [[b]] SingleElementVectorINTEL
; SPV-DAG: Decorate [[glob]] SingleElementVectorINTEL 2

; LLVM-DAG: "VCSingleElementVector"="0" i8 @_Z24__cm_intrinsic_impl_sdivu2CMvb1_cS_(i8 "VCSingleElementVector"="0" %a, i8 "VCSingleElementVector"="0" %b)
; LLVM-DAG: i8 @some.unknown.intrinsic(i8 "VCSingleElementVector"="0", i8)
; Function Attrs: noinline norecurse nounwind readnone
define dso_local "VCSingleElementVector" i8 @_Z24__cm_intrinsic_impl_sdivu2CMvb1_cS_(i8 "VCSingleElementVector" %a, i8 "VCSingleElementVector" %b) local_unnamed_addr #1 {
entry:
  %conv.i.i = sitofp i8 %a to float
  %conv1.i.i = sitofp i8 %b to float
  %div.i.i = fdiv float 1.000000e+00, %conv1.i.i
  %mul.i.i = fmul float %conv.i.i, 0x3FF0000100000000
  %mul2.i.i = fmul float %mul.i.i, %div.i.i
  %conv3.i.i = fptosi float %mul2.i.i to i32
  %conv3.i = trunc i32 %conv3.i.i to i8
  ret i8 %conv3.i
}


; Function Attrs: noinline nounwind
define dso_local dllexport spir_kernel void @linear(i8 %ibuf, i8 %obuf) local_unnamed_addr #1 {
entry:
  %0 = call i8 @_Z24__cm_intrinsic_impl_sdivu2CMvb1_cS_(i8 %ibuf, i8 %obuf)
  %1 = tail call i8 @some.unknown.intrinsic(i8 %ibuf, i8 %obuf)
  ret void
}

; Function Attrs: nounwind readnone
declare i8 @some.unknown.intrinsic(i8 "VCSingleElementVector", i8) #1

; LLVM: "VCGlobalVariable"
; LLVM-SAME: "VCSingleElementVector"="2"

attributes #1 = { noinline norecurse nounwind readnone "VCFunction"}
attributes #2 = { "VCGlobalVariable" "VCSingleElementVector"="2" }
