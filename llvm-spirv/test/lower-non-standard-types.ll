; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -s %t.bc -o - | llvm-dis -o - | FileCheck %s --implicit-check-not="<6 x i32>"

; CHECK: [[ASCastInst:%.*]] = addrspacecast ptr addrspace(1) @Id to ptr addrspace(4)
; CHECK: [[LoadInst1:%.*]] = load <3 x i64>, ptr addrspace(4) [[ASCastInst]], align 32
; CHECK: [[LoadInst2:%.*]] = load <3 x i64>, ptr addrspace(4) [[ASCastInst]], align 32
; CHECK: [[ExtrElInst1:%.*]] = extractelement <3 x i64> [[LoadInst1]], i64 0
; CHECK: [[TruncInst1:%.*]] = trunc i64 [[ExtrElInst1]] to i32
; CHECK: [[ExtrElInst2:%.*]] = extractelement <3 x i64> [[LoadInst2]], i64 2
; CHECK: [[LShrInst:%.*]] = lshr i64 [[ExtrElInst2]], 32
; CHECK: [[TruncInst2:%.*]] = trunc i64 [[LShrInst]] to i32
; CHECK: %conv1 = sitofp i32 [[TruncInst1]] to float
; CHECK: %conv2 = sitofp i32 [[TruncInst2]] to float

; ModuleID = 'lower-non-standard-types'
source_filename = "lower-non-standard-types.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@Id = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent norecurse
define dso_local spir_func void @vmult2() local_unnamed_addr #0 !sycl_explicit_simd !4 !intel_reqd_sub_group_size !6 {
entry:
  %0 = load <6 x i32>, ptr addrspace(4) addrspacecast (ptr addrspace(1) @Id to ptr addrspace(4)), align 32
  %1 = load <6 x i32>, ptr addrspace(4) addrspacecast (ptr addrspace(1) @Id to ptr addrspace(4)), align 32
  %2 = extractelement <6 x i32> %0, i32 0
  %3 = extractelement <6 x i32> %1, i32 5
  %conv1 = sitofp i32 %2 to float
  %conv2 = sitofp i32 %3 to float
  ret void
}

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="256" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="lower-external-funcs-with-z.cpp" }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!opencl.used.extensions = !{!4}
!opencl.used.optional.core.features = !{!4}
!opencl.compiler.options = !{!4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 0, i32 100000}
!4 = !{}
!5 = !{!"Compiler"}
!6 = !{i32 1}
