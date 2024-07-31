; Check support of OpPtrDiff instruction that was added in SPIR-V 1.4

; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv --spirv-max-version=1.3 %t.bc 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s

; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: spirv-val %t.spv

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-ERROR: RequiresVersion: Cannot fulfill SPIR-V version restriction:
; CHECK-ERROR-NEXT: SPIR-V version was restricted to at most 1.3 (66304) but a construct from the input requires SPIR-V version 1.4 (66560) or above

; SPIR-V 1.4
; CHECK-SPIRV: 66560
; CHECK-SPIRV: TypeInt [[#TypeInt:]] 32 0
; CHECK-SPIRV: TypeFloat [[#TypeFloat:]] 32
; CHECK-SPIRV: TypePointer [[#TypePointer:]] [[#]] [[#TypeFloat]]

; CHECK-SPIRV: Variable [[#TypePointer]] [[#Var:]]
; CHECK-SPIRV: PtrDiff [[#TypeInt]] [[#]] [[#Var]] [[#Var]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @test(float %a) local_unnamed_addr #0 {
entry:
  %0 = alloca float, align 4
  store float %a, ptr %0, align 4
; CHECK-LLVM: %[[#Arg1:]] = ptrtoint ptr %[[#]] to i64
; CHECK-LLVM: %[[#Arg2:]] = ptrtoint ptr %[[#]] to i64
; CHECK-LLVM: %[[#Sub:]] = sub i64 %[[#Arg1]], %[[#Arg2]]
; CHECK-LLVM: sdiv exact i64 %[[#Sub]], ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64)
  %1 = call spir_func noundef i32 @_Z15__spirv_PtrDiff(ptr %0, ptr %0)
  ret void
}

declare spir_func noundef i32 @_Z15__spirv_PtrDiff(ptr, ptr)

attributes #0 = { convergent nounwind writeonly }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
