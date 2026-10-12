; Multiple basic blocks: some abort, some don't.
; Verifies that:
;   1. Non-abort blocks are completely unaffected (normal terminators preserved)
;   2. Multiple abort blocks in the same function each get their own OpAbortKHR
;   3. No cross-contamination between blocks.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_abort -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: spirv-val %t.spv

; Round-trip
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; ---- SPIR-V checks ----
; CHECK-SPIRV-DAG: Capability AbortKHR
; CHECK-SPIRV-DAG: Extension "SPV_KHR_abort"

; CHECK-SPIRV: Name [[#LabelEntry:]] "entry"
; CHECK-SPIRV: Name [[#LabelWork:]] "work"
; CHECK-SPIRV: Name [[#LabelRet:]] "ret"
; CHECK-SPIRV: Name [[#LabelErr1:]] "err1"
; CHECK-SPIRV: Name [[#LabelErr2:]] "err2"

; Function with multiple paths
; CHECK-SPIRV: Function
;
; Entry: conditional branch to %work / %err1
; CHECK-SPIRV: Label [[#LabelEntry]]
; CHECK-SPIRV: BranchConditional [[#]] [[#LabelWork]] [[#LabelErr1]]
;
; Work block: second conditional branch to %ret / %err2
; CHECK-SPIRV: Label [[#LabelWork]]
; CHECK-SPIRV: BranchConditional [[#]] [[#LabelRet]] [[#LabelErr2]]
;
; Return block: normal return
; CHECK-SPIRV: Label [[#LabelRet]]
; CHECK-SPIRV: ReturnValue
;
; First abort block
; CHECK-SPIRV: Label [[#LabelErr1]]
; CHECK-SPIRV: AbortKHR
;
; Second abort block
; CHECK-SPIRV: Label [[#LabelErr2]]
; CHECK-SPIRV: AbortKHR
; CHECK-SPIRV: FunctionEnd

; ---- Round-trip ----
; CHECK-LLVM: define spir_func i32 @multi_abort
; CHECK-LLVM: br i1
; CHECK-LLVM: br i1
; CHECK-LLVM: ret i32
; CHECK-LLVM: call spir_func void @{{.*__spirv_AbortKHR.*}}(i32 {{.*}})
; CHECK-LLVM-NEXT: unreachable
; CHECK-LLVM: call spir_func void @{{.*__spirv_AbortKHR.*}}(i32 {{.*}})
; CHECK-LLVM-NEXT: unreachable

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func i32 @multi_abort(i32 %x, i32 %m1, i32 %m2) {
entry:
  %cmp1 = icmp sgt i32 %x, 0
  br i1 %cmp1, label %work, label %err1

work:
  %result = mul i32 %x, 42
  %cmp2 = icmp slt i32 %result, 1000
  br i1 %cmp2, label %ret, label %err2

ret:
  ret i32 %result

err1:
  call spir_func void @_Z16__spirv_AbortKHRIiEvT_(i32 %m1)
  unreachable

err2:
  call spir_func void @_Z16__spirv_AbortKHRIiEvT_(i32 %m2)
  unreachable
}

declare spir_func void @_Z16__spirv_AbortKHRIiEvT_(i32)

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
