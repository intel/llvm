; Edge case: instructions after the __spirv_AbortKHR call in the same basic
; block.
;
; In real code (e.g. device libraries' __assert_fail), the pattern is:
;   call void @__spirv_AbortKHR(i32 %msg)
;   ; ... possibly lifetime.end intrinsics ...
;   ret void            ; or unreachable
;
; OpAbortKHR is itself a SPIR-V block terminator, so all subsequent
; instructions in the same BB must be suppressed to produce valid SPIR-V.
; This test verifies that no instructions appear after OpAbortKHR in any
; function.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_abort -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: spirv-val %t.spv

; Round-trip
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; ---- abort followed by unreachable (common pattern) ----
; CHECK-SPIRV: Function
; CHECK-SPIRV: AbortKHR
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: FunctionEnd

; ---- abort followed by ret void (device library pattern) ----
; CHECK-SPIRV: Function
; CHECK-SPIRV: AbortKHR
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: FunctionEnd

; ---- abort followed by lifetime.end + ret void (full device library pattern) ----
; CHECK-SPIRV: Function
; CHECK-SPIRV: AbortKHR
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: FunctionEnd

; ---- Round-trip: all variants recover __spirv_AbortKHR + unreachable ----
; CHECK-LLVM: define spir_func void @abort_then_unreachable
; CHECK-LLVM: call spir_func void @{{.*__spirv_AbortKHR.*}}(i32 {{.*}})
; CHECK-LLVM-NEXT: unreachable
;
; CHECK-LLVM: define spir_func void @abort_then_ret
; CHECK-LLVM: call spir_func void @{{.*__spirv_AbortKHR.*}}(i32 {{.*}})
; CHECK-LLVM-NEXT: unreachable
;
; CHECK-LLVM: define spir_func void @abort_then_lifetime_ret
; CHECK-LLVM: call spir_func void @{{.*__spirv_AbortKHR.*}}(i32 {{.*}})
; CHECK-LLVM-NEXT: unreachable

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Pattern 1: abort + unreachable (standard LLVM pattern after noreturn call)
define spir_func void @abort_then_unreachable(i32 %msg) {
entry:
  call spir_func void @_Z16__spirv_AbortKHRIiEvT_(i32 %msg)
  unreachable
}

; Pattern 2: abort + ret void (seen in device library __assert_fail)
define spir_func void @abort_then_ret(i32 %msg) {
entry:
  call spir_func void @_Z16__spirv_AbortKHRIiEvT_(i32 %msg)
  ret void
}

; Pattern 3: abort + lifetime.end + ret void (full device library pattern)
define spir_func void @abort_then_lifetime_ret(i32 %msg) {
entry:
  %buf = alloca i8, align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr %buf)
  call spir_func void @_Z16__spirv_AbortKHRIiEvT_(i32 %msg)
  call void @llvm.lifetime.end.p0(i64 1, ptr %buf)
  ret void
}

declare spir_func void @_Z16__spirv_AbortKHRIiEvT_(i32)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #0

attributes #0 = { argmemonly nounwind willreturn }

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
