; Kernel function with assert-like abort pattern. Models the real-world HIP
; assert() use case: kernel calls a helper which calls __spirv_AbortKHR.
;
; Verifies that OpAbortKHR works correctly inside callee functions invoked
; from kernel entry points.
;
; Note: the kernel's assert.fail block has FunctionCall + Unreachable
; (the unreachable is after the call, not after an abort — this is correct
; because the abort is inside the callee, not the caller).

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_abort -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: spirv-val %t.spv

; Round-trip
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; ---- SPIR-V ----
; CHECK-SPIRV-DAG: Capability AbortKHR
; CHECK-SPIRV-DAG: Extension "SPV_KHR_abort"
; CHECK-SPIRV-DAG: EntryPoint 6 [[#KernelId:]] "test_kernel"

; __assert_fail_internal: contains the abort -> OpAbortKHR
; CHECK-SPIRV: Function
; CHECK-SPIRV: AbortKHR
; CHECK-SPIRV: FunctionEnd

; test_kernel: calls __assert_fail_internal, then unreachable (in caller)
; CHECK-SPIRV: Function {{.*}} [[#KernelId]]
; CHECK-SPIRV: BranchConditional
; CHECK-SPIRV: Return
; CHECK-SPIRV: FunctionCall
; CHECK-SPIRV: Unreachable
; CHECK-SPIRV: FunctionEnd

; ---- Round-trip ----
; CHECK-LLVM: define spir_func void @__assert_fail_internal
; CHECK-LLVM: call spir_func void @{{.*__spirv_AbortKHR.*}}(i32 {{.*}})
; CHECK-LLVM-NEXT: unreachable
;
; CHECK-LLVM: define spir_kernel void @test_kernel

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Models __assert_fail from device libraries
define spir_func void @__assert_fail_internal(i32 %msg) #0 {
entry:
  call spir_func void @_Z16__spirv_AbortKHRIiEvT_(i32 %msg)
  unreachable
}

; Kernel entry point with conditional assert
define spir_kernel void @test_kernel(ptr addrspace(1) %in, i32 %N) #1
  !kernel_arg_addr_space !1 !kernel_arg_access_qual !2
  !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !4 {
entry:
  %gid = call spir_func i64 @_Z13get_global_idj(i32 0)
  %gid32 = trunc i64 %gid to i32
  %cmp = icmp slt i32 %gid32, %N
  br i1 %cmp, label %ok, label %assert.fail

ok:
  ret void

assert.fail:
  call spir_func void @__assert_fail_internal(i32 42)
  unreachable
}

declare spir_func i64 @_Z13get_global_idj(i32) #2
declare spir_func void @_Z16__spirv_AbortKHRIiEvT_(i32)

attributes #0 = { noinline noreturn nounwind }
attributes #1 = { nounwind }
attributes #2 = { nounwind }

!opencl.spir.version = !{!0}
!spirv.Source = !{!5}

!0 = !{i32 1, i32 2}
!1 = !{i32 1, i32 0}
!2 = !{!"none", !"none"}
!3 = !{!"int*", !"int"}
!4 = !{!"", !""}
!5 = !{i32 4, i32 100000}
