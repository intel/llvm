;; kernel void test(global uint* dst)
;; {
;;    __spirv_ControlBarrierArriveINTEL(2, 2, 260);  // local
;;    __spirv_ControlBarrierWaitINTEL(2, 2, 258);    // local
;;    __spirv_ControlBarrierArriveINTEL(2, 2, 516);  // global
;;    __spirv_ControlBarrierWaitINTEL(2, 2, 514);    // global
;;    __spirv_ControlBarrierArriveINTEL(2, 2, 2052); // image
;;    __spirv_ControlBarrierWaitINTEL(2, 2, 2050);   // image
;;
;;    __spirv_ControlBarrierArriveINTEL(2, 2, 772);  // local + global
;;    __spirv_ControlBarrierWaitINTEL(2, 2, 770);    // local + global
;;    __spirv_ControlBarrierArriveINTEL(2, 2, 2820); // local + global + image
;;    __spirv_ControlBarrierWaitINTEL(2, 2, 2818);   // local + global + image
;;
;;    __spirv_ControlBarrierArriveINTEL(2, 4, 260);  // local, work_item
;;    __spirv_ControlBarrierWaitINTEL(2, 4, 258);    // local, work_item
;;    __spirv_ControlBarrierArriveINTEL(2, 2, 260);  // local, work_group
;;    __spirv_ControlBarrierWaitINTEL(2, 2, 258);    // local, work_group
;;    __spirv_ControlBarrierArriveINTEL(2, 1, 260);  // local, device
;;    __spirv_ControlBarrierWaitINTEL(2, 1, 258);    // local, device
;;    __spirv_ControlBarrierArriveINTEL(2, 0, 260);  // local, all_svm_devices
;;    __spirv_ControlBarrierWaitINTEL(2, 0, 258);    // local, all_svm_devices
;;    __spirv_ControlBarrierArriveINTEL(2, 3, 260);  // local, subgroup
;;    __spirv_ControlBarrierWaitINTEL(2, 3, 258);    // local, subgroup
;;}

; Test for SPV_INTEL_split_barrier (SPIR-V friendly LLVM IR)
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_split_barrier
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_INTEL_split_barrier

; ModuleID = 'split_barrier_spirv.cl'
source_filename = "split_barrier_spirv.cl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; CHECK-SPIRV: Capability SplitBarrierINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_split_barrier"
; CHECK-SPIRV: TypeInt [[UINT:[0-9]+]] 32 0
;
; Scopes:
; CHECK-SPIRV-DAG: Constant [[UINT]] [[SCOPE_WORK_GROUP:[0-9]+]] 2
; CHECK-SPIRV-DAG: Constant [[UINT]] [[SCOPE_INVOCATION:[0-9]+]] 4
; CHECK-SPIRV-DAG: Constant [[UINT]] [[SCOPE_DEVICE:[0-9]+]] 1
; CHECK-SPIRV-DAG: Constant [[UINT]] [[SCOPE_CROSS_DEVICE:[0-9]+]] 0
; CHECK-SPIRV-DAG: Constant [[UINT]] [[SCOPE_SUBGROUP:[0-9]+]] 3
;
; Memory Semantics:
; 0x2 Acquire + 0x100 WorkgroupMemory
; CHECK-SPIRV-DAG: Constant [[UINT]] [[ACQUIRE_LOCAL:[0-9]+]] 258
; 0x4 Release + 0x100 WorkgroupMemory
; CHECK-SPIRV-DAG: Constant [[UINT]] [[RELEASE_LOCAL:[0-9]+]] 260
; 0x2 Acquire + 0x200 CrossWorkgroupMemory
; CHECK-SPIRV-DAG: Constant [[UINT]] [[ACQUIRE_GLOBAL:[0-9]+]] 514
; 0x4 Release + 0x200 CrossWorkgroupMemory
; CHECK-SPIRV-DAG: Constant [[UINT]] [[RELEASE_GLOBAL:[0-9]+]] 516
; 0x2 Acquire + 0x800 ImageMemory
; CHECK-SPIRV-DAG: Constant [[UINT]] [[ACQUIRE_IMAGE:[0-9]+]] 2050
; 0x4 Acquire + 0x800 ImageMemory
; CHECK-SPIRV-DAG: Constant [[UINT]] [[RELEASE_IMAGE:[0-9]+]] 2052
; 0x2 Acquire + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory
; CHECK-SPIRV-DAG: Constant [[UINT]] [[ACQUIRE_LOCAL_GLOBAL:[0-9]+]] 770
; 0x4 Release + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory
; CHECK-SPIRV-DAG: Constant [[UINT]] [[RELEASE_LOCAL_GLOBAL:[0-9]+]] 772
; 0x2 Acquire + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory + 0x800 ImageMemory
; CHECK-SPIRV-DAG: Constant [[UINT]] [[ACQUIRE_LOCAL_GLOBAL_IMAGE:[0-9]+]] 2818
; 0x4 Release + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory + 0x800 ImageMemory
; CHECK-SPIRV-DAG: Constant [[UINT]] [[RELEASE_LOCAL_GLOBAL_IMAGE:[0-9]+]] 2820
;
; CHECK-SPIRV: ControlBarrierArriveINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[RELEASE_LOCAL]]
; CHECK-SPIRV: ControlBarrierWaitINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[ACQUIRE_LOCAL]]
; CHECK-SPIRV: ControlBarrierArriveINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[RELEASE_GLOBAL]]
; CHECK-SPIRV: ControlBarrierWaitINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[ACQUIRE_GLOBAL]]
; CHECK-SPIRV: ControlBarrierArriveINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[RELEASE_IMAGE]]
; CHECK-SPIRV: ControlBarrierWaitINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[ACQUIRE_IMAGE]]
;
; CHECK-SPIRV: ControlBarrierArriveINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[RELEASE_LOCAL_GLOBAL]]
; CHECK-SPIRV: ControlBarrierWaitINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[ACQUIRE_LOCAL_GLOBAL]]
; CHECK-SPIRV: ControlBarrierArriveINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[RELEASE_LOCAL_GLOBAL_IMAGE]]
; CHECK-SPIRV: ControlBarrierWaitINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[ACQUIRE_LOCAL_GLOBAL_IMAGE]]
;
; CHECK-SPIRV: ControlBarrierArriveINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_INVOCATION]] [[RELEASE_LOCAL]]
; CHECK-SPIRV: ControlBarrierWaitINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_INVOCATION]] [[ACQUIRE_LOCAL]]
; CHECK-SPIRV: ControlBarrierArriveINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[RELEASE_LOCAL]]
; CHECK-SPIRV: ControlBarrierWaitINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[ACQUIRE_LOCAL]]
; CHECK-SPIRV: ControlBarrierArriveINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_DEVICE]] [[RELEASE_LOCAL]]
; CHECK-SPIRV: ControlBarrierWaitINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_DEVICE]] [[ACQUIRE_LOCAL]]
; CHECK-SPIRV: ControlBarrierArriveINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_CROSS_DEVICE]] [[RELEASE_LOCAL]]
; CHECK-SPIRV: ControlBarrierWaitINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_CROSS_DEVICE]] [[ACQUIRE_LOCAL]]
; CHECK-SPIRV: ControlBarrierArriveINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_SUBGROUP]] [[RELEASE_LOCAL]]
; CHECK-SPIRV: ControlBarrierWaitINTEL [[SCOPE_WORK_GROUP]] [[SCOPE_SUBGROUP]] [[ACQUIRE_LOCAL]]

; CHECK-LLVM-LABEL: define spir_kernel void @test
; Function Attrs: convergent norecurse nounwind
define dso_local spir_kernel void @test(ptr addrspace(1) nocapture noundef readnone align 4 %0) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 260) #2
    ; CHECK-LLVM: call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 2, i32 2, i32 260) #1
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 258) #2
    ; CHECK-LLVM: call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 2, i32 2, i32 258) #1
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 516) #2
    ; CHECK-LLVM: call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 2, i32 2, i32 516) #1
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 514) #2
    ; CHECK-LLVM: call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 2, i32 2, i32 514) #1
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 2052) #2
    ; CHECK-LLVM: call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 2, i32 2, i32 2052) #1
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 2050) #2
    ; CHECK-LLVM: call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 2, i32 2, i32 2050) #1
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 772) #2
    ; CHECK-LLVM: call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 2, i32 2, i32 772) #1
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 770) #2
    ; CHECK-LLVM: call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 2, i32 2, i32 770) #1
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 2820) #2
    ; CHECK-LLVM: call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 2, i32 2, i32 2820) #1
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 2818) #2
    ; CHECK-LLVM: call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 2, i32 2, i32 2818) #1
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 4, i32 noundef 260) #2
    ; CHECK-LLVM: call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 2, i32 4, i32 260) #1
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 4, i32 noundef 258) #2
    ; CHECK-LLVM: call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 2, i32 4, i32 258) #1
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 260) #2
    ; CHECK-LLVM: call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 2, i32 2, i32 260) #1
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 2, i32 noundef 258) #2
    ; CHECK-LLVM: call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 2, i32 2, i32 258) #1
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 1, i32 noundef 260) #2
    ; CHECK-LLVM: call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 2, i32 1, i32 260) #1
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 1, i32 noundef 258) #2
    ; CHECK-LLVM: call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 2, i32 1, i32 258) #1
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 0, i32 noundef 260) #2
    ; CHECK-LLVM: call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 2, i32 0, i32 260) #1
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 0, i32 noundef 258) #2
    ; CHECK-LLVM: call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 2, i32 0, i32 258) #1
  tail call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef 2, i32 noundef 3, i32 noundef 260) #2
    ; CHECK-LLVM: call spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 2, i32 3, i32 260) #1
  tail call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef 2, i32 noundef 3, i32 noundef 258) #2
    ; CHECK-LLVM: call spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 2, i32 3, i32 258) #1
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func void @_Z33__spirv_ControlBarrierArriveINTELiii(i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func void @_Z31__spirv_ControlBarrierWaitINTELiii(i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #1

attributes #0 = { convergent norecurse nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0, !1}
!opencl.ocl.version = !{!2}
!opencl.spir.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 2, i32 0}
!3 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project 861386dbd6ff0d91636b7c674c2abb2eccd9d3f2)"}
!4 = !{i32 1}
!5 = !{!"none"}
!6 = !{!"uint*"}
!7 = !{!""}
