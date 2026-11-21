; RUN: opt -passes=LowerWGScope -S %s -o - | FileCheck %s

; Check that work-group scope code is executed only by the leader work-item
; when parallel_for_work_item (PFWI) is reached through one or more intermediate
; function calls ("indirect" PFWI), and that calls which (transitively) reach a
; PFWI are left outside of the leader guard so that all work-items reach the
; work-group barriers inside them (convergent control flow).
;
; Call graph:
;   @direct     -> @pfwi
;   @one_level  -> @leaf -> @pfwi
;   @two_level  -> @mid  -> @leaf -> @pfwi
;   @mixed      -> @leaf -> @pfwi, @pfwi
;   @no_pfwi    -> @plain, @pfwi
;
; The front end marks every function on the path from the PFWG lambda to a
; PFWI call with !work_group_scope, and the PFWI itself with !work_item_scope
; and !parallel_for_work_item.
;
; Each "atomicrmw add ... i32 N" is a work-group scope side effect that must
; be executed once per work-group, i.e. be placed in a "wg_leader" block.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

; Baseline: direct PFWI call.
; CHECK-LABEL: define spir_func void @direct(
; CHECK:       wg_leader:
; CHECK-NEXT:    atomicrmw add ptr addrspace(1) %p, i32 1 monotonic
; CHECK-NEXT:    br label %wg_cf
; CHECK:       wg_cf:
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrieriii(
; CHECK-NEXT:    call spir_func void @pfwi(ptr addrspace(1) %p)
; CHECK-NEXT:    {{.*}}load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationIndex
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrieriii(
; CHECK-NEXT:    %{{.*}} = icmp eq i64 %{{.*}}, 0
; CHECK-NEXT:    br i1 %{{.*}}, label %[[L:wg_leader[0-9]*]], label %[[CF:wg_cf[0-9]*]]
; CHECK:       [[L]]:
; CHECK-NEXT:    atomicrmw add ptr addrspace(1) %p, i32 2 monotonic
; CHECK-NEXT:    br label %[[CF]]
; CHECK:         ret void
define spir_func void @direct(ptr addrspace(1) %p) !work_group_scope !0 {
entry:
  %0 = atomicrmw add ptr addrspace(1) %p, i32 1 monotonic
  call spir_func void @pfwi(ptr addrspace(1) %p)
  %1 = atomicrmw add ptr addrspace(1) %p, i32 2 monotonic
  ret void
}

; Reproducer: the PFWG lambda reaches PFWI only through @leaf. The side effect
; before the call must be guarded; the call to @leaf must not be.
; CHECK-LABEL: define spir_func void @one_level(
; CHECK:       wg_leader:
; CHECK-NEXT:    atomicrmw add ptr addrspace(1) %p, i32 10 monotonic
; CHECK-NEXT:    br label %wg_cf
; CHECK:       wg_cf:
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrieriii(
; CHECK-NEXT:    call spir_func void @leaf(ptr addrspace(1) %p)
; CHECK-NEXT:    ret void
define spir_func void @one_level(ptr addrspace(1) %p) !work_group_scope !0 {
entry:
  %0 = atomicrmw add ptr addrspace(1) %p, i32 10 monotonic
  call spir_func void @leaf(ptr addrspace(1) %p)
  ret void
}

; Two levels of indirection. Side effects before and after the call chain must
; be guarded, the call itself must not be.
; CHECK-LABEL: define spir_func void @two_level(
; CHECK:       wg_leader:
; CHECK-NEXT:    atomicrmw add ptr addrspace(1) %p, i32 20 monotonic
; CHECK-NEXT:    br label %wg_cf
; CHECK:       wg_cf:
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrieriii(
; CHECK-NEXT:    call spir_func void @mid(ptr addrspace(1) %p)
; CHECK-NEXT:    {{.*}}load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationIndex
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrieriii(
; CHECK-NEXT:    %{{.*}} = icmp eq i64 %{{.*}}, 0
; CHECK-NEXT:    br i1 %{{.*}}, label %[[L:wg_leader[0-9]*]], label %[[CF:wg_cf[0-9]*]]
; CHECK:       [[L]]:
; CHECK-NEXT:    atomicrmw add ptr addrspace(1) %p, i32 21 monotonic
; CHECK-NEXT:    br label %[[CF]]
; CHECK:         ret void
define spir_func void @two_level(ptr addrspace(1) %p) !work_group_scope !0 {
entry:
  %0 = atomicrmw add ptr addrspace(1) %p, i32 20 monotonic
  call spir_func void @mid(ptr addrspace(1) %p)
  %1 = atomicrmw add ptr addrspace(1) %p, i32 21 monotonic
  ret void
}

; Indirect and direct PFWI calls in the same function.
; CHECK-LABEL: define spir_func void @mixed(
; CHECK:       wg_leader:
; CHECK-NEXT:    atomicrmw add ptr addrspace(1) %p, i32 30 monotonic
; CHECK-NEXT:    br label %wg_cf
; CHECK:       wg_cf:
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrieriii(
; CHECK-NEXT:    call spir_func void @leaf(ptr addrspace(1) %p)
; CHECK-NEXT:    {{.*}}load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationIndex
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrieriii(
; CHECK-NEXT:    %{{.*}} = icmp eq i64 %{{.*}}, 0
; CHECK-NEXT:    br i1 %{{.*}}, label %[[L1:wg_leader[0-9]*]], label %[[CF1:wg_cf[0-9]*]]
; CHECK:       [[L1]]:
; CHECK-NEXT:    atomicrmw add ptr addrspace(1) %p, i32 31 monotonic
; CHECK-NEXT:    br label %[[CF1]]
; CHECK:       [[CF1]]:
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrieriii(
; CHECK-NEXT:    call spir_func void @pfwi(ptr addrspace(1) %p)
; CHECK-NEXT:    {{.*}}load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationIndex
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrieriii(
; CHECK-NEXT:    %{{.*}} = icmp eq i64 %{{.*}}, 0
; CHECK-NEXT:    br i1 %{{.*}}, label %[[L2:wg_leader[0-9]*]], label %[[CF2:wg_cf[0-9]*]]
; CHECK:       [[L2]]:
; CHECK-NEXT:    atomicrmw add ptr addrspace(1) %p, i32 32 monotonic
; CHECK-NEXT:    br label %[[CF2]]
; CHECK:         ret void
define spir_func void @mixed(ptr addrspace(1) %p) !work_group_scope !0 {
entry:
  %0 = atomicrmw add ptr addrspace(1) %p, i32 30 monotonic
  call spir_func void @leaf(ptr addrspace(1) %p)
  %1 = atomicrmw add ptr addrspace(1) %p, i32 31 monotonic
  call spir_func void @pfwi(ptr addrspace(1) %p)
  %2 = atomicrmw add ptr addrspace(1) %p, i32 32 monotonic
  ret void
}

; A call to a function which does not reach a PFWI is ordinary work-group scope
; code and must stay inside the leader guard.
; CHECK-LABEL: define spir_func void @no_pfwi(
; CHECK:       wg_leader:
; CHECK-NEXT:    call spir_func void @plain(ptr addrspace(1) %p)
; CHECK-NEXT:    br label %wg_cf
; CHECK:       wg_cf:
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrieriii(
; CHECK-NEXT:    call spir_func void @pfwi(ptr addrspace(1) %p)
; CHECK:         ret void
define spir_func void @no_pfwi(ptr addrspace(1) %p) !work_group_scope !0 {
entry:
  call spir_func void @plain(ptr addrspace(1) %p)
  call spir_func void @pfwi(ptr addrspace(1) %p)
  ret void
}

; Intermediate function: reaches PFWI only through @leaf.
; CHECK-LABEL: define spir_func void @mid(
; CHECK:       wg_leader:
; CHECK-NEXT:    atomicrmw add ptr addrspace(1) %p, i32 40 monotonic
; CHECK-NEXT:    br label %wg_cf
; CHECK:       wg_cf:
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrieriii(
; CHECK-NEXT:    call spir_func void @leaf(ptr addrspace(1) %p)
; CHECK-NEXT:    ret void
define spir_func void @mid(ptr addrspace(1) %p) !work_group_scope !0 {
entry:
  %0 = atomicrmw add ptr addrspace(1) %p, i32 40 monotonic
  call spir_func void @leaf(ptr addrspace(1) %p)
  ret void
}

; Direct caller of PFWI with side effects on both sides of the call.
; CHECK-LABEL: define spir_func void @leaf(
; CHECK:       wg_leader:
; CHECK-NEXT:    atomicrmw add ptr addrspace(1) %p, i32 50 monotonic
; CHECK-NEXT:    br label %wg_cf
; CHECK:       wg_cf:
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrieriii(
; CHECK-NEXT:    call spir_func void @pfwi(ptr addrspace(1) %p)
; CHECK-NEXT:    {{.*}}load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationIndex
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrieriii(
; CHECK-NEXT:    %{{.*}} = icmp eq i64 %{{.*}}, 0
; CHECK-NEXT:    br i1 %{{.*}}, label %[[L:wg_leader[0-9]*]], label %[[CF:wg_cf[0-9]*]]
; CHECK:       [[L]]:
; CHECK-NEXT:    atomicrmw add ptr addrspace(1) %p, i32 51 monotonic
; CHECK-NEXT:    br label %[[CF]]
; CHECK:         ret void
define spir_func void @leaf(ptr addrspace(1) %p) !work_group_scope !0 {
entry:
  %0 = atomicrmw add ptr addrspace(1) %p, i32 50 monotonic
  call spir_func void @pfwi(ptr addrspace(1) %p)
  %1 = atomicrmw add ptr addrspace(1) %p, i32 51 monotonic
  ret void
}

; Ordinary helper, not on a path to PFWI: not transformed.
; CHECK-LABEL: define spir_func void @plain(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    atomicrmw add ptr addrspace(1) %p, i32 60 monotonic
; CHECK-NEXT:    ret void
define spir_func void @plain(ptr addrspace(1) %p) {
entry:
  %0 = atomicrmw add ptr addrspace(1) %p, i32 60 monotonic
  ret void
}

; The PFWI itself: executed by all work-items, not transformed.
; CHECK-LABEL: define spir_func void @pfwi(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret void
define spir_func void @pfwi(ptr addrspace(1) %p) !work_item_scope !0 !parallel_for_work_item !0 {
entry:
  ret void
}

!0 = !{}
