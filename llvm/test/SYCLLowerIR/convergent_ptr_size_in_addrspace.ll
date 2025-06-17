; RUN: opt < %s -passes=LowerWGScope -S | FileCheck %s

; This test checks that pointer size in default global address space is used for
; size_t type, which is return type of __spirv_LocalInvocationId_* functions.
; Note that pointer size in the default address space is 4.

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-i64:64-v16:16-v32:32-n16:32:64-G1"

%struct.baz = type { i64 }

define internal void @wibble(ptr byval(%struct.baz) %arg1) !work_group_scope !0 {
; CHECK-PTX:   call i64 @_Z27__spirv_LocalInvocationId_xv()
; CHECK-PTX:   call i64 @_Z27__spirv_LocalInvocationId_yv()
; CHECK-PTX:   call i64 @_Z27__spirv_LocalInvocationId_zv()
; CHECK:   call void @_Z22__spirv_ControlBarrieriii(i32 2, i32 2, i32 272)
  ret void
}

; CHECK-PTX: declare i64 @_Z27__spirv_LocalInvocationId_xv()

; CHECK-PTX: declare i64 @_Z27__spirv_LocalInvocationId_yv()

; CHECK-PTX: declare i64 @_Z27__spirv_LocalInvocationId_zv()

; CHECK: ; Function Attrs: convergent
; CHECK: declare void @_Z22__spirv_ControlBarrieriii(i32, i32, i32) #[[ATTR_NUM:[0-9]+]]

; CHECK: attributes #[[ATTR_NUM]] = { convergent }

!0 = !{}
