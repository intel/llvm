; RUN: opt < %s -LowerWGScope -S -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -LowerWGScope --mtriple=nvptx -S -enable-new-pm=0 | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-PTX

; RUN: opt < %s -passes=LowerWGScope -S | FileCheck %s
; RUN: opt < %s -passes=LowerWGScope --mtriple=nvptx -S | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-PTX


%struct.baz = type { i64 }

define internal spir_func void @wibble(%struct.baz* byval(%struct.baz) %arg1) !work_group_scope !0 {
; CHECK-PTX:   call i64 @_Z27__spirv_LocalInvocationId_xv()
; CHECK-PTX:   call i64 @_Z27__spirv_LocalInvocationId_yv()
; CHECK-PTX:   call i64 @_Z27__spirv_LocalInvocationId_zv()
; CHECK:   call void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272)
  ret void
}

; CHECK-PTX: declare i64 @_Z27__spirv_LocalInvocationId_xv()

; CHECK-PTX: declare i64 @_Z27__spirv_LocalInvocationId_yv()

; CHECK-PTX: declare i64 @_Z27__spirv_LocalInvocationId_zv()

; CHECK: ; Function Attrs: convergent
; CHECK: declare void @_Z22__spirv_ControlBarrierjjj(i32, i32, i32) #[[ATTR_NUM:[0-9]+]]

; CHECK: attributes #[[ATTR_NUM]] = { convergent }

!0 = !{}
