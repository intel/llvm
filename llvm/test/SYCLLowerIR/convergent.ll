; RUN: opt < %s -LowerWGScope -S | FileCheck %s
; RUN: opt < %s -LowerWGScope --mtriple=nvptx -S | FileCheck %s -check-prefix=CHECK-PTX


%struct.baz = type { i64 }

define internal spir_func void @wibble(%struct.baz* byval(%struct.baz) %arg1) !work_group_scope !0 {
; CHECK-PTX:   %1 = call i64 @_Z27__spirv_LocalInvocationId_xv() #0
; CHECK-PTX:   %2 = call i64 @_Z27__spirv_LocalInvocationId_yv() #0
; CHECK-PTX:   %3 = call i64 @_Z27__spirv_LocalInvocationId_zv() #0
; CHECK-PTX:   %4 = call i64 @_Z23__spirv_WorkgroupSize_yv() #0
; CHECK-PTX:   %5 = call i64 @_Z23__spirv_WorkgroupSize_zv() #0
; CHECK-PTX:   call void @_Z22__spirv_ControlBarrierN5__spv5ScopeES0_j(i32 2, i32 2, i32 272) #0
; CHECK:   call void @__spirv_ControlBarrier(i32 2, i32 2, i32 272) #1
  ret void
}

; CHECK-PTX: ; Function Attrs: convergent
; CHECK-PTX: declare i64 @_Z27__spirv_LocalInvocationId_xv() #0

; CHECK-PTX: ; Function Attrs: convergent
; CHECK-PTX: declare i64 @_Z27__spirv_LocalInvocationId_yv() #0

; CHECK-PTX: ; Function Attrs: convergent
; CHECK-PTX: declare i64 @_Z27__spirv_LocalInvocationId_zv() #0

; CHECK-PTX: ; Function Attrs: convergent
; CHECK-PTX: declare i64 @_Z23__spirv_WorkgroupSize_yv() #0

; CHECK-PTX: ; Function Attrs: convergent
; CHECK-PTX: declare i64 @_Z23__spirv_WorkgroupSize_zv() #0

; CHECK-PTX: ; Function Attrs: convergent
; CHECK-PTX: declare void @_Z22__spirv_ControlBarrierN5__spv5ScopeES0_j(i32, i32, i32) #0

; CHECK-PTX: attributes #0 = { convergent }

; CHECK: ; Function Attrs: convergent
; CHECK: declare void @__spirv_ControlBarrier(i32, i32, i32) #1

; CHECK: attributes #1 = { convergent }

!0 = !{}
