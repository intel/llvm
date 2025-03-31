; RUN: opt -passes=cleanup-sycl-metadata -S < %s | FileCheck %s
;
; Test checks that the pass is able to cleanup srcloc metadata
; function metadata

; CHECK-NOT: srcloc

; ModuleID = 'test.cpp'
source_filename = "test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

$_ZNK13KernelFunctorclEv = comdat any

define dso_local spir_func void @_Z6func10v() !sycl_declared_aspects !1 !srcloc !2 {
entry:
  ret void
}


define linkonce_odr spir_func void @_ZNK13KernelFunctorclEv() !sycl_declared_aspects !3 !srcloc !4 {
entry:
  call spir_func void @_Z6func10v()
  ret void
}

!1 = !{i32 5}
!2 = !{i32 2457}
!3 = !{i32 1}
!4 = !{i32 2547}
