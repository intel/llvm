; RUN: sycl-post-link -split=auto -symbols -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.sym

; This test checkes that module is not split if function pointer's user is not
; CallInst.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64_x86_64-unknown-unknown"

@_Z2f1iTable = weak global [1 x i32 (i32)*] [i32 (i32)* @_Z2f1i], align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define dso_local spir_func i32 @_Z2f1i(i32 %a) #0 {
entry:
  ret i32 %a
}

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @kernel1() #1 {
entry:
  %0 = call i32 @indirect_call(i32 (i32)* addrspace(4)* addrspacecast (i32 (i32)** getelementptr inbounds ([1 x i32 (i32)*], [1 x i32 (i32)*]* @_Z2f1iTable, i64 0, i64 0) to i32 (i32)* addrspace(4)*), i32 0)
  ret void
}

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel2() #2 {
entry:
  ret void
}

declare dso_local spir_func i32 @indirect_call(i32 (i32)* addrspace(4)*, i32) local_unnamed_addr

attributes #0 = { mustprogress nofree norecurse nosync nounwind readnone willreturn }
attributes #1 = { convergent norecurse "sycl-module-id"="TU1.cpp" }
attributes #2 = { convergent norecurse "sycl-module-id"="TU2.cpp" }

; CHECK: kernel1
; CHECK: kernel2
