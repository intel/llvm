; RUN: opt < %s -passes=vector-combine -S | FileCheck --implicit-check-not=shufflevector --implicit-check-not="load <4 x i32>" %s

; Verify we don't replace the insertelement with a shufflevector and a vector load for SPIR targets.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"twoInt" = type { i32, i32 }
%"myS" = type { <1 x i32> }
; Function Attrs: convergent inlinehint norecurse nounwind
define linkonce_odr dso_local spir_func void @foo(ptr addrspace(4) dereferenceable(32) %arg) {
entry:
  %0 = alloca %"myS", align 4
  %1 = getelementptr inbounds %"twoInt", ptr addrspace(4) %arg, i64 0, i32 1
  %2 = load i32, ptr addrspace(4) %1, align 4
; CHECK: %3 = insertelement <1 x i32> poison, i32 %[[#Op:]], i64 0
  %3 = insertelement <1 x i32> poison, i32 %2, i64 0
  %4 = addrspacecast ptr %0 to ptr addrspace(4)
  %5 = getelementptr inbounds %"myS", ptr addrspace(4) %4, i64 0, i32 0
  store <1 x i32> %3, ptr addrspace(4) %5, align 4
  ret void
 }