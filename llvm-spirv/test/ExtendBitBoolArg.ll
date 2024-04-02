; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -s %t.bc -o %t.regularized.bc
; RUN: llvm-dis %t.regularized.bc -o %t.regularized.ll
; RUN: FileCheck < %t.regularized.ll %s

; Translation cycle should be successful:
; RUN: llvm-spirv %t.regularized.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc

; CHECK: %[[#Base:]] = load i1, ptr addrspace(4){{.*}}, align 1
; CHECK: %[[#LoadShift:]] = load i32, ptr addrspace(4){{.*}} align 4
; CHECK: %[[#AndShift:]] = and i32 %[[#LoadShift]], 1
; CHECK: %[[#CmpShift:]] = icmp ne i32 %[[#AndShift]], 0
; CHECK: %[[#ExtBase:]] = select i1 %[[#Base]], i32 1, i32 0
; CHECK: %[[#ExtShift:]] = select i1 %[[#CmpShift]], i32 1, i32 0
; CHECK: %[[#LSHR:]] = lshr i32 %[[#ExtBase]], %[[#ExtShift]]
; CHECK: and i32 %[[#LSHR]], 1

; CHECK: %[[#ExtVecBase:]] = select <2 x i1> %vec1, <2 x i32> <i32 1, i32 1>, <2 x i32> zeroinitializer
; CHECK: %[[#ExtVecShift:]] = select <2 x i1> %vec2, <2 x i32> <i32 1, i32 1>, <2 x i32> zeroinitializer
; CHECK: lshr <2 x i32> %[[#ExtVecBase]], %[[#ExtVecShift]]

; ModuleID = 'source.bc'
source_filename = "source.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.ac" = type { i1 }

; Function Attrs: convergent mustprogress norecurse nounwind
define linkonce_odr dso_local spir_func void @foo(<2 x i1> %vec1, <2 x i1> %vec2) align 2 {
  %1 = alloca ptr addrspace(4), align 8
  %2 = alloca i32, align 4
  %3 = addrspacecast ptr %1 to ptr addrspace(4)
  %4 = addrspacecast ptr %2 to ptr addrspace(4)
  %5 = load ptr addrspace(4), ptr addrspace(4) %3, align 8
  %6 = load i1, ptr addrspace(4) %5, align 1
  %7 = load i32, ptr addrspace(4) %4, align 4
  %8 = trunc i32 %7 to i1
  %9 = lshr i1 %6, %8
  %10 = zext i1 %9 to i32
  %11 = and i32 %10, 1
  %12 = lshr <2 x i1> %vec1, %vec2
  ret void
}
