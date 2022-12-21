; RUN: llvm-as -opaque-pointers=0 %s -o %t.bc
; RUN: llvm-spirv -s %t.bc -opaque-pointers=0 -o %t.regulzarized.bc
; RUN: llvm-dis -opaque-pointers=0 %t.regulzarized.bc -o %t.regulzarized.ll
; RUN: FileCheck < %t.regulzarized.ll %s

; Translation cycle should be successfull:
; RUN: llvm-spirv %t.regulzarized.bc -o %t.spv
; RUN: llvm-spirv -r -emit-opaque-pointers %t.spv -o %t.rev.bc

; CHECK: %[[#Base:]] = load i1, i1 addrspace(4)*{{.*}}, align 1
; CHECK: %[[#LoadShift:]] = load i32, i32 addrspace(4)*{{.*}} align 4
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
  %1 = alloca %"class.ac" addrspace(4)*, align 8
  %2 = alloca i32, align 4
  %3 = addrspacecast %"class.ac" addrspace(4)** %1 to %"class.ac" addrspace(4)* addrspace(4)*
  %4 = addrspacecast i32* %2 to i32 addrspace(4)*
  %5 = load %"class.ac" addrspace(4)*, %"class.ac" addrspace(4)* addrspace(4)* %3, align 8
  %6 = getelementptr inbounds %"class.ac", %"class.ac" addrspace(4)* %5, i32 0, i32 0
  %7 = load i1, i1 addrspace(4)* %6, align 1
  %8 = load i32, i32 addrspace(4)* %4, align 4
  %9 = trunc i32 %8 to i1
  %10 = lshr i1 %7, %9
  %11 = zext i1 %10 to i32
  %12 = and i32 %11, 1
  %13 = lshr <2 x i1> %vec1, %vec2
  ret void
}
