; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-SPV-IR
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

@v1 = addrspace(1) global i32 42, !spirv.Decorations !2
@v2 = addrspace(1) global float 1.0, !spirv.Decorations !4

; CHECK-SPIRV: Decorate [[PId1:[0-9]+]] Constant
; CHECK-SPIRV: Decorate [[PId2:[0-9]+]] Constant
; CHECK-SPIRV: Decorate [[PId2]] Binding 1
; CHECK-SPIRV: Variable {{[0-9]+}} [[PId1]]
; CHECK-SPIRV: Variable {{[0-9]+}} [[PId2]]

!1 = !{i32 22}
!2 = !{!1}
!3 = !{i32 33, i32 1}
!4 = !{!1, !3}

; CHECK-SPV-IR: @v1 = addrspace(1) constant i32 42, !spirv.Decorations ![[Var1DecosId:[0-9]+]]
; CHECK-SPV-IR: @v2 = addrspace(1) constant float 1.000000e+00, !spirv.Decorations ![[Var2DecosId:[0-9]+]]
; CHECK-SPV-IR-DAG: ![[Var1DecosId]] = !{![[Deco1Id:[0-9]+]], ![[LinkageDeco1Id:[0-9]+]]}
; CHECK-SPV-IR-DAG: ![[Var2DecosId]] = !{![[Deco1Id]], ![[Deco2Id:[0-9]+]], ![[LinkageDeco2Id:[0-9]+]]}
; CHECK-SPV-IR-DAG: ![[Deco1Id]] = !{i32 22}
; CHECK-SPV-IR-DAG: ![[Deco2Id]] = !{i32 33, i32 1}
; CHECK-SPV-IR-DAG: ![[LinkageDeco1Id]] = !{i32 41, !"v1", i32 0}
; CHECK-SPV-IR-DAG: ![[LinkageDeco2Id]] = !{i32 41, !"v2", i32 0}

; CHECK-LLVM-NOT: @v1 = {{.*}}, !spirv.Decorations !{{[0-9]+}}
; CHECK-LLVM-NOT: @v2 = {{.*}}, !spirv.Decorations !{{[0-9]+}}
; CHECK-LLVM: @v1 = addrspace(1) constant i32 42
; CHECK-LLVM: @v2 = addrspace(1) constant float 1.000000e+00
