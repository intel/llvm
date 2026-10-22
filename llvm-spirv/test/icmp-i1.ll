; icmp relational predicates on i1/<N x i1> have no OpBool-typed opcode in SPIR-V.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -s %t.bc -o %t.regularized.bc
; RUN: llvm-dis %t.regularized.bc -o %t.regularized.ll
; RUN: FileCheck < %t.regularized.ll %s --check-prefix=CHECK-LOWER

; RUN: llvm-spirv %t.regularized.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.regularized.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV-NOT: {{U|S}}{{Greater|Less}}Than
; CHECK-SPIRV-COUNT-2: LogicalAnd
; CHECK-SPIRV-COUNT-2:  LogicalOr
; CHECK-SPIRV-COUNT-2:  LogicalAnd
; CHECK-SPIRV-COUNT-2:  LogicalOr
; CHECK-SPIRV:  LogicalAnd

; ugt/slt -> p & !q, ult/sgt -> q & !p, uge/sle -> p | !q, ule/sge -> q | !p
; CHECK-LOWER: %[[#NOT0:]] = xor i1 %q, true, !dbg [[DBG:![0-9]+]]
; CHECK-LOWER: %c0 = and i1 %p, %[[#NOT0]], !dbg [[DBG]]
; CHECK-LOWER: %[[#NOT1:]] = xor i1 %p, true
; CHECK-LOWER: %c1 = and i1 %q, %[[#NOT1]]
; CHECK-LOWER: %[[#NOT2:]] = xor i1 %q, true
; CHECK-LOWER: %c2 = or i1 %p, %[[#NOT2]]
; CHECK-LOWER: %[[#NOT3:]] = xor i1 %p, true
; CHECK-LOWER: %c3 = or i1 %q, %[[#NOT3]]
; CHECK-LOWER: %[[#NOT4:]] = xor i1 %p, true
; CHECK-LOWER: %c4 = and i1 %q, %[[#NOT4]]
; CHECK-LOWER: %[[#NOT5:]] = xor i1 %q, true
; CHECK-LOWER: %c5 = and i1 %p, %[[#NOT5]]
; CHECK-LOWER: %[[#NOT6:]] = xor i1 %p, true
; CHECK-LOWER: %c6 = or i1 %q, %[[#NOT6]]
; CHECK-LOWER: %[[#NOT7:]] = xor i1 %q, true
; CHECK-LOWER: %c7 = or i1 %p, %[[#NOT7]]
; CHECK-LOWER: xor <2 x i1> %vq, splat (i1 true)
; CHECK-LOWER: %vc0 = and <2 x i1> %vp,

; CHECK-LLVM: define spir_kernel void @test(i1 [[P:%[a-zA-Z0-9_.]+]], i1 [[Q:%[a-zA-Z0-9_.]+]], ptr addrspace(1) %{{[a-zA-Z0-9_.]+}}, <2 x i1> [[VP:%[a-zA-Z0-9_.]+]],
; CHECK-LLVM: and i1 [[P]], {{.*}}
; CHECK-LLVM: and i1 [[Q]], {{.*}}
; CHECK-LLVM: or i1 [[P]], {{.*}}
; CHECK-LLVM: or i1 [[Q]], {{.*}}
; CHECK-LLVM: and i1 [[Q]], {{.*}}
; CHECK-LLVM: and i1 [[P]], {{.*}}
; CHECK-LLVM: or i1 [[Q]], {{.*}}
; CHECK-LLVM: or i1 [[P]], {{.*}}
; CHECK-LLVM: and <2 x i1> [[VP]],

define spir_kernel void @test(i1 %p, i1 %q, ptr addrspace(1) %out, <2 x i1> %vp, <2 x i1> %vq, ptr addrspace(1) %vout) !dbg !4 {
entry:
  %c0 = icmp ugt i1 %p, %q, !dbg !7
  %e0 = zext i1 %c0 to i32
  %o0 = getelementptr i32, ptr addrspace(1) %out, i32 0
  store i32 %e0, ptr addrspace(1) %o0, align 4

  %c1 = icmp ult i1 %p, %q
  %e1 = zext i1 %c1 to i32
  %o1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  store i32 %e1, ptr addrspace(1) %o1, align 4

  %c2 = icmp uge i1 %p, %q
  %e2 = zext i1 %c2 to i32
  %o2 = getelementptr i32, ptr addrspace(1) %out, i32 2
  store i32 %e2, ptr addrspace(1) %o2, align 4

  %c3 = icmp ule i1 %p, %q
  %e3 = zext i1 %c3 to i32
  %o3 = getelementptr i32, ptr addrspace(1) %out, i32 3
  store i32 %e3, ptr addrspace(1) %o3, align 4

  %c4 = icmp sgt i1 %p, %q
  %e4 = zext i1 %c4 to i32
  %o4 = getelementptr i32, ptr addrspace(1) %out, i32 4
  store i32 %e4, ptr addrspace(1) %o4, align 4

  %c5 = icmp slt i1 %p, %q
  %e5 = zext i1 %c5 to i32
  %o5 = getelementptr i32, ptr addrspace(1) %out, i32 5
  store i32 %e5, ptr addrspace(1) %o5, align 4

  %c6 = icmp sge i1 %p, %q
  %e6 = zext i1 %c6 to i32
  %o6 = getelementptr i32, ptr addrspace(1) %out, i32 6
  store i32 %e6, ptr addrspace(1) %o6, align 4

  %c7 = icmp sle i1 %p, %q
  %e7 = zext i1 %c7 to i32
  %o7 = getelementptr i32, ptr addrspace(1) %out, i32 7
  store i32 %e7, ptr addrspace(1) %o7, align 4

  %vc0 = icmp ugt <2 x i1> %vp, %vq
  %ve0 = zext <2 x i1> %vc0 to <2 x i32>
  store <2 x i32> %ve0, ptr addrspace(1) %vout, align 8

  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "icmp-i1.ll", directory: ".")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test", scope: !1, file: !1, unit: !0, type: !5)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DILocation(line: 1, column: 1, scope: !4)
