; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv

; This is a regression test for reported issue https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/524.
; Test checks that reverse translation will not fail with assertion.

; Build from the following source with clang -c -emit-llvm -O0 -target spir64 -gline-tables-only
; float bar(int x);

; __kernel void foo(__global float* outPtr, int i) {
; #pragma clang loop unroll(enable)
;   for (int j = 0; j < i; ++j) {
;     outPtr[j] = bar(j);
;   }
; }

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @foo(ptr addrspace(1) %outPtr, i32 %i) #0 !dbg !9 !kernel_arg_addr_space !6 !kernel_arg_access_qual !11 !kernel_arg_type !12 !kernel_arg_base_type !12 !kernel_arg_type_qual !13 {
entry:
  %outPtr.addr = alloca ptr addrspace(1), align 8
  %i.addr = alloca i32, align 4
  %j = alloca i32, align 4
  store ptr addrspace(1) %outPtr, ptr %outPtr.addr, align 8
  store i32 %i, ptr %i.addr, align 4
  store i32 0, ptr %j, align 4, !dbg !14
  br label %for.cond, !dbg !15

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %j, align 4, !dbg !16
  %1 = load i32, ptr %i.addr, align 4, !dbg !17
  %cmp = icmp slt i32 %0, %1, !dbg !18
  br i1 %cmp, label %for.body, label %for.end, !dbg !19

for.body:                                         ; preds = %for.cond
  %2 = load i32, ptr %j, align 4, !dbg !20
  %call = call spir_func float @bar(i32 %2) #2, !dbg !21
  %3 = load ptr addrspace(1), ptr %outPtr.addr, align 8, !dbg !22
  %4 = load i32, ptr %j, align 4, !dbg !23
  %idxprom = sext i32 %4 to i64, !dbg !22
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %3, i64 %idxprom, !dbg !22
  store float %call, ptr addrspace(1) %arrayidx, align 4, !dbg !24
  br label %for.inc, !dbg !25

for.inc:                                          ; preds = %for.body
  %5 = load i32, ptr %j, align 4, !dbg !26
  %inc = add nsw i32 %5, 1, !dbg !26
  store i32 %inc, ptr %j, align 4, !dbg !26
  br label %for.cond, !dbg !19, !llvm.loop !27

for.end:                                          ; preds = %for.cond
  ret void, !dbg !29
}

; Function Attrs: convergent
declare dso_local spir_func float @bar(i32) #1

attributes #0 = { convergent noinline norecurse nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!opencl.ocl.version = !{!6}
!opencl.spir.version = !{!7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 6671a81bc71cc2635c5a10d6f688fea46ca4e5d6)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "loop.cl", directory: "/export/users/work/khr_spirv/llvm/build/bin")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, i32 0}
!7 = !{i32 1, i32 2}
!8 = !{!"clang version 11.0.0"}
!9 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !2)
!11 = !{!"none", !"none"}
!12 = !{!"float*", !"int"}
!13 = !{!"", !""}
!14 = !DILocation(line: 5, column: 12, scope: !9)
!15 = !DILocation(line: 5, column: 8, scope: !9)
!16 = !DILocation(line: 5, column: 19, scope: !9)
!17 = !DILocation(line: 5, column: 23, scope: !9)
!18 = !DILocation(line: 5, column: 21, scope: !9)
!19 = !DILocation(line: 5, column: 3, scope: !9)
!20 = !DILocation(line: 6, column: 21, scope: !9)
!21 = !DILocation(line: 6, column: 17, scope: !9)
!22 = !DILocation(line: 6, column: 5, scope: !9)
!23 = !DILocation(line: 6, column: 12, scope: !9)
!24 = !DILocation(line: 6, column: 15, scope: !9)
!25 = !DILocation(line: 7, column: 3, scope: !9)
!26 = !DILocation(line: 5, column: 26, scope: !9)
!27 = distinct !{!27, !19, !25, !28}
!28 = !{!"llvm.loop.unroll.enable"}
!29 = !DILocation(line: 8, column: 1, scope: !9)
