; Check debug info of builtin get_global_id is preserved from LLVM IR to spirv
; and spirv to LLVM IR translation.

; Original .cl source:
; kernel void test() {
;   size_t gid = get_global_id(0);
; }

; Command line:
; ./clang -cc1 1.cl -triple spir64 -cl-std=cl2.0 -emit-llvm -finclude-default-header -debug-info-kind=line-tables-only -O0

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; CHECK-SPIRV: ExtInst {{.*}} DebugScope
; CHECK-SPIRV-NEXT: Line {{[0-9]+}} 2 16
; CHECK-SPIRV-NEXT: Load {{[0-9]+}} [[LoadRes:[0-9]+]]
; CHECK-SPIRV-NEXT: CompositeExtract {{[0-9]+}} {{[0-9]+}} [[LoadRes]] 0

; Function Attrs: convergent noinline norecurse nounwind optnone
define spir_kernel void @test() #0 !dbg !7 !kernel_arg_addr_space !2 !kernel_arg_access_qual !2 !kernel_arg_type !2 !kernel_arg_base_type !2 !kernel_arg_type_qual !2 {
entry:
  %gid = alloca i64, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #2, !dbg !10
; CHECK: [[I0:%[0-9]]] = call spir_func i64 @_Z13get_global_idj(i32 0) #1, !dbg [[DBG:![0-9]+]]
; CHECK-NEXT: [[I1:%[0-9]]] = insertelement <3 x i64> undef, i64 [[I0]], i32 0, !dbg [[DBG]]
; CHECK-NEXT: [[I2:%[0-9]]] = call spir_func i64 @_Z13get_global_idj(i32 1) #1, !dbg [[DBG]]
; CHECK-NEXT: [[I3:%[0-9]]] = insertelement <3 x i64> [[I1]], i64 [[I2]], i32 1, !dbg [[DBG]]
; CHECK-NEXT: [[I4:%[0-9]]] = call spir_func i64 @_Z13get_global_idj(i32 2) #1, !dbg [[DBG]]
; CHECK-NEXT: [[I5:%[0-9]]] = insertelement <3 x i64> [[I3]], i64 [[I4]], i32 2, !dbg [[DBG]]
; CHECK-NEXT: %call = extractelement <3 x i64> [[I5]], i32 0, !dbg [[DBG]]
  store i64 %call, i64* %gid, align 8, !dbg !11
  ret void, !dbg !12
}

; Function Attrs: convergent nounwind readnone
declare spir_func i64 @_Z13get_global_idj(i32) #1

attributes #0 = { convergent noinline norecurse nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!opencl.ocl.version = !{!5}
!opencl.spir.version = !{!5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0 (https://github.com/llvm/llvm-project.git b5bc56da8aa23dc57db9d286b0591dbcf9b1bdd3)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 2, i32 0}
!6 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project.git b5bc56da8aa23dc57db9d286b0591dbcf9b1bdd3)"}
!7 = distinct !DISubprogram(name: "test", scope: !8, file: !8, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DIFile(filename: "1.cl", directory: "")
!9 = !DISubroutineType(types: !2)
!10 = !DILocation(line: 2, column: 16, scope: !7)
!11 = !DILocation(line: 2, column: 10, scope: !7)
!12 = !DILocation(line: 3, column: 1, scope: !7)
