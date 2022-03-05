; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv --to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefixes=CHECK,CHECK-SPV

; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_reg -o %t.fpga_reg.spv
; RUN: llvm-spirv %t.fpga_reg.spv --to-text -o %t.fpga_reg.spt
; RUN: FileCheck < %t.fpga_reg.spt %s --check-prefixes=CHECK,CHECK-SPV-FPGA_REG

; -- Check that reverse translation is not failed.
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-spirv -r %t.fpga_reg.spv -o %t.rev.fpga_reg.bc

; ModuleID = 'annotation_dbg_info_drop.cpp'
source_filename = "annotation_dbg_info_drop.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" = type { i64 }
%"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_d.s_d" = type { [8 x i32] }
%"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_e.s_e" = type { [4 x i32] }

$"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE11test_kernel" = comdat any

@.str = private unnamed_addr constant [25 x i8] c"{memory:DEFAULT}{pump:1}\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [29 x i8] c"annotation_dbg_info_drop.cpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr constant [25 x i8] c"__builtin_intel_fpga_reg\00", section "llvm.metadata"
@.str.3 = private unnamed_addr constant [29 x i8] c"{memory:DEFAULT}{numbanks:2}\00", section "llvm.metadata"

; CHECK: {{[0-9]+}} Name [[ANNO_PUMP_ID:[0-9]+]] ".str"
; CHECK: {{[0-9]+}} Name [[FILE_ID:[0-9]+]] ".str.1"
; CHECK: {{[0-9]+}} Name [[FPGA_REG_ID:[0-9]+]] ".str.2"
; CHECK: {{[0-9]+}} Name [[ANNO_NUMBANKS_ID:[0-9]+]] ".str.3"
; CHECK: {{[0-9]+}} Name [[FUNC_ID:[0-9]+]] "_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEv"
; CHECK: {{[0-9]+}} Name [[S_E_STRUCT_ID:[0-9]+]] "struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_e.s_e"
; CHECK: {{[0-9]+}} Name [[F_ID:[0-9]+]] "_Z1fv"

; CHECK: {{[0-9]+}} TypePointer [[S_E_STRUCT_PTR_ID:[0-9]+]] {{[0-9]+}} [[S_E_STRUCT_ID]]

; Function Attrs: inlinehint nounwind
define internal spir_func void @"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEv"(%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %this) #3 align 2 !dbg !48 {
entry:
  %this.addr = alloca %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)*, align 8
  %Buf = alloca [1 x i8], align 1
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i64, align 8
  %d = alloca %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_d.s_d", align 4
  %e = alloca %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_e.s_e", align 4
; CHECK: {{[0-9]+}} Variable [[S_E_STRUCT_PTR_ID]] [[VAR_E_ID:[0-9]+]]
  %f = alloca %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_e.s_e", align 4
; CHECK: {{[0-9]+}} Variable [[S_E_STRUCT_PTR_ID]] [[VAR_F_ID:[0-9]+]]
  %agg-temp = alloca %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_e.s_e", align 4
; CHECK: {{[0-9]+}} Variable [[S_E_STRUCT_PTR_ID]] [[AGG_TMP_ID:[0-9]+]]
  store %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %this, %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)** %this.addr, align 8, !tbaa !69
  call void @llvm.dbg.declare(metadata %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)** %this.addr, metadata !50, metadata !DIExpression()), !dbg !71
  %this1 = load %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)*, %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)** %this.addr, align 8
; CHECK:      {{[0-9]+}} ExtInst {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} DebugNoScope
  %0 = bitcast [1 x i8]* %Buf to i8*, !dbg !72
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %0) #5, !dbg !72
  call void @llvm.dbg.declare(metadata [1 x i8]* %Buf, metadata !52, metadata !DIExpression()), !dbg !73
  %Buf2 = bitcast [1 x i8]* %Buf to i8*, !dbg !72
  call void @llvm.var.annotation(i8* %Buf2, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.1, i32 0, i32 0), i32 15, i8* null), !dbg !72
; CHECK:      {{[0-9]+}} ExtInst {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} DebugNoScope
; CHECK-NEXT: {{[0-9]+}} InBoundsPtrAccessChain {{[0-9]+}} {{[0-9]+}} [[ANNO_PUMP_ID]]
; CHECK-NEXT: {{[0-9]+}} InBoundsPtrAccessChain {{[0-9]+}} {{[0-9]+}} [[FILE_ID]]
  call spir_func void @_Z1fv(), !dbg !74
; -- var.annotation call is dropped. Restore debug scope after the call.
; CHECK-NEXT: {{[0-9]+}} ExtInst {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} DebugScope
; CHECK-NEXT: {{[0-9]+}} Line
; CHECK-NEXT: {{[0-9]+}} FunctionCall {{[0-9]+}} {{[0-9]+}} [[F_ID]]
  %1 = bitcast i32* %a to i8*, !dbg !75
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1) #5, !dbg !75
  call void @llvm.dbg.declare(metadata i32* %a, metadata !53, metadata !DIExpression()), !dbg !76
  store i32 123, i32* %a, align 4, !dbg !76, !tbaa !77
  %2 = bitcast i32* %b to i8*, !dbg !79
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %2) #5, !dbg !79
  call void @llvm.dbg.declare(metadata i32* %b, metadata !54, metadata !DIExpression()), !dbg !80
  %3 = load i32, i32* %a, align 4, !dbg !81, !tbaa !77
  %4 = call i32 @llvm.annotation.i32(i32 %3, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.2, i32 0, i32 0), i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.1, i32 0, i32 0), i32 18), !dbg !82
; CHECK:      {{[0-9]+}} ExtInst {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} DebugNoScope
; CHECK-NEXT: {{[0-9]+}} InBoundsPtrAccessChain {{[0-9]+}} {{[0-9]+}} [[FPGA_REG_ID]]
  store i32 %4, i32* %b, align 4, !dbg !80, !tbaa !77
; -- Restore debug scope after the call in both cases with or without SPV_INTEL_fpga_reg extension.
; CHECK-NEXT: {{[0-9]+}} ExtInst {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} DebugScope
; CHECK-NEXT: {{[0-9]+}} Line
; CHECK-SPV-NEXT: {{[0-9]+}} Store
; CHECK-SPV-FPGA_REG-NEXT: {{[0-9]+}} FPGARegINTEL
  %5 = bitcast i64* %c to i8*, !dbg !83
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %5) #5, !dbg !83
  call void @llvm.dbg.declare(metadata i64* %c, metadata !55, metadata !DIExpression()), !dbg !84
  %6 = getelementptr inbounds %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon", %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %this1, i32 0, i32 0, !dbg !85
; CHECK:          {{[0-9]+}} InBoundsPtrAccessChain
  %7 = load i64, i64 addrspace(4)* %6, align 8, !dbg !85, !tbaa !44
; -- annotation call is dropped. No debug scope or line change for first argument declaration.
; CHECK-NEXT:     {{[0-9]+}} Load
; CHECK-NEXT:     {{[0-9]+}} Line
  %8 = call i64 @llvm.annotation.i64(i64 %7, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.2, i32 0, i32 0), i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.1, i32 0, i32 0), i32 19), !dbg !86
; CHECK-SPV-FPGA_REG-NEXT: {{[0-9]+}} FPGARegINTEL
  store i64 %8, i64* %c, align 8, !dbg !84, !tbaa !39
; CHECK-SPV-NEXT: {{[0-9]+}} Store
  %9 = bitcast %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_d.s_d"* %d to i8*, !dbg !87
  call void @llvm.lifetime.start.p0i8(i64 32, i8* %9) #5, !dbg !87
  call void @llvm.dbg.declare(metadata %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_d.s_d"* %d, metadata !56, metadata !DIExpression()), !dbg !88
  %mem = getelementptr inbounds %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_d.s_d", %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_d.s_d"* %d, i32 0, i32 0, !dbg !89
  %10 = bitcast [8 x i32]* %mem to i8*, !dbg !89
  %11 = call i8* @llvm.ptr.annotation.p0i8(i8* %10, i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.3, i32 0, i32 0), i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.1, i32 0, i32 0), i32 21, i8* null), !dbg !89
; CHECK:      {{[0-9]+}} ExtInst {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} DebugNoScope
; CHECK-NEXT: {{[0-9]+}} InBoundsPtrAccessChain {{[0-9]+}} {{[0-9]+}} [[ANNO_NUMBANKS_ID]]
  %12 = bitcast i8* %11 to [8 x i32]*, !dbg !89
; -- annotation call is dropped. Restore debug scope
; CHECK-NEXT: {{[0-9]+}} ExtInst {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} DebugScope
; CHECK-NEXT: {{[0-9]+}} Bitcast
  %arrayidx = getelementptr inbounds [8 x i32], [8 x i32]* %12, i64 0, i64 0, !dbg !90
  store i32 42, i32* %arrayidx, align 4, !dbg !91, !tbaa !77
  %13 = bitcast %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_e.s_e"* %e to i8*, !dbg !92
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %13) #5, !dbg !92
  call void @llvm.dbg.declare(metadata %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_e.s_e"* %e, metadata !63, metadata !DIExpression()), !dbg !93
  %14 = bitcast %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_e.s_e"* %f to i8*, !dbg !94
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %14) #5, !dbg !94
  call void @llvm.dbg.declare(metadata %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_e.s_e"* %f, metadata !68, metadata !DIExpression()), !dbg !95
  %15 = bitcast %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_e.s_e"* %agg-temp to i8*, !dbg !96
; CHECK:      {{[0-9]+}} Bitcast {{[0-9]+}} {{[0-9]+}} [[AGG_TMP_ID]]
  %16 = bitcast %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_e.s_e"* %e to i8*, !dbg !96
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %15, i8* align 4 %16, i64 16, i1 false), !dbg !96, !tbaa.struct !97
  %17 = bitcast %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_e.s_e"* %agg-temp to i8*, !dbg !99
; CHECK:      {{[0-9]+}} Bitcast {{[0-9]+}} {{[0-9]+}} [[AGG_TMP_ID]]
  %18 = call i8* @llvm.ptr.annotation.p0i8(i8* %17, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.2, i32 0, i32 0), i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.1, i32 0, i32 0), i32 27, i8* null), !dbg !99
; -- No change of debug scope after the call in both cases with or without SPV_INTEL_fpga_reg extension.
; CHECK-SPV-FPGA_REG-NEXT: {{[0-9]+}} FPGARegINTEL
  %19 = bitcast i8* %18 to %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_e.s_e"*, !dbg !99
; CHECK-NEXT: {{[0-9]+}} Bitcast
  %20 = bitcast %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_e.s_e"* %f to i8*, !dbg !99
  %21 = bitcast %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_e.s_e"* %19 to i8*, !dbg !99
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %20, i8* align 4 %21, i64 8, i1 false), !dbg !99
  %22 = bitcast %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_e.s_e"* %f to i8*, !dbg !100
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %22) #5, !dbg !100
  %23 = bitcast %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_e.s_e"* %e to i8*, !dbg !100
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %23) #5, !dbg !100
  %24 = bitcast %"struct._ZTSZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEvE3s_d.s_d"* %d to i8*, !dbg !100
  call void @llvm.lifetime.end.p0i8(i64 32, i8* %24) #5, !dbg !100
  %25 = bitcast i64* %c to i8*, !dbg !100
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %25) #5, !dbg !100
  %26 = bitcast i32* %b to i8*, !dbg !100
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %26) #5, !dbg !100
  %27 = bitcast i32* %a to i8*, !dbg !100
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %27) #5, !dbg !100
  %28 = bitcast [1 x i8]* %Buf to i8*, !dbg !100
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %28) #5, !dbg !100
  ret void, !dbg !100
; CHECK:      {{[0-9]+}} Return{{[[:space:]]+}}
}
; -- no DebugScope instructions at the end of the function call.
; CHECK-SAME: {{[0-9]+}} FunctionEnd

define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE11test_kernel"(i64 %_arg_) #0 comdat !dbg !20 !kernel_arg_addr_space !34 !kernel_arg_access_qual !35 !kernel_arg_type !36 !kernel_arg_base_type !37 !kernel_arg_type_qual !38 {
entry:
  %_arg_.addr = alloca i64, align 8
  %0 = alloca %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon", align 8
  store i64 %_arg_, i64* %_arg_.addr, align 8, !tbaa !39
  call void @llvm.dbg.declare(metadata i64* %_arg_.addr, metadata !24, metadata !DIExpression()), !dbg !43
  %1 = bitcast %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %1) #5
  call void @llvm.dbg.declare(metadata %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0, metadata !25, metadata !DIExpression()), !dbg !43
  %2 = getelementptr inbounds %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon", %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0, i32 0, i32 0, !dbg !43
  %3 = load i64, i64* %_arg_.addr, align 8, !dbg !43, !tbaa !39
  store i64 %3, i64* %2, align 8, !dbg !43, !tbaa !44
  %4 = addrspacecast %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0 to %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)*, !dbg !46
  call spir_func void @"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEv"(%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %4), !dbg !46
  %5 = bitcast %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0 to i8*, !dbg !43
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %5) #5, !dbg !43
  ret void, !dbg !43
}

; Function Attrs: nounwind
define dso_local spir_func void @_Z1fv() #4 !dbg !101 {
entry:
  ret void, !dbg !103
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: nounwind
declare void @llvm.var.annotation(i8*, i8*, i8*, i32, i8*) #5

; Function Attrs: nounwind
declare i32 @llvm.annotation.i32(i32, i8*, i8*, i32) #5

; Function Attrs: nounwind
declare i64 @llvm.annotation.i64(i64, i8*, i8*, i32) #5

; Function Attrs: nounwind
declare i8* @llvm.ptr.annotation.p0i8(i8*, i8*, i8*, i32, i8*) #5

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #2

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }
attributes #3 = { inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15, !16}
!opencl.spir.version = !{!17}
!spirv.Source = !{!18}
!llvm.ident = !{!19}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 9.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!1 = !DIFile(filename: "annotation_dbg_info_drop.cpp", directory: "/localdisk2/test")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!4 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !5)
!5 = !{null}
!6 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !8, line: 46, baseType: !3)
!8 = !DIFile(filename: "clang/9.0.0/include/stddef.h", directory: "/usr/lib/")
!9 = !{!10}
!10 = !DISubrange(count: 4)
!11 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 8, elements: !12)
!12 = !{!13}
!13 = !DISubrange(count: 1)
!14 = !{i32 2, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{i32 1, i32 2}
!18 = !{i32 4, i32 100000}
!19 = !{!"clang version 9.0.0"}
!20 = distinct !DISubprogram(name: "_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE11test_kernel", scope: !1, file: !1, line: 14, type: !21, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !23)
!21 = !DISubroutineType(cc: DW_CC_LLVM_OpenCLKernel, types: !22)
!22 = !{null, !7}
!23 = !{!24, !25}
!24 = !DILocalVariable(name: "_arg_", arg: 1, scope: !20, file: !1, type: !7)
!25 = !DILocalVariable(scope: !20, file: !1, type: !26)
!26 = distinct !DICompositeType(tag: DW_TAG_class_type, file: !1, line: 14, size: 64, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !27)
!27 = !{!28, !29}
!28 = !DIDerivedType(tag: DW_TAG_member, name: "ga", scope: !26, file: !1, line: 19, baseType: !7, size: 64)
!29 = !DISubprogram(name: "operator()", scope: !26, file: !1, line: 14, type: !30, scopeLine: 14, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!30 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !31)
!31 = !{null, !32}
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !33, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!33 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !26)
!34 = !{i32 0}
!35 = !{!"none"}
!36 = !{!"size_t"}
!37 = !{!"ulong"}
!38 = !{!""}
!39 = !{!40, !40, i64 0}
!40 = !{!"long", !41, i64 0}
!41 = !{!"omnipotent char", !42, i64 0}
!42 = !{!"Simple C++ TBAA"}
!43 = !DILocation(line: 0, scope: !20)
!44 = !{!45, !40, i64 0}
!45 = !{!"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_", !40, i64 0}
!46 = !DILocation(line: 0, scope: !47)
!47 = distinct !DILexicalBlock(scope: !20, file: !1)
!48 = distinct !DISubprogram(name: "operator()", linkageName: "_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEv", scope: !26, file: !1, line: 14, type: !30, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !29, retainedNodes: !49)
!49 = !{!50, !52, !53, !54, !55, !56, !63, !68}
!50 = !DILocalVariable(name: "this", arg: 1, scope: !48, type: !51, flags: DIFlagArtificial | DIFlagObjectPointer)
!51 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !33, size: 64)
!52 = !DILocalVariable(name: "Buf", scope: !48, file: !1, line: 15, type: !11)
!53 = !DILocalVariable(name: "a", scope: !48, file: !1, line: 17, type: !2)
!54 = !DILocalVariable(name: "b", scope: !48, file: !1, line: 18, type: !2)
!55 = !DILocalVariable(name: "c", scope: !48, file: !1, line: 19, type: !3)
!56 = !DILocalVariable(name: "d", scope: !48, file: !1, line: 22, type: !57)
!57 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s_d", scope: !48, file: !1, line: 20, size: 256, flags: DIFlagTypePassByValue, elements: !58)
!58 = !{!59}
!59 = !DIDerivedType(tag: DW_TAG_member, name: "mem", scope: !57, file: !1, line: 21, baseType: !60, size: 256)
!60 = !DICompositeType(tag: DW_TAG_array_type, baseType: !2, size: 256, elements: !61)
!61 = !{!62}
!62 = !DISubrange(count: 8)
!63 = !DILocalVariable(name: "e", scope: !48, file: !1, line: 26, type: !64)
!64 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s_e", scope: !48, file: !1, line: 24, size: 128, flags: DIFlagTypePassByValue, elements: !65)
!65 = !{!66}
!66 = !DIDerivedType(tag: DW_TAG_member, name: "mem", scope: !64, file: !1, line: 25, baseType: !67, size: 128)
!67 = !DICompositeType(tag: DW_TAG_array_type, baseType: !2, size: 128, elements: !9)
!68 = !DILocalVariable(name: "f", scope: !48, file: !1, line: 27, type: !64)
!69 = !{!70, !70, i64 0}
!70 = !{!"any pointer", !41, i64 0}
!71 = !DILocation(line: 0, scope: !48)
!72 = !DILocation(line: 15, column: 33, scope: !48)
!73 = !DILocation(line: 15, column: 38, scope: !48)
!74 = !DILocation(line: 16, column: 7, scope: !48)
!75 = !DILocation(line: 17, column: 7, scope: !48)
!76 = !DILocation(line: 17, column: 11, scope: !48)
!77 = !{!78, !78, i64 0}
!78 = !{!"int", !41, i64 0}
!79 = !DILocation(line: 18, column: 7, scope: !48)
!80 = !DILocation(line: 18, column: 11, scope: !48)
!81 = !DILocation(line: 18, column: 40, scope: !48)
!82 = !DILocation(line: 18, column: 15, scope: !48)
!83 = !DILocation(line: 19, column: 7, scope: !48)
!84 = !DILocation(line: 19, column: 12, scope: !48)
!85 = !DILocation(line: 19, column: 41, scope: !48)
!86 = !DILocation(line: 19, column: 16, scope: !48)
!87 = !DILocation(line: 20, column: 7, scope: !48)
!88 = !DILocation(line: 22, column: 9, scope: !48)
!89 = !DILocation(line: 23, column: 9, scope: !48)
!90 = !DILocation(line: 23, column: 7, scope: !48)
!91 = !DILocation(line: 23, column: 16, scope: !48)
!92 = !DILocation(line: 24, column: 7, scope: !48)
!93 = !DILocation(line: 26, column: 9, scope: !48)
!94 = !DILocation(line: 27, column: 7, scope: !48)
!95 = !DILocation(line: 27, column: 18, scope: !48)
!96 = !DILocation(line: 27, column: 47, scope: !48)
!97 = !{i64 0, i64 16, !98}
!98 = !{!41, !41, i64 0}
!99 = !DILocation(line: 27, column: 22, scope: !48)
!100 = !DILocation(line: 28, column: 5, scope: !48)
!101 = distinct !DISubprogram(name: "f", linkageName: "_Z1fv", scope: !1, file: !1, line: 5, type: !4, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !102)
!102 = !{}
!103 = !DILocation(line: 6, column: 3, scope: !101)
