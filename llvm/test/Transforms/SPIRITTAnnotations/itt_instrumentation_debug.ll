; RUN: opt < %s -SPIRITTAnnotations -S | FileCheck %s

; Verify that SPIRITTAnnotations pass inherits the debug information
; from the insertion points:
; CHECK: call void @__itt_offload_wi_start_wrapper(), !dbg ![[D1:[0-9]+]]
; CHECK: call spir_func void @foo(){{.*}}!dbg ![[D1]]
; CHECK: call void @__itt_offload_wg_barrier_wrapper(), !dbg ![[D2:[0-9]+]]
; CHECK: call spir_func void @_Z22__spirv_ControlBarrieriii(i32 2, i32 2, i32 528){{.*}}!dbg ![[D2]]
; CHECK: call void @__itt_offload_wi_resume_wrapper(), !dbg ![[D2]]
; CHECK: [[LD:%[A-Za-z0-9._]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* @res
; CHECK: [[AC1:%[A-Za-z0-9._]+]] = addrspacecast i32 addrspace(1)* [[LD]] to i8 addrspace(4)*
; CHECK: call void @__itt_offload_atomic_op_start(i8 addrspace(4)* [[AC1]], i32 2, i32 0), !dbg ![[D3:[0-9]+]]
; CHECK: %call4 = call spir_func i32 @_Z18__spirv_AtomicIAddPU3AS1iiii(i32 addrspace(1)* [[LD]], i32 2, i32 0, i32 1){{.*}}!dbg ![[D3]]
; CHECK: [[AC2:%[A-Za-z0-9._]+]] = addrspacecast i32 addrspace(1)* [[LD]] to i8 addrspace(4)*
; CHECK: call void @__itt_offload_atomic_op_finish(i8 addrspace(4)* [[AC2]], i32 2, i32 0), !dbg ![[D3]]
; CHECK: call void @__itt_offload_wi_finish_wrapper(), !dbg ![[D4:[0-9]+]]
; CHECK: ret void, !dbg ![[D4]]
; CHECK: ![[D2]] = !DILocation(line: 8, column: 3
; CHECK: ![[D3]] = !DILocation(line: 11, column: 3
; CHECK: ![[D4]] = !DILocation(line: 14, column: 1

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@.str = private unnamed_addr addrspace(2) constant [7 x i8] c"enter\0A\00", align 1
@.str.1 = private unnamed_addr addrspace(2) constant [16 x i8] c"before barrier\0A\00", align 1
@.str.2 = private unnamed_addr addrspace(2) constant [15 x i8] c"after barrier\0A\00", align 1
@.str.3 = private unnamed_addr addrspace(2) constant [15 x i8] c"before atomic\0A\00", align 1
@res = dso_local addrspace(1) global i32 addrspace(1)* null, align 8, !dbg !0
@.str.4 = private unnamed_addr addrspace(2) constant [14 x i8] c"after atomic\0A\00", align 1
@.str.5 = private unnamed_addr addrspace(2) constant [6 x i8] c"exit\0A\00", align 1

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @test() #0 !dbg !13 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 !kernel_arg_host_accessible !4 !kernel_arg_pipe_depth !4 !kernel_arg_pipe_io !4 !kernel_arg_buffer_location !4 {
entry:
  call spir_func void @foo() #2, !dbg !16
  %0 = getelementptr inbounds [7 x i8], [7 x i8] addrspace(2)* @.str, i64 0, i64 0
  %call = call spir_func i32 @_Z18__spirv_ocl_printfPU3AS2c(i8 addrspace(2)* %0) #1, !dbg !17
  %1 = getelementptr inbounds [16 x i8], [16 x i8] addrspace(2)* @.str.1, i64 0, i64 0
  %call1 = call spir_func i32 @_Z18__spirv_ocl_printfPU3AS2c(i8 addrspace(2)* %1) #1, !dbg !18
  call spir_func void @_Z22__spirv_ControlBarrieriii(i32 2, i32 2, i32 528) #1, !dbg !19
  %2 = getelementptr inbounds [15 x i8], [15 x i8] addrspace(2)* @.str.2, i64 0, i64 0
  %call2 = call spir_func i32 @_Z18__spirv_ocl_printfPU3AS2c(i8 addrspace(2)* %2) #1, !dbg !20
  %3 = getelementptr inbounds [15 x i8], [15 x i8] addrspace(2)* @.str.3, i64 0, i64 0
  %call3 = call spir_func i32 @_Z18__spirv_ocl_printfPU3AS2c(i8 addrspace(2)* %3) #1, !dbg !21
  %4 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* @res, align 8, !dbg !22
  %call4 = call spir_func i32 @_Z18__spirv_AtomicIAddPU3AS1iiii(i32 addrspace(1)* %4, i32 2, i32 0, i32 1) #1, !dbg !23
  %5 = getelementptr inbounds [14 x i8], [14 x i8] addrspace(2)* @.str.4, i64 0, i64 0
  %call5 = call spir_func i32 @_Z18__spirv_ocl_printfPU3AS2c(i8 addrspace(2)* %5) #1, !dbg !24
  %6 = getelementptr inbounds [6 x i8], [6 x i8] addrspace(2)* @.str.5, i64 0, i64 0
  %call6 = call spir_func i32 @_Z18__spirv_ocl_printfPU3AS2c(i8 addrspace(2)* %6) #1, !dbg !25
  ret void, !dbg !26
}

; Function Attrs: convergent
declare spir_func void @foo() #1

; Function Attrs: convergent
declare spir_func i32 @_Z18__spirv_ocl_printfPU3AS2c(i8 addrspace(2)*) #1

; Function Attrs: convergent
declare spir_func void @_Z22__spirv_ControlBarrieriii(i32, i32, i32) #1

; Function Attrs: convergent
declare spir_func i32 @_Z18__spirv_AtomicIAddPU3AS1iiii(i32 addrspace(1)*, i32, i32, i32) #1

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { convergent "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9}
!opencl.compiler.options = !{!4}
!llvm.ident = !{!10}
!spirv.Source = !{!11}
!spirv.MemoryModel = !{!12}
!spirv.ExecutionMode = !{}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "res", scope: !2, file: !6, line: 1, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !3, producer: "clang 11.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "<stdin>", directory: "/test")
!4 = !{}
!5 = !{!0}
!6 = !DIFile(filename: "itt.c", directory: "/test")
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang 11.0.0"}
!11 = !{i32 3, i32 200000}
!12 = !{i32 2, i32 2}
!13 = distinct !DISubprogram(name: "test", scope: !6, file: !6, line: 4, type: !14, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!14 = !DISubroutineType(cc: DW_CC_LLVM_OpenCLKernel, types: !15)
!15 = !{null}
!16 = !DILocation(line: 5, column: 3, scope: !13)
!17 = !DILocation(line: 6, column: 3, scope: !13)
!18 = !DILocation(line: 7, column: 3, scope: !13)
!19 = !DILocation(line: 8, column: 3, scope: !13)
!20 = !DILocation(line: 9, column: 3, scope: !13)
!21 = !DILocation(line: 10, column: 3, scope: !13)
!22 = !DILocation(line: 11, column: 12, scope: !13)
!23 = !DILocation(line: 11, column: 3, scope: !13)
!24 = !DILocation(line: 12, column: 3, scope: !13)
!25 = !DILocation(line: 13, column: 3, scope: !13)
!26 = !DILocation(line: 14, column: 1, scope: !13)
