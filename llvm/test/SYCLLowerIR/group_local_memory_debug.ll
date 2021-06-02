; RUN: opt -S -sycllowerwglocalmemory < %s | FileCheck %s

; CHECK: @[[WGLOCALMEM:WGLocalMem.*]] = internal addrspace(3) global [128 x i8] undef, align 4, !dbg ![[DI_WGLOCMEM:[0-9]+]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

%struct._ZTS3Foo.Foo = type { [32 x i32] }

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTS7KernelA() local_unnamed_addr #0 !dbg !35 {
; CHECK: define{{.*}} spir_kernel void @_ZTS7KernelA(){{.*}} !dbg ![[DI_SP:[0-9]+]]
entry:
  %0 = tail call spir_func i8 addrspace(3)* @__sycl_allocateLocalMemory(i64 128, i64 4) #2, !dbg !37
  %1 = bitcast i8 addrspace(3)* %0 to %struct._ZTS3Foo.Foo addrspace(3)*, !dbg !39
  %2 = getelementptr inbounds i8, i8 addrspace(3)* %0, i64 4, !dbg !41
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i8 addrspace(3)* @__sycl_allocateLocalMemory(i64, i64) local_unnamed_addr #1

attributes #0 = { convergent norecurse "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!28, !29, !30, !31}
!opencl.spir.version = !{!32}
!spirv.Source = !{!33}
!llvm.ident = !{!34}

; CHECK: ![[DI_WGLOCMEM]] = !DIGlobalVariableExpression(var: ![[DI_GV:[0-9]+]], expr: !DIExpression())
; CHECK: ![[DI_GV]] = distinct !DIGlobalVariable(name: "[[WGLOCALMEM]]", scope: ![[DI_SP]], file: ![[DI_FILE:[0-9]+]], line: 45, type: ![[DI_TY_ARR:[0-9]+]], isLocal: true, isDefinition: true, align: 32)
; CHECK: ![[DI_SP]] = distinct !DISubprogram(name: "_ZTS7KernelA", scope: ![[DI_FILE]], file: ![[DI_FILE]],{{.*}} unit: ![[DI_UNIT:[0-9]+]]
; CHECK: ![[DI_FILE]] = !DIFile(filename: "group_local_memory_debug.cpp", directory: "/home/user/test")
; CHECK: ![[DI_UNIT]] = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: ![[DI_FILE]],{{.*}} globals: ![[DI_GVEs:[0-9]+]]
; CHECK: ![[DI_GVEs]] = !{{{.*}}![[DI_WGLOCMEM]]{{[^0-9]}}
; CHECK: ![[DI_TY_U8:.*]] = !DIDerivedType(tag: DW_TAG_typedef, name: "uint8_t"
; CHECK: ![[DI_TY_ARR]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[DI_TY_U8]], size: 1024, align: 32, elements: ![[DI_ELEM:.*]])
; CHECK: ![[DI_ELEM]] = !{![[DI_SR:.*]]}
; CHECK: ![[DI_SR]] = !DISubrange(count: 128)

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 13.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, globals: !12, imports: !19, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "group_local_memory_debug.cpp", directory: "/home/user/test")
!2 = !{}
!3 = !{!4, !5}
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !1, line: 13, size: 1024, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTS3Foo")
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "Values", scope: !6, file: !1, line: 20, baseType: !9, size: 1024)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 1024, elements: !10)
!10 = !{!11}
!11 = !DISubrange(count: 32)
!12 = !{!13}
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression(DW_OP_constu, 32, DW_OP_stack_value))
!14 = distinct !DIGlobalVariable(name: "WgSize", scope: !0, file: !1, line: 9, type: !15, isLocal: true, isDefinition: true)
!15 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !16)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !17, line: 46, baseType: !18)
!17 = !DIFile(filename: "build/lib/clang/13.0.0/include/stddef.h", directory: "/usr")
!18 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!19 = !{!20}
!20 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !21, entity: !22, file: !27, line: 66)
!21 = !DINamespace(name: "std", scope: null)
!22 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint8_t", file: !23, line: 24, baseType: !24)
!23 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/stdint-uintn.h", directory: "")
!24 = !DIDerivedType(tag: DW_TAG_typedef, name: "__uint8_t", file: !25, line: 37, baseType: !26)
!25 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types.h", directory: "")
!26 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!27 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/10.1.0/../../../../include/c++/10.1.0/cstdint", directory: "")
!28 = !{i32 7, !"Dwarf Version", i32 4}
!29 = !{i32 2, !"Debug Info Version", i32 3}
!30 = !{i32 1, !"wchar_size", i32 4}
!31 = !{i32 7, !"frame-pointer", i32 2}
!32 = !{i32 1, i32 2}
!33 = !{i32 4, i32 100000}
!34 = !{!"clang version 13.0.0"}
!35 = distinct !DISubprogram(name: "_ZTS7KernelA", scope: !1, file: !1, line: 43, type: !36, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!36 = !DISubroutineType(cc: DW_CC_LLVM_OpenCLKernel, types: !2)
!37 = !DILocation(line: 50, column: 7, scope: !42, inlinedAt: !38)
!38 = distinct !DILocation(line: 45, column: 17, scope: !35)
!39 = !DILocation(line: 55, column: 5, scope: !40, inlinedAt: !38)
!40 = distinct !DILexicalBlock(scope: !42, file: !45, line: 54, column: 7)
!41 = !DILocation(line: 48, column: 57, scope: !35)
!42 = distinct !DISubprogram(name: "group_local_memory<Foo, sycl::group<1>, int, int &>", linkageName: "_ZN2cl4sycl18group_local_memoryI3FooNS0_5groupILi1EEEJiRiEEENSt9enable_ifIXaasr3std25is_trivially_destructibleIT_EE5valuesr6detail8is_groupIT0_EE5valueENS0_9multi_ptrIS7_LNS0_6access13address_spaceE3EEEE4typeES8_DpOT1_", scope: !43, file: !45, line: 46, type: !46, scopeLine: 46, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!43 = !DINamespace(name: "sycl", scope: !44)
!44 = !DINamespace(name: "cl", scope: null, exportSymbols: true)
!45 = !DIFile(filename: "build/bin/../include/sycl/CL/sycl/group_local_memory.hpp", directory: "/home/user/sycl")
!46 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !47)
!47 = !{!48, !50, !51}
!48 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "group<1>", scope: !43, file: !49, line: 92, size: 256, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !2, identifier: "_ZTSN2cl4sycl5groupILi1EEE")
!49 = !DIFile(filename: "build/bin/../include/sycl/CL/sycl/group.hpp", directory: "/home/user/sycl")
!50 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !4, size: 64)
!51 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !4, size: 64)
