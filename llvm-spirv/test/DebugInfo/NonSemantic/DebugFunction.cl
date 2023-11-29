// Check for 2 things:
// - After round trip translation function definition has !dbg metadata attached
//   specifically if -gline-tables-only was used for Clang
// - Parent operand of DebugFunction is DebugCompilationUnit, not an OpString,
//   even if in LLVM IR it points to a DIFile instead of DICompileUnit.

// RUN: %clang_cc1 %s -cl-std=clc++ -emit-llvm-bc -triple spir -debug-info-kind=line-tables-only -O0 -o %t.bc
// RUN: llvm-spirv %t.bc --spirv-debug-info-version=nonsemantic-shader-100 -o %t.spv
// RUN: llvm-spirv %t.spv -to-text -o %t.spt
// RUN: FileCheck %s --input-file %t.spt  --check-prefix=CHECK-SPIRV

// RUN: llvm-spirv %t.bc --spirv-debug-info-version=nonsemantic-shader-200 -o %t.spv
// RUN: llvm-spirv %t.spv -to-text -o %t.spt
// RUN: FileCheck %s --input-file %t.spt  --check-prefix=CHECK-SPIRV

// RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

float foo(int i) {
    return i * 3.14;
}
void kernel k() {
    float a = foo(2);
}

// CHECK-SPIRV-DAG: String [[foo:[0-9]+]] "foo"
// CHECK-SPIRV-DAG: String [[#EmptyStr:]] ""
// CHECK-SPIRV-DAG: String [[k:[0-9]+]] "k"
// CHECK-SPIRV-DAG: String [[#CV:]] "{{.*}}clang version [[#]].[[#]].[[#]]
// CHECK-SPIRV: [[#CU:]] [[#]] DebugCompilationUnit
// CHECK-SPIRV: [[#FuncFoo:]] [[#]] DebugFunction [[foo]] {{.*}} [[#CU]]
// CHECK-SPIRV: [[#FuncK:]] [[#]] DebugFunction [[k]] {{.*}} [[#CU]]
// CHECK-SPIRV: DebugEntryPoint [[#FuncK]] [[#CU]] [[#CV]] [[#EmptyStr]] {{$}}
// CHECK-SPIRV-NOT: DebugEntryPoint
// CHECK-SPIRV-NOT: DebugFunctionDefinition

// CHECK-SPIRV: Function {{[0-9]+}} [[#foo_id:]]
// CHECK-SPIRV: DebugFunctionDefinition [[#FuncFoo]] [[#foo_id]]
// CHECK-LLVM: define spir_func float @_Z3fooi(i32 %i) #{{[0-9]+}} !dbg ![[#foo_id:]] {

// CHECK-SPIRV: Function {{[0-9]+}} [[#k_id:]]
// CHECK-SPIRV: DebugFunctionDefinition [[#FuncK]] [[#k_id]]
// CHECK-LLVM: define spir_kernel void @k() #{{[0-9]+}} !dbg ![[#k_id:]]

// CHECK-LLVM: ![[#foo_id]] = distinct !DISubprogram(name: "foo"
// CHECK-LLVM-SAME: spFlags: DISPFlagDefinition,
// CHECK-LLVM: ![[#k_id]] = distinct !DISubprogram(name: "k"
// CHECK-LLVM-SAME: spFlags: DISPFlagDefinition | DISPFlagMainSubprogram,
