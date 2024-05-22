// Check that we can translate llvm.dbg.declare for a local variable which was
// deleted by mem2reg pass(disabled by default in llvm-spirv)

// RUN: %clang_cc1 %s -triple spir -disable-llvm-passes -debug-info-kind=standalone -emit-llvm-bc -o - | llvm-spirv -spirv-mem2reg -o %t.spv
// RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

// RUN: %clang_cc1 %s -triple spir -disable-llvm-passes -debug-info-kind=standalone -emit-llvm-bc -o - | llvm-spirv -spirv-mem2reg -o %t.spv
// RUN: llvm-spirv --experimental-debuginfo-iterators=1 %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv -r --experimental-debuginfo-iterators=1 %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM


void foo() {
  int a;
}

// CHECK-SPIRV: Undef [[#]] [[#Undef:]]
// CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] DebugDeclare [[#]] [[#Undef]] [[#]]
// CHECK-LLVM: call void @llvm.dbg.declare(metadata ptr undef, metadata ![[#]], metadata !DIExpression({{.*}}))
