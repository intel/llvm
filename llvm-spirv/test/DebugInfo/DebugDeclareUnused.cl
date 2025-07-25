// Check that we can translate llvm.dbg.declare for a local variable which was
// deleted by mem2reg pass(disabled by default in llvm-spirv)

// RUN: %clang_cc1 %s -triple spir -disable-llvm-passes -debug-info-kind=standalone -emit-llvm-bc -o - | llvm-spirv -spirv-mem2reg -o %t.spv
// RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

// RUN: %clang_cc1 %s -triple spir -disable-llvm-passes -debug-info-kind=standalone -emit-llvm-bc -o - | llvm-spirv -spirv-mem2reg -o %t.spv
// RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM


void foo() {
  int a;
}

// CHECK-SPIRV: ExtInst [[#]] [[#InfoNone:]] [[#]] DebugInfoNone
// CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] DebugDeclare [[#]] [[#InfoNone]] [[#]]
// CHECK-LLVM:  #dbg_declare(ptr null, ![[#]], !DIExpression(DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef), ![[#]])
