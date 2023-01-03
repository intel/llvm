// Check that we can translate llvm.dbg.declare for a local variable which was
// deleted by mem2reg pass(disabled by default in llvm-spirv)

// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: %clang_cc1 %s -triple spir -disable-llvm-passes -debug-info-kind=standalone -emit-llvm-bc -o - | llvm-spirv -opaque-pointers -spirv-mem2reg -o %t.spv
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers -r -emit-opaque-pointers %t.spv -o - | llvm-dis -opaque-pointers -o - | FileCheck %s --check-prefix=CHECK-LLVM


void foo() {
  int a;
}

// CHECK-SPIRV: Undef [[#]] [[#Undef:]]
// CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] DebugDeclare [[#]] [[#Undef]] [[#]]
// CHECK-LLVM: call void @llvm.dbg.declare(metadata ptr undef, metadata ![[#]], metadata !DIExpression({{.*}}))
