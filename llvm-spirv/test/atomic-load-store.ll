; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s

; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_KHR_untyped_pointers
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s

; CHECK-DAG: Constant [[#]] [[#CrossDeviceScope:]] 0
; CHECK-DAG: Constant [[#]] [[#Release:]] 4
; CHECK-DAG: Constant [[#]] [[#SequentiallyConsistent:]] 16
; CHECK-DAG: Constant [[#]] [[#Acquire:]] 2

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; Function Attrs: nounwind
define dso_local spir_func void @test() {
entry:
; CHECK: {{(Variable|UntypedVariableKHR)}} [[#]] [[#PTR:]]
  %0 = alloca i32

; CHECK: AtomicStore [[#PTR]] [[#CrossDeviceScope]] {{.+}} [[#]]
  store atomic i32 0, ptr %0 monotonic, align 4
; CHECK: AtomicStore [[#PTR]] [[#CrossDeviceScope]] [[#Release]] [[#]]
  store atomic i32 0, ptr %0 release, align 4
; CHECK: AtomicStore [[#PTR]] [[#CrossDeviceScope]] [[#SequentiallyConsistent]] [[#]]
  store atomic i32 0, ptr %0 seq_cst, align 4

; CHECK: AtomicLoad [[#]] [[#]] [[#PTR]] [[#CrossDeviceScope]] {{.+}}
  %1 = load atomic i32, ptr %0 monotonic, align 4
; CHECK: AtomicLoad [[#]] [[#]] [[#PTR]] [[#CrossDeviceScope]] [[#Acquire]]
  %2 = load atomic i32, ptr %0 acquire, align 4
; CHECK: AtomicLoad [[#]] [[#]] [[#PTR]] [[#CrossDeviceScope]] [[#SequentiallyConsistent]]
  %3 = load atomic i32, ptr %0 seq_cst, align 4
  ret void
}
