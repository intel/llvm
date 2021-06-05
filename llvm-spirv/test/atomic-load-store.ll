; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s

; CHECK-DAG: Constant [[#]] [[#Relaxed:]] 0
; CHECK-DAG: Constant [[#]] [[#DeviceScope:]] 1
; CHECK-DAG: Constant [[#]] [[#Acquire:]] 2
; CHECK-DAG: Constant [[#]] [[#Release:]] 4
; CHECK-DAG: Constant [[#]] [[#SequentiallyConsistent:]] 16

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; Function Attrs: nounwind
define dso_local spir_func void @test() {
entry:
; CHECK: Variable [[#]] [[#PTR:]]
  %0 = alloca i32

; CHECK: AtomicStore [[#PTR]] [[#DeviceScope]] [[#Relaxed]] [[#]]
  store atomic i32 0, i32* %0 monotonic, align 4
; CHECK: AtomicStore [[#PTR]] [[#DeviceScope]] [[#Release]] [[#]]
  store atomic i32 0, i32* %0 release, align 4
; CHECK: AtomicStore [[#PTR]] [[#DeviceScope]] [[#SequentiallyConsistent]] [[#]]
  store atomic i32 0, i32* %0 seq_cst, align 4

; CHECK: AtomicLoad [[#]] [[#]] [[#PTR]] [[#DeviceScope]] [[#Relaxed]]
  %1 = load atomic i32, i32* %0 monotonic, align 4
; CHECK: AtomicLoad [[#]] [[#]] [[#PTR]] [[#DeviceScope]] [[#Acquire]]
  %2 = load atomic i32, i32* %0 acquire, align 4
; CHECK: AtomicLoad [[#]] [[#]] [[#PTR]] [[#DeviceScope]] [[#SequentiallyConsistent]]
  %3 = load atomic i32, i32* %0 seq_cst, align 4
  ret void
}
