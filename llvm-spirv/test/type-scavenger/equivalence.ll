; Test if llvm-spirv type scavenging has an assertion 
; failure due to incorrect lookup of an equivalence class leader.

; RUN: llvm-as < %s | llvm-spirv -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -o - -to-text | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o - -r | llvm-dis | FileCheck %s --check-prefix=CHECK-LLVM

; Incorrect lookup of equivalence class leader caused an assertion failure when
; processing call instruction to this name
; CHECK-SPIRV: _func0
; CHECK-LLVM: _func0

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func void @_func1() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %call3 = call spir_func ptr addrspace(4) @_func2()
  %call5 = call spir_func ptr addrspace(4) @_func0(ptr addrspace(4) %call3, i64 0)
  br label %for.cond
}

define spir_func void @_func3() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %call3 = call spir_func ptr @_func4()
  %call3.ascast = addrspacecast ptr %call3 to ptr addrspace(4)
  %call5 = call spir_func ptr addrspace(4) @_func0(ptr addrspace(4) %call3.ascast, i64 0)
  br label %for.cond
}

declare spir_func ptr addrspace(4) @_func5()

define spir_func void @_func6(ptr addrspace(4) %call3.ascast) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %call5 = call spir_func ptr addrspace(4) @_func0(ptr addrspace(4) %call3.ascast, i64 0)
  br label %for.cond
}

define spir_func void @_func7() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %call3 = call spir_func ptr addrspace(4) @_func5()
  %call5 = call spir_func ptr addrspace(4) @_func0(ptr addrspace(4) %call3, i64 0)
  br label %for.cond
}

declare spir_func ptr @_func4()

declare spir_func ptr addrspace(4) @_func2()

define spir_func ptr addrspace(4) @_func0(ptr addrspace(4) %this, i64 %index) {
entry:
  %arrayidx = getelementptr [5 x i32], ptr addrspace(4) %this, i64 0, i64 %index
  ret ptr addrspace(4) null
}

define spir_func void @_func8() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %call8 = call spir_func ptr addrspace(4) @_func0(ptr addrspace(4) null, i64 0)
  br label %for.cond
}

; uselistorder directives
uselistorder ptr @_func0, { 0, 4, 3, 2, 1 }
