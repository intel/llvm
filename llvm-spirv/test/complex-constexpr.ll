; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck %s --input-file %t.spt -check-prefix=CHECK-SPIRV
; RUN: FileCheck %s --input-file %t.rev.ll  -check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

@.str.1 = private unnamed_addr addrspace(1) constant [1 x i8] zeroinitializer, align 1

define linkonce_odr hidden spir_func void @foo() {
entry:
; CHECK-SPIRV: Constant [[#]] [[#MinusOne:]] 4294967295 4294967295
; CHECK-SPIRV: ConvertUToPtr [[#]] [[#Ptr:]] [[#MinusOne:]]
; CHECK-SPIRV: Bitcast [[#]] [[#BitCast:]] [[#Ptr]]
; CHECK-SPIRV: FunctionCall [[#]] [[#]] [[#]] [[#]] [[#BitCast]]

; CHECK-LLVM: %[[#Ptr:]] = inttoptr i64 -1 to ptr addrspace(4)
; CHECK-LLVM: %[[#BitCast:]] = bitcast ptr addrspace(4) %[[#Ptr]] to ptr addrspace(4)
; CHECK-LLVM: call spir_func void @bar({{.*}}, ptr addrspace(4) %[[#BitCast]]) #0

  %0 = bitcast ptr addrspace(4) inttoptr (i64 -1 to ptr addrspace(4)) to ptr addrspace(4)
  call spir_func void @bar(ptr addrspace(4) addrspacecast (ptr addrspace(1) @.str.1 to ptr addrspace(4)), ptr addrspace(4) %0)
  ret void
}

define linkonce_odr hidden spir_func void @bar(ptr addrspace(4) %__beg, ptr addrspace(4) %__end) {
entry:
  ret void
}

