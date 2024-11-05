; Source:
; struct Example { int a; int b; };
; void test(int val) {
;     Example obj;
;     obj.b = val;
; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_untyped_pointers -o %t.spv
; RUN: spirv-val %t.spv

; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_untyped_pointers -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability UntypedPointersKHR
; CHECK-SPIRV: Extension "SPV_KHR_untyped_pointers"
; CHECK-SPIRV: TypeInt [[#IntTy:]] 32
; CHECK-SPIRV: Constant [[#IntTy]] [[#Const0:]] 0 
; CHECK-SPIRV: Constant [[#IntTy]] [[#Const1:]] 1 
; CHECK-SPIRV: TypeUntypedPointerKHR [[#UntypedPtrTy:]] 7
; CHECK-SPIRV: TypeStruct [[#StructTy:]] [[#IntTy]] [[#IntTy]]
; CHECK-SPIRV: TypePointer [[#PtrStructTy:]] 7 [[#StructTy]]

; CHECK-SPIRV: Variable [[#PtrStructTy]] [[#StructVarId:]] 7
; CHECK-SPIRV: UntypedInBoundsPtrAccessChainKHR [[#UntypedPtrTy]] [[#]] [[#StructTy]] [[#StructVarId]] [[#Const0]] [[#Const1]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%struct.Example = type { i32, i32 }

define spir_func void @test(i32 noundef %0) {
  %2 = alloca i32, align 4
  %3 = alloca %struct.Example, align 4
  store i32 %0, ptr %2, align 4
  %4 = load i32, ptr %2, align 4
; CHECK-LLVM: %[[#Str:]] = alloca %struct.Example, align 4
; CHECK-LLVM: getelementptr inbounds %struct.Example, ptr %[[#Str]], i32 0, i32 1
  %5 = getelementptr inbounds nuw %struct.Example, ptr %3, i32 0, i32 1
  store i32 %4, ptr %5, align 4
  ret void
}
