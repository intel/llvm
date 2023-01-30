; RUN: llvm-as -opaque-pointers=0 %s -o %t.bc
; RUN: llvm-spirv %t.bc -opaque-pointers=0 -o %t.spv
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: llvm-spirv -r -emit-opaque-pointers %t.spv -o %t.bc
; RUN: llvm-dis %t.bc -o %t.ll
; RUN: FileCheck %s --input-file %t.spt -check-prefix=CHECK-SPIRV
; RUN: FileCheck %s --input-file %t.ll  -check-prefix=CHECK-LLVM
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

; CHECK-SPIRV: TypeInt [[Int_Ty:[0-9]+]] 8 0
; CHECK-SPIRV: TypeVoid [[Void_Ty:[0-9]+]]
; CHECK-SPIRV: TypeFunction [[Func_Ty1:[0-9]+]] [[Void_Ty]]
; CHECK-SPIRV: TypePointer [[Ptr_Ty:[0-9]+]] 8 [[Int_Ty]]
; CHECK-SPIRV: TypeFunction [[Func_Ty2:[0-9]+]] [[Void_Ty]] [[Ptr_Ty]] [[Ptr_Ty]]

@.str.1 = private unnamed_addr addrspace(1) constant [1 x i8] zeroinitializer, align 1

define linkonce_odr hidden spir_func void @foo() {
entry:
; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[Cast:[0-9]+]]
; CHECK-SPIRV: InBoundsPtrAccessChain {{[0-9]+}} [[Gep:[0-9]+]] [[Cast]]
; CHECK-SPIRV: InBoundsPtrAccessChain {{[0-9]+}} [[Gep1:[0-9]+]] [[Cast]]
; CHECK-SPIRV: ConvertPtrToU {{[0-9]+}} [[PtrToU1:[0-9]+]] [[Gep1]]
; CHECK-SPIRV: ConvertPtrToU {{[0-9]+}} [[PtrToU2:[0-9]+]]
; CHECK-SPIRV: IEqual {{[0-9]+}} [[IEq:[0-9]+]] [[PtrToU1]] [[PtrToU2]]
; CHECK-SPIRV: ConvertUToPtr {{[0-9]+}} [[UToPtr:[0-9]+]]
; CHECK-SPIRV: Select {{[0-9]+}} [[Sel:[0-9]+]] [[IEq]] [[UToPtr]] [[Gep1]]
; CHECK-SPIRV: FunctionCall [[Void_Ty]] {{[0-9]+}} [[Func:[0-9]+]] [[Gep]] [[Sel]]
; CHECK-LLVM:  %[[Cast:[0-9]+]] = addrspacecast ptr addrspace(1) @.str.1 to ptr addrspace(4)
; CHECK-LLVM:  %[[Gep:[0-9]+]] = getelementptr inbounds [1 x i8], ptr addrspace(4) %0, i64 0, i64 0
; CHECK-LLVM:  %[[Gep1:[0-9]+]] = getelementptr inbounds [1 x i8], ptr addrspace(4) %0, i64 0, i64 0
; CHECK-LLVM:  %[[PtrToU1:[0-9]+]] = ptrtoint ptr addrspace(4) %[[Gep1]] to i64
; CHECK-LLVM:  %[[PtrToU2:[0-9]+]] = ptrtoint ptr addrspace(4) null to i64
; CHECK-LLVM:  %[[IEq:[0-9]+]] = icmp eq i64 %[[PtrToU1]], %[[PtrToU2]]
; CHECK-LLVM:  %[[UToPtr:[0-9]+]] = inttoptr i64 -1 to ptr addrspace(4)
; CHECK-LLVM:  %[[Sel:[0-9]+]] = select i1 %[[IEq]], ptr addrspace(4) %[[UToPtr]], ptr addrspace(4) %[[Gep1]]
; CHECK-LLVM:  call spir_func void @bar(ptr addrspace(4) %[[Gep]], ptr addrspace(4) %[[Sel]]) #0
  call spir_func void @bar(i8 addrspace(4)* getelementptr inbounds ([1 x i8], [1 x i8] addrspace(4)* addrspacecast ([1 x i8] addrspace(1)* @.str.1 to [1 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* select (i1 icmp eq (i8 addrspace(4)* getelementptr inbounds ([1 x i8], [1 x i8] addrspace(4)* addrspacecast ([1 x i8] addrspace(1)* @.str.1 to [1 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* null), i8 addrspace(4)* inttoptr (i64 -1 to i8 addrspace(4)*), i8 addrspace(4)* getelementptr inbounds ([1 x i8], [1 x i8] addrspace(4)* addrspacecast ([1 x i8] addrspace(1)* @.str.1 to [1 x i8] addrspace(4)*), i64 0, i64 0)))
  ret void
}

; CHECK-SPIRV: Function [[Void_Ty]] [[Func]] 0 [[Func_Ty2]]

define linkonce_odr hidden spir_func void @bar(i8 addrspace(4)* %__beg, i8 addrspace(4)* %__end) {
entry:
  ret void
}

