; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: spirv-val %t.spv
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Decorate [[#NonConstMemset:]] LinkageAttributes "spirv.llvm_memset_p3i8_i32"
; CHECK-SPIRV: TypeInt [[Int8:[0-9]+]] 8 0
; CHECK-SPIRV: Constant {{[0-9]+}} [[Lenmemset21:[0-9]+]] 4
; CHECK-SPIRV: Constant {{[0-9]+}} [[Lenmemset0:[0-9]+]] 12
; CHECK-SPIRV: Constant {{[0-9]+}} [[Const21:[0-9]+]] 21
; CHECK-SPIRV: TypeArray [[Int8x4:[0-9]+]] [[Int8]] [[Lenmemset21]]
; CHECK-SPIRV: TypePointer [[Int8Ptr:[0-9]+]] 8 [[Int8]]
; CHECK-SPIRV: TypeArray [[Int8x12:[0-9]+]] [[Int8]] [[Lenmemset0]]
; CHECK-SPIRV: TypePointer [[Int8PtrConst:[0-9]+]] 0 [[Int8]]

; CHECK-SPIRV: ConstantNull [[Int8x12]] [[Init:[0-9]+]]
; CHECK-SPIRV: Variable {{[0-9]+}} [[Val:[0-9]+]] 0 [[Init]]
; CHECK-SPIRV: 7 ConstantComposite [[Int8x4]] [[InitComp:[0-9]+]] [[Const21]] [[Const21]] [[Const21]] [[Const21]]
; CHECK-SPIRV: Variable {{[0-9]+}} [[ValComp:[0-9]+]] 0 [[InitComp]]
; CHECK-SPIRV: ConstantFalse [[#]] [[#False:]]

; CHECK-SPIRV: Bitcast [[Int8Ptr]] [[Target:[0-9]+]] {{[0-9]+}}
; CHECK-SPIRV: Bitcast [[Int8PtrConst]] [[Source:[0-9]+]] [[Val]]
; CHECK-SPIRV: CopyMemorySized [[Target]] [[Source]] [[Lenmemset0]] 2 4

; CHECK-SPIRV: Bitcast [[Int8PtrConst]] [[SourceComp:[0-9]+]] [[ValComp]]
; CHECK-SPIRV: CopyMemorySized {{[0-9]+}} [[SourceComp]] [[Lenmemset21]] 2 4

; CHECK-SPIRV: FunctionCall [[#]] [[#]] [[#NonConstMemset]] [[#]] [[#]] [[#]] [[#False]]

; CHECK-SPIRV: Function [[#]] [[#NonConstMemset]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#Dest:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#Value:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#Len:]]
; CHECK-SPIRV: FunctionParameter [[#]] [[#Volatile:]]

; CHECK-SPIRV: Label [[#Entry:]]
; CHECK-SPIRV: IEqual [[#]] [[#IsZeroLen:]] [[#Zero:]] [[#Len]]
; CHECK-SPIRV: BranchConditional [[#IsZeroLen]] [[#End:]] [[#WhileBody:]]

; CHECK-SPIRV: Label [[#WhileBody]]
; CHECK-SPIRV: Phi [[#]] [[#Offset:]] [[#Zero]] [[#Entry]] [[#OffsetInc:]] [[#WhileBody]]
; CHECK-SPIRV: InBoundsPtrAccessChain [[#]] [[#Ptr:]] [[#Dest]] [[#Offset]]
; CHECK-SPIRV: Store [[#Ptr]] [[#Value]] 2 1
; CHECK-SPIRV: IAdd [[#]] [[#OffsetInc]] [[#Offset]] [[#One:]]
; CHECK-SPIRV: ULessThan [[#]] [[#NotEnd:]] [[#OffsetInc]] [[#Len]]
; CHECK-SPIRV: BranchConditional [[#NotEnd]] [[#WhileBody]] [[#End]]

; CHECK-SPIRV: Label [[#End]]
; CHECK-SPIRV: Return

; CHECK-SPIRV: FunctionEnd

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir"

%struct.S1 = type { i32, i32, i32 }

; CHECK-LLVM: internal unnamed_addr addrspace(2) constant [12 x i8] zeroinitializer
; CHECK-LLVM: internal unnamed_addr addrspace(2) constant [4 x i8] c"\15\15\15\15"

; Function Attrs: nounwind
define spir_func void @_Z5foo11v(%struct.S1 addrspace(4)* noalias nocapture sret(%struct.S1 addrspace(4)*) %agg.result, i32 %s1, i64 %s2, i8 %v) #0 {
  %x = alloca [4 x i8]
  %x.bc = bitcast [4 x i8]* %x to i8*
  %1 = bitcast %struct.S1 addrspace(4)* %agg.result to i8 addrspace(4)*
  tail call void @llvm.memset.p4i8.i32(i8 addrspace(4)* align 4 %1, i8 0, i32 12, i1 false)
; CHECK-LLVM: call void @llvm.memset.p4i8.i32(i8 addrspace(4)* align 4 %1, i8 0, i32 12, i1 false)
  tail call void @llvm.memset.p0i8.i32(i8* align 4 %x.bc, i8 21, i32 4, i1 false)
; CHECK-LLVM: call void @llvm.memcpy.p0i8.p2i8.i32(i8* align 4 %x.bc, i8 addrspace(2)* align 4 %3, i32 4, i1 false)

  ; non-const value
  tail call void @llvm.memset.p0i8.i32(i8* align 4 %x.bc, i8 %v, i32 3, i1 false)
; CHECK-LLVM: call void @llvm.memset.p0i8.i32(i8* %x.bc, i8 %v, i32 3, i1 false)

  ; non-const value and size
  tail call void @llvm.memset.p0i8.i32(i8*  align 4 %x.bc, i8 %v, i32 %s1, i1 false)
; CHECK-LLVM: call void @llvm.memset.p0i8.i32(i8* %x.bc, i8 %v, i32 %s1, i1 false)

  ; Address spaces, non-const value and size
  %a = addrspacecast i8 addrspace(4)* %1 to i8 addrspace(3)*
  tail call void @llvm.memset.p3i8.i32(i8 addrspace(3)* align 4 %a, i8 %v, i32 %s1, i1 false)
; CHECK-LLVM: call void @llvm.memset.p3i8.i32(i8 addrspace(3)* %a, i8 %v, i32 %s1, i1 false)
  %b = addrspacecast i8 addrspace(4)* %1 to i8 addrspace(1)*
  tail call void @llvm.memset.p1i8.i64(i8 addrspace(1)* align 4 %b, i8 %v, i64 %s2, i1 false)
; CHECK-LLVM: call void @llvm.memset.p1i8.i64(i8 addrspace(1)* %b, i8 %v, i64 %s2, i1 false)

  ; Volatile
  tail call void @llvm.memset.p1i8.i64(i8 addrspace(1)* align 4 %b, i8 %v, i64 %s2, i1 true)
; CHECK-LLVM: call void @llvm.memset.p1i8.i64(i8 addrspace(1)* %b, i8 %v, i64 %s2, i1 true)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memset.p4i8.i32(i8 addrspace(4)* nocapture, i8, i32, i1) #1

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1) #1

; Function Attrs: nounwind
declare void @llvm.memset.p3i8.i32(i8 addrspace(3)*, i8, i32, i1) #1

; Function Attrs: nounwind
declare void @llvm.memset.p1i8.i64(i8 addrspace(1)*, i8, i64, i1) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!3}
!spirv.Source = !{!4}
!spirv.String = !{!5}

!0 = !{i32 1, i32 2}
!1 = !{i32 2, i32 2}
!2 = !{}
!3 = !{!"clang version 3.6.1 "}
!4 = !{i32 4, i32 202000, !5}
!5 = !{!"llvm.memset.cl"}
