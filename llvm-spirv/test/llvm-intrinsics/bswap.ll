; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-text %t.bc -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -to-binary %t.spt -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM "--implicit-check-not={{(shl|lshr|or|and) i(16|32|64)}}"

; CHECK-SPIRV: Name [[#FuncNameInt16:]] "spirv.llvm_bswap_i16"
; CHECK-SPIRV: Name [[#FuncNameInt32:]] "spirv.llvm_bswap_i32"
; CHECK-SPIRV: Name [[#FuncNameInt64:]] "spirv.llvm_bswap_i64"

; CHECK-SPIRV: TypeInt [[#TypeInt32:]] 32 0
; CHECK-SPIRV: TypeInt [[#TypeInt16:]] 16 0
; CHECK-SPIRV: TypeInt [[#TypeInt64:]] 64 0

; CHECK-SPIRV: Function [[#TypeInt16]] [[#FuncNameInt16]]
; CHECK-SPIRV: FunctionParameter [[#TypeInt16]] [[#FuncParameter:]]
; CHECK-SPIRV: ShiftLeftLogical [[#TypeInt16]] [[#]] [[#FuncParameter]]
; CHECK-SPIRV: ShiftRightLogical [[#TypeInt16]] [[#]] [[#FuncParameter]]
; CHECK-SPIRV: BitwiseOr [[#TypeInt16]] [[#RetVal:]]
; CHECK-SPIRV: ReturnValue [[#RetVal]]
; CHECK-SPIRV: FunctionEnd 

; CHECK-SPIRV: Function [[#TypeInt32]] [[#FuncNameInt32]]
; CHECK-SPIRV: FunctionParameter [[#TypeInt32]] [[#FuncParameter:]]
; CHECK-SPIRV-COUNT-2: ShiftLeftLogical [[#TypeInt32]] [[#]] [[#FuncParameter]]
; CHECK-SPIRV-COUNT-2: ShiftRightLogical [[#TypeInt32]] [[#]] [[#FuncParameter]]
; CHECK-SPIRV-COUNT-2: BitwiseAnd [[#TypeInt32]]
; CHECK-SPIRV-COUNT-2: BitwiseOr [[#TypeInt32]]
; CHECK-SPIRV: BitwiseOr [[#TypeInt32]] [[#RetVal:]]
; CHECK-SPIRV: ReturnValue [[#RetVal:]]
; CHECK-SPIRV: FunctionEnd

; CHECK-SPIRV: Function [[#TypeInt64]] [[#FuncNameInt64]] 
; CHECK-SPIRV: FunctionParameter [[#TypeInt64]] [[#FuncParameter:]] 
; CHECK-SPIRV-COUNT-4: ShiftLeftLogical [[#TypeInt64]] [[#]] [[#FuncParameter]] [[#]]
; CHECK-SPIRV-COUNT-4: ShiftRightLogical [[#TypeInt64]] [[#]] [[#FuncParameter]] [[#]]
; CHECK-SPIRV-COUNT-6: BitwiseAnd [[#TypeInt64]]
; CHECK-SPIRV-COUNT-6: BitwiseOr [[#TypeInt64]]
; CHECK-SPIRV: BitwiseOr [[#TypeInt64]] [[#RetVal:]]
; CHECK-SPIRV: ReturnValue [[#RetVal]]
; CHECK-SPIRV: FunctionEnd 

; CHECK-LLVM-COUNT-2: call i16 @llvm.bswap.i16(i16
; CHECK-LLVM: call i32 @llvm.bswap.i32(i32
; CHECK-LLVM: call i64 @llvm.bswap.i64(i64

; ModuleID = 'source.cpp'
source_filename = "source.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

; Function Attrs: mustprogress noinline norecurse nounwind optnone
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %a = alloca i16, align 2
  %b = alloca i16, align 2
  %h = alloca i16, align 2
  %i = alloca i16, align 2
  %c = alloca i32, align 4
  %d = alloca i32, align 4
  %e = alloca i64, align 8
  %f = alloca i64, align 8
  store i32 0, i32* %retval, align 4
  store i16 258, i16* %a, align 2
  %0 = load i16, i16* %a, align 2
  %1 = call i16 @llvm.bswap.i16(i16 %0)
  store i16 %1, i16* %b, align 2
  store i16 234, i16* %h, align 2
  %2 = load i16, i16* %h, align 2
  %3 = call i16 @llvm.bswap.i16(i16 %2)
  store i16 %3, i16* %i, align 2
  store i32 566, i32* %c, align 4
  %4 = load i32, i32* %c, align 4
  %5 = call i32 @llvm.bswap.i32(i32 %4)
  store i32 %5, i32* %d, align 4
  store i64 12587, i64* %e, align 8
  %6 = load i64, i64* %e, align 8
  %7 = call i64 @llvm.bswap.i64(i64 %6)
  store i64 %7, i64* %f, align 8
  ret i32 0
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare i16 @llvm.bswap.i16(i16) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.bswap.i32(i32) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare i64 @llvm.bswap.i64(i64) #1

attributes #0 = { mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0, !1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{}
!3 = !{!"Compiler"}
