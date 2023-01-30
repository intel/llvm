; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_memory_attributes -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_memory_attributes -o %t.spv
; RUN: llvm-spirv -r -emit-opaque-pointers %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.S = type { i32* }

@.str = private unnamed_addr constant [114 x i8] c"{5825}{5826:\22testmem\22}{5828:42}{5827:\2241\22}{5829:3}{5831}{5830}{5832:2}{5834:str1,\22str2\22}{5835:\221\22,3,\222\22}{5836:24}\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [14 x i8] c"<invalid loc>\00", section "llvm.metadata"

define dso_local noundef i32 @main() {
  %1 = alloca i32, align 4
  %2 = alloca %struct.S, align 8
  store i32 0, i32* %1, align 4
  %3 = getelementptr inbounds %struct.S, %struct.S* %2, i32 0, i32 0
  %4 = bitcast i32** %3 to i8*
  %5 = call i8* @llvm.ptr.annotation.p0i8(i8* %4, i8* getelementptr inbounds ([114 x i8], [114 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.1, i32 0, i32 0), i32 3, i8* null)
  %6 = bitcast i8* %5 to i32**
  %7 = load i32*, i32** %6, align 8
  ret i32 0
}

declare i8* @llvm.ptr.annotation.p0i8(i8*, i8*, i8*, i32, i8*)

; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 0 RegisterINTEL
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 0 MemoryINTEL "testmem"
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 0 BankwidthINTEL 42
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 0 NumbanksINTEL 41
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 0 MaxPrivateCopiesINTEL 3
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 0 DoublepumpINTEL
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 0 SinglepumpINTEL
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 0 MaxReplicatesINTEL 2
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 0 MergeINTEL "str1" "str2"
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 0 BankBitsINTEL 1 3 2
; CHECK-SPIRV-DAG: MemberDecorate {{[0-9]+}} 0 ForcePow2DepthINTEL 24

; CHECK-LLVM: @{{.*}} = private unnamed_addr constant [163 x i8] c"
; CHECK-LLVM-DAG: {register:1}
; CHECK-LLVM-DAG: {memory:testmem}
; CHECK-LLVM-DAG: {bankwidth:42}
; CHECK-LLVM-DAG: {numbanks:41}
; CHECK-LLVM-DAG: {private_copies:3}
; CHECK-LLVM-DAG: {pump:2}
; CHECK-LLVM-DAG: {pump:1}
; CHECK-LLVM-DAG: {max_replicates:2}
; CHECK-LLVM-DAG: {merge:str1:str2}
; CHECK-LLVM-DAG: {bank_bits:1,3,2}
; CHECK-LLVM-DAG: {force_pow2_depth:24}
