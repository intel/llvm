; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_memory_attributes -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_memory_attributes -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.S = type { ptr }

@.str = private unnamed_addr addrspace(1) constant [114 x i8] c"{5825}{5826:\22testmem\22}{5828:42}{5827:\2241\22}{5829:3}{5831}{5830}{5832:2}{5834:str1,\22str2\22}{5835:\221\22,3,\222\22}{5836:24}\00", section "llvm.metadata"
@.str.1 = private unnamed_addr addrspace(1) constant [14 x i8] c"<invalid loc>\00", section "llvm.metadata"

define dso_local noundef i32 @main() {
  %1 = alloca i32, align 4
  %2 = alloca %struct.S, align 8
  store i32 0, ptr %1, align 4
  %3 = call ptr @llvm.ptr.annotation.p0.p1(ptr %2, ptr addrspace(1) @.str, ptr addrspace(1) @.str.1, i32 3, ptr addrspace(1) null)
  %4 = load ptr, ptr %3, align 8
  ret i32 0
}

declare ptr @llvm.ptr.annotation.p0.p1(ptr, ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1))

; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} RegisterINTEL
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} MemoryINTEL "testmem"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} BankwidthINTEL 42
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} NumbanksINTEL 41
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} MaxPrivateCopiesINTEL 3
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} DoublepumpINTEL
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} SinglepumpINTEL
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} MaxReplicatesINTEL 2
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} MergeINTEL "str1" "str2"
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} BankBitsINTEL 1 3 2
; CHECK-SPIRV-DAG: Decorate {{[0-9]+}} ForcePow2DepthINTEL 24

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
