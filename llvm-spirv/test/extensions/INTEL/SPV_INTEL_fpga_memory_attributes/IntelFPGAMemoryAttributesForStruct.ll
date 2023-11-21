; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_fpga_memory_attributes -spirv-text -o - %t.bc | FileCheck --check-prefix CHECK-SPIRV %s
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_fpga_memory_attributes %t.bc -o %t.spv
; spirv-val %t.spv
; RUN: llvm-spirv -r --spirv-ext=+SPV_INTEL_fpga_memory_attributes %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck --check-prefix CHECK-LLVM %s 

target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Capability FPGAMemoryAttributesINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_fpga_memory_attributes"
; CHECK-SPIRV: Name [[#REGISTER_FUNC_NAME:]] "test_fpga_register_attr"
; CHECK-SPIRV: Name [[#REGISTER_TYPE:]] "register_type"
; CHECK-SPIRV: Name [[#MEMORY_FUNC_NAME:]] "test_fpga_memory_attr"
; CHECK-SPIRV: Name [[#MEMORY_TYPE:]] "memory_type"
; CHECK-SPIRV: Name [[#NUMBANKS_FUNC_NAME:]] "test_fpga_numbanks_attr"
; CHECK-SPIRV: Name [[#NUMBANKS_TYPE:]] "numbanks_type"
; CHECK-SPIRV: Name [[#BANKWIDTH_FUNC_NAME:]] "test_fpga_bankwidth_attr"
; CHECK-SPIRV: Name [[#BANKWIDTH_TYPE:]] "bankwidth_type"
; CHECK-SPIRV: Name [[#MAX_PRIVATE_COPIES_FUNC_NAME:]] "test_fpga_max_private_copies_attr"
; CHECK-SPIRV: Name [[#MAX_PRIVATE_COPIES_TYPE:]] "max_private_copies_type"
; CHECK-SPIRV: Name [[#SINGLEPUMP_FUNC_NAME:]] "test_fpga_singlepump_attr"
; CHECK-SPIRV: Name [[#SINGLEPUMP_TYPE:]] "singlepump_type"
; CHECK-SPIRV: Name [[#DOUBLEPUMP_FUNC_NAME:]] "test_fpga_doublepump_attr"
; CHECK-SPIRV: Name [[#DOUBLEPUMP_TYPE:]] "doublepump_type"
; CHECK-SPIRV: Name [[#MAX_REPLICATES_FUNC_NAME:]] "test_fpga_max_replicates_attr"
; CHECK-SPIRV: Name [[#MAX_REPLICATES_TYPE:]] "max_replicates_type"
; CHECK-SPIRV: Name [[#SIMPLE_DUAL_PORT_FUNC_NAME:]] "test_fpga_simple_dual_port_attr"
; CHECK-SPIRV: Name [[#SIMPLE_DUAL_PORT_TYPE:]] "simple_dual_port_type"
; CHECK-SPIRV: Name [[#MERGE_FUNC_NAME:]] "test_fpga_merge_attr"
; CHECK-SPIRV: Name [[#MERGE_TYPE:]] "merge_type"
; CHECK-SPIRV: Name [[#BANKBITS_FUNC_NAME:]] "test_fpga_bankbits_attr"
; CHECK-SPIRV: Name [[#BANKBITS_TYPE:]] "bankbits_type"
; CHECK-SPIRV: Name [[#FORCE_POW_2_DEPTH_FUNC_NAME:]] "test_fpga_force_pow_2_depth_attr"
; CHECK-SPIRV: Name [[#FORCE_POW_2_DEPTH_TYPE:]] "force_pow_2_depth_type"
; CHECK-SPIRV: Name [[#STRIDESIZE_FUNC_NAME:]] "test_fpga_stride_size_attr"
; CHECK-SPIRV: Name [[#STRIDESIZE_TYPE:]] "stride_size_type"
; CHECK-SPIRV: Name [[#WORDSIZE_FUNC_NAME:]] "test_fpga_word_size_attr"
; CHECK-SPIRV: Name [[#WORDSIZE_TYPE:]] "word_size_type"
; CHECK-SPIRV: Name [[#TRUE_DUAL_PORT_FUNC_NAME:]] "test_fpga_true_dual_port_attr"
; CHECK-SPIRV: Name [[#TRUE_DUAL_PORT_TYPE:]] "true_dual_port_type"

; CHECK-SPIRV: Decorate [[#REGISTER_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#REGISTER:]]
; CHECK-SPIRV: MemberDecorate [[#REGISTER_TYPE]] 0 RegisterINTEL
; CHECK-SPIRV: Decorate [[#MEMORY_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#MEMORY:]]
; CHECK-SPIRV: MemberDecorate [[#MEMORY_TYPE]] 0 MemoryINTEL "DEFAULT"
; CHECK-SPIRV: Decorate [[#NUMBANKS_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#NUMBANKS:]]
; CHECK-SPIRV: MemberDecorate [[#NUMBANKS_TYPE]] 0 NumbanksINTEL 4
; CHECK-SPIRV: Decorate [[#BANKWIDTH_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#BANKWIDTH:]]
; CHECK-SPIRV: MemberDecorate [[#BANKWIDTH_TYPE]] 0 BankwidthINTEL 4
; CHECK-SPIRV: Decorate [[#MAX_PRIVATE_COPIES_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#MAX_PRIVATE_COPIES:]]
; CHECK-SPIRV: MemberDecorate [[#MAX_PRIVATE_COPIES_TYPE]] 0 MaxPrivateCopiesINTEL 1
; CHECK-SPIRV: Decorate [[#SINGLEPUMP_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#SINGLEPUMP:]]
; CHECK-SPIRV: MemberDecorate [[#SINGLEPUMP_TYPE]] 0 SinglepumpINTEL
; CHECK-SPIRV: Decorate [[#DOUBLEPUMP_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#DOUBLEPUMP:]]
; CHECK-SPIRV: MemberDecorate [[#DOUBLEPUMP_TYPE]] 0 DoublepumpINTEL
; CHECK-SPIRV: Decorate [[#MAX_REPLICATES_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#MAX_REPLICATES:]]
; CHECK-SPIRV: MemberDecorate [[#MAX_REPLICATES_TYPE]] 0 MaxReplicatesINTEL 2
; CHECK-SPIRV: Decorate [[#SIMPLE_DUAL_PORT_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#SIMPLE_DUAL_PORT:]]
; CHECK-SPIRV: MemberDecorate [[#SIMPLE_DUAL_PORT_TYPE]] 0 SimpleDualPortINTEL
; CHECK-SPIRV: Decorate [[#MERGE_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#MERGE:]]
; CHECK-SPIRV: MemberDecorate [[#MERGE_TYPE]] 0 MergeINTEL "key" "type"
; CHECK-SPIRV: Decorate [[#BANKBITS_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#BANKBITS:]]
; CHECK-SPIRV: MemberDecorate [[#BANKBITS_TYPE]] 0 BankBitsINTEL 2
; CHECK-SPIRV: Decorate [[#FORCE_POW_2_DEPTH_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#FORCE_POW_2_DEPTH:]]
; CHECK-SPIRV: MemberDecorate [[#FORCE_POW_2_DEPTH_TYPE]] 0 ForcePow2DepthINTEL 2
; CHECK-SPIRV: Decorate [[#STRIDESIZE_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#STRIDESIZE:]]
; CHECK-SPIRV: MemberDecorate [[#STRIDESIZE_TYPE]] 0 StridesizeINTEL 4
; CHECK-SPIRV: Decorate [[#WORDSIZE_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#WORDSIZE:]]
; CHECK-SPIRV: MemberDecorate [[#WORDSIZE_TYPE]] 0 WordsizeINTEL 8
; CHECK-SPIRV: Decorate [[#TRUE_DUAL_PORT_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#TRUE_DUAL_PORT:]]
; CHECK-SPIRV: MemberDecorate [[#TRUE_DUAL_PORT_TYPE]] 0 TrueDualPortINTEL
; CHECK-SPIRV: Variable {{[0-9]+}} [[#REGISTER]] {{[0-9]+}}
; CHECK-SPIRV: Variable {{[0-9]+}} [[#MEMORY]] {{[0-9]+}}
; CHECK-SPIRV: Variable {{[0-9]+}} [[#NUMBANKS]] {{[0-9]+}}
; CHECK-SPIRV: Variable {{[0-9]+}} [[#BANKWIDTH]] {{[0-9]+}}
; CHECK-SPIRV: Variable {{[0-9]+}} [[#MAX_PRIVATE_COPIES]] {{[0-9]+}}
; CHECK-SPIRV: Variable {{[0-9]+}} [[#SINGLEPUMP]] {{[0-9]+}}
; CHECK-SPIRV: Variable {{[0-9]+}} [[#DOUBLEPUMP]] {{[0-9]+}}
; CHECK-SPIRV: Variable {{[0-9]+}} [[#MAX_REPLICATES]] {{[0-9]+}}
; CHECK-SPIRV: Variable {{[0-9]+}} [[#SIMPLE_DUAL_PORT]] {{[0-9]+}}
; CHECK-SPIRV: Variable {{[0-9]+}} [[#MERGE]] {{[0-9]+}}
; CHECK-SPIRV: Variable {{[0-9]+}} [[#BANKBITS]] {{[0-9]+}}
; CHECK-SPIRV: Variable {{[0-9]+}} [[#FORCE_POW_2_DEPTH]] {{[0-9]+}}
; CHECK-SPIRV: Variable {{[0-9]+}} [[#STRIDESIZE]] {{[0-9]+}}
; CHECK-SPIRV: Variable {{[0-9]+}} [[#WORDSIZE]] {{[0-9]+}}
; CHECK-SPIRV: Variable {{[0-9]+}} [[#TRUE_DUAL_PORT]] {{[0-9]+}}

; CHECK-LLVM: [[REGISTER:@[0-9_.]+]] = {{.*}}{register:1}
; CHECK-LLVM: [[MEMORY:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}
; CHECK-LLVM: [[NUMBANKS:@[0-9_.]+]] = {{.*}}{numbanks:4}
; CHECK-LLVM: [[BANKWIDTH:@[0-9_.]+]] = {{.*}}{bankwidth:4}
; CHECK-LLVM: [[MAX_PRIVATE_COPIES:@[0-9_.]+]] = {{.*}}{private_copies:1}
; CHECK-LLVM: [[SINGLEPUMP:@[0-9_.]+]] = {{.*}}{pump:1}
; CHECK-LLVM: [[DOUBLEPUMP:@[0-9_.]+]] = {{.*}}{pump:2}
; CHECK-LLVM: [[MAX_REPLICATES:@[0-9_.]+]] = {{.*}}{max_replicates:2}
; CHECK-LLVM: [[SIMPLE_DUAL_PORT:@[0-9_.]+]] = {{.*}}{simple_dual_port:1}
; CHECK-LLVM: [[MERGE:@[0-9_.]+]] = {{.*}}{merge:key:type}
; CHECK-LLVM: [[BANK_BITS:@[0-9_.]+]] = {{.*}}{bank_bits:2}
; CHECK-LLVM: [[FORCE_POW_2_DEPTH:@[0-9_.]+]] = {{.*}}{force_pow2_depth:2}
; CHECK-LLVM: [[STRIDESIZE:@[0-9_.]+]] = {{.*}}{stride_size:4}
; CHECK-LLVM: [[WORDSIZE:@[0-9_.]+]] = {{.*}}{word_size:8}
; CHECK-LLVM: [[TRUE_DUAL_PORT:@[0-9_.]+]] = {{.*}}{true_dual_port}
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[REGISTER]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[MEMORY]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[NUMBANKS]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[BANKWIDTH]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[MAX_PRIVATE_COPIES]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[SINGLEPUMP]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[DOUBLEPUMP]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[MAX_REPLICATES]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[SIMPLE_DUAL_PORT]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[MERGE]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[BANK_BITS]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[FORCE_POW_2_DEPTH]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[STRIDESIZE]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[WORDSIZE]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call ptr @llvm.ptr.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[TRUE_DUAL_PORT]], ptr undef, i32 undef, ptr undef)

%"register_type" = type { i32 }
%"memory_type" = type { i32 }
%"numbanks_type" = type { i32 }
%"bankwidth_type" = type { i32 }
%"max_private_copies_type" = type { i32 }
%"singlepump_type" = type { i32 }
%"doublepump_type" = type { i32 }
%"max_replicates_type" = type { i32 }
%"simple_dual_port_type" = type { i32 }
%"merge_type" = type { i32 }
%"bankbits_type" = type { i32 }
%"force_pow_2_depth_type" = type { i32 }
%"stride_size_type" = type { i32 }
%"word_size_type" = type { i32 }
%"true_dual_port_type" = type { i32 }

@register_attr = private unnamed_addr addrspace(1) constant [7 x i8] c"{5825}\00", section "llvm.metadata"
@memory_attr = private unnamed_addr addrspace(1) constant [15 x i8] c"{5826:DEFAULT}\00", section "llvm.metadata"
@numbanks_attr = private unnamed_addr addrspace(1) constant [9 x i8] c"{5827:4}\00", section "llvm.metadata"
@bankwidth_attr = private unnamed_addr addrspace(1) constant [9 x i8] c"{5828:4}\00", section "llvm.metadata"
@max_private_copies_attr = private unnamed_addr addrspace(1) constant [9 x i8] c"{5829:1}\00", section "llvm.metadata"
@singlepump_attr = private unnamed_addr addrspace(1) constant [7 x i8] c"{5830}\00", section "llvm.metadata"
@doublepump_attr = private unnamed_addr addrspace(1) constant [7 x i8] c"{5831}\00", section "llvm.metadata"
@max_replicates_attr = private unnamed_addr addrspace(1) constant [9 x i8] c"{5832:2}\00", section "llvm.metadata"
@simple_dual_port_attr = private unnamed_addr addrspace(1) constant [7 x i8] c"{5833}\00", section "llvm.metadata"
@merge_attr = private unnamed_addr addrspace(1) constant [16 x i8] c"{5834:key,type}\00", section "llvm.metadata"
@bankbits_attr = private unnamed_addr addrspace(1) constant [9 x i8] c"{5835:2}\00", section "llvm.metadata"
@force_pow_2_depth_attr = private unnamed_addr addrspace(1) constant [9 x i8] c"{5836:2}\00", section "llvm.metadata"
@stride_size_attr = private unnamed_addr addrspace(1) constant [9 x i8] c"{5883:4}\00", section "llvm.metadata"
@word_size_attr = private unnamed_addr addrspace(1) constant [9 x i8] c"{5884:8}\00", section "llvm.metadata"
@true_dual_port_attr = private unnamed_addr addrspace(1) constant [7 x i8] c"{5885}\00", section "llvm.metadata"

define spir_func void @test_fpga_register_attr() {
entry:
  %0 = alloca %"register_type", align 4
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %1, ptr addrspace(1) @register_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_memory_attr() {
entry:
  %0 = alloca %"memory_type", align 4
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %1, ptr addrspace(1) @memory_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_numbanks_attr() {
entry:
  %0 = alloca %"numbanks_type", align 4
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %1, ptr addrspace(1) @numbanks_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_bankwidth_attr() {
entry:
  %0 = alloca %"bankwidth_type", align 4
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %1, ptr addrspace(1) @bankwidth_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_max_private_copies_attr() {
entry:
  %0 = alloca %"max_private_copies_type", align 4
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %1, ptr addrspace(1) @max_private_copies_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_singlepump_attr() {
entry:
  %0 = alloca %"singlepump_type", align 4
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %1, ptr addrspace(1) @singlepump_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_doublepump_attr() {
entry:
  %0 = alloca %"doublepump_type", align 4
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %1, ptr addrspace(1) @doublepump_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_max_replicates_attr() {
entry:
  %0 = alloca %"max_replicates_type", align 4
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %1, ptr addrspace(1) @max_replicates_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_simple_dual_port_attr() {
entry:
  %0 = alloca %"simple_dual_port_type", align 4
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %1, ptr addrspace(1) @simple_dual_port_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}
define spir_func void @test_fpga_merge_attr() {
entry:
  %0 = alloca %"merge_type", align 4
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %1, ptr addrspace(1) @merge_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_bankbits_attr() {
entry:
  %0 = alloca %"bankbits_type", align 4
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %1, ptr addrspace(1) @bankbits_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_force_pow_2_depth_attr() {
entry:
  %0 = alloca %"force_pow_2_depth_type", align 4
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %1, ptr addrspace(1) @force_pow_2_depth_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_stride_size_attr() {
entry:
  %0 = alloca %"stride_size_type", align 4
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %1, ptr addrspace(1) @stride_size_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_word_size_attr() {
entry:
  %0 = alloca %"word_size_type", align 4
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %1, ptr addrspace(1) @word_size_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_true_dual_port_attr() {
entry:
  %0 = alloca %"true_dual_port_type", align 4
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %1, ptr addrspace(1) @true_dual_port_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

declare ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1))
