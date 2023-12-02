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
; CHECK-SPIRV: Name [[#MEMORY_FUNC_NAME:]] "test_fpga_memory_attr"
; CHECK-SPIRV: Name [[#NUMBANKS_FUNC_NAME:]] "test_fpga_numbanks_attr"
; CHECK-SPIRV: Name [[#BANKWIDTH_FUNC_NAME:]] "test_fpga_bankwidth_attr"
; CHECK-SPIRV: Name [[#MAX_PRIVATE_COPIES_FUNC_NAME:]] "test_fpga_max_private_copies_attr"
; CHECK-SPIRV: Name [[#SINGLEPUMP_FUNC_NAME:]] "test_fpga_singlepump_attr"
; CHECK-SPIRV: Name [[#DOUBLEPUMP_FUNC_NAME:]] "test_fpga_doublepump_attr"
; CHECK-SPIRV: Name [[#MAX_REPLICATES_FUNC_NAME:]] "test_fpga_max_replicates_attr"
; CHECK-SPIRV: Name [[#SIMPLE_DUAL_PORT_FUNC_NAME:]] "test_fpga_simple_dual_port_attr"
; CHECK-SPIRV: Name [[#MERGE_FUNC_NAME:]] "test_fpga_merge_attr"
; CHECK-SPIRV: Name [[#BANKBITS_FUNC_NAME:]] "test_fpga_bankbits_attr"
; CHECK-SPIRV: Name [[#FORCE_POW_2_DEPTH_FUNC_NAME:]] "test_fpga_force_pow_2_depth_attr"
; CHECK-SPIRV: Name [[#STRIDESIZE_FUNC_NAME:]] "test_fpga_stride_size_attr"
; CHECK-SPIRV: Name [[#WORDSIZE_FUNC_NAME:]] "test_fpga_word_size_attr"
; CHECK-SPIRV: Name [[#TRUE_DUAL_PORT_FUNC_NAME:]] "test_fpga_true_dual_port_attr"

; CHECK-SPIRV: Decorate [[#REGISTER_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#REGISTER:]] RegisterINTEL
; CHECK-SPIRV: Decorate [[#MEMORY_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#MEMORY:]] MemoryINTEL "DEFAULT"
; CHECK-SPIRV: Decorate [[#NUMBANKS_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#NUMBANKS:]] NumbanksINTEL 4
; CHECK-SPIRV: Decorate [[#BANKWIDTH_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#BANKWIDTH:]] BankwidthINTEL 4
; CHECK-SPIRV: Decorate [[#MAX_PRIVATE_COPIES_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#MAX_PRIVATE_COPIES:]] MaxPrivateCopiesINTEL 1
; CHECK-SPIRV: Decorate [[#SINGLEPUMP_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#SINGLEPUMP:]] SinglepumpINTEL
; CHECK-SPIRV: Decorate [[#DOUBLEPUMP_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#DOUBLEPUMP:]] DoublepumpINTEL
; CHECK-SPIRV: Decorate [[#MAX_REPLICATES_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#MAX_REPLICATES:]] MaxReplicatesINTEL 2
; CHECK-SPIRV: Decorate [[#SIMPLE_DUAL_PORT_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#SIMPLE_DUAL_PORT:]] SimpleDualPortINTEL
; CHECK-SPIRV: Decorate [[#MERGE_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#MERGE:]] MergeINTEL "key" "type"
; CHECK-SPIRV: Decorate [[#BANKBITS_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#BANKBITS:]] BankBitsINTEL 2
; CHECK-SPIRV: Decorate [[#FORCE_POW_2_DEPTH_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#FORCE_POW_2_DEPTH:]] ForcePow2DepthINTEL 2
; CHECK-SPIRV: Decorate [[#STRIDESIZE_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#STRIDESIZE:]] StridesizeINTEL 4
; CHECK-SPIRV: Decorate [[#WORDSIZE_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#WORDSIZE:]] WordsizeINTEL 8
; CHECK-SPIRV: Decorate [[#TRUE_DUAL_PORT_FUNC_NAME]] LinkageAttributes
; CHECK-SPIRV: Decorate [[#TRUE_DUAL_PORT:]] TrueDualPortINTEL

; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[#REGISTER]] {{[0-9]+}}
; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[#MEMORY]] {{[0-9]+}}
; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[#NUMBANKS]] {{[0-9]+}}
; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[#BANKWIDTH]] {{[0-9]+}}
; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[#MAX_PRIVATE_COPIES]] {{[0-9]+}}
; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[#SINGLEPUMP]] {{[0-9]+}}
; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[#DOUBLEPUMP]] {{[0-9]+}}
; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[#MAX_REPLICATES]] {{[0-9]+}}
; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[#SIMPLE_DUAL_PORT]] {{[0-9]+}}
; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[#MERGE]] {{[0-9]+}}
; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[#BANKBITS]] {{[0-9]+}}
; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[#FORCE_POW_2_DEPTH]] {{[0-9]+}}
; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[#STRIDESIZE]] {{[0-9]+}}
; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[#WORDSIZE]] {{[0-9]+}}
; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[#TRUE_DUAL_PORT]] {{[0-9]+}}

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

; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr {{.*}}, ptr [[REGISTER]]
; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr {{.*}}, ptr [[MEMORY]]
; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr {{.*}}, ptr [[NUMBANKS]]
; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr {{.*}}, ptr [[BANKWIDTH]]
; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr {{.*}}, ptr [[MAX_PRIVATE_COPIES]]
; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr {{.*}}, ptr [[SINGLEPUMP]]
; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr {{.*}}, ptr [[DOUBLEPUMP]]
; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr {{.*}}, ptr [[MAX_REPLICATES]]
; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr {{.*}}, ptr [[SIMPLE_DUAL_PORT]]
; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr {{.*}}, ptr [[MERGE]]
; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr {{.*}}, ptr [[BANK_BITS]]
; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr {{.*}}, ptr [[FORCE_POW_2_DEPTH]]
; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr {{.*}}, ptr [[STRIDESIZE]]
; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr {{.*}}, ptr [[WORDSIZE]]
; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr {{.*}}, ptr [[TRUE_DUAL_PORT]]

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
