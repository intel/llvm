; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_fpga_memory_attributes -spirv-text -o - %t.bc | FileCheck --check-prefix CHECK-SPIRV %s
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_fpga_memory_attributes %t.bc -o %t.spv
; spirv-val %t.spv
; RUN: llvm-spirv -r --spirv-ext=+SPV_INTEL_fpga_memory_attributes %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck --check-prefix CHECK-LLVM %s

target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Capability FPGAMemoryAttributesINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_fpga_memory_attributes"
; CHECK-SPIRV: Decorate [[#REGISTER:]] RegisterINTEL
; CHECK-SPIRV: Decorate [[#MEMORY:]] MemoryINTEL "DEFAULT"
; CHECK-SPIRV: Decorate [[#NUMBANKS:]] NumbanksINTEL 4
; CHECK-SPIRV: Decorate [[#BANKWIDTH:]] BankwidthINTEL 4
; CHECK-SPIRV: Decorate [[#MAX_PRIVATE_COPIES:]] MaxPrivateCopiesINTEL 1
; CHECK-SPIRV: Decorate [[#SINGLEPUMP:]] SinglepumpINTEL
; CHECK-SPIRV: Decorate [[#DOUBLEPUMP:]] DoublepumpINTEL
; CHECK-SPIRV: Decorate [[#MAX_REPLICATES:]] MaxReplicatesINTEL 2
; CHECK-SPIRV: Decorate [[#SIMPLE_DUAL_PORT:]] SimpleDualPortINTEL
; CHECK-SPIRV: Decorate [[#MERGE:]] MergeINTEL "key" "type"
; CHECK-SPIRV: Decorate [[#BANK_BITS:]] BankBitsINTEL 2
; CHECK-SPIRV: Decorate [[#FORCE_POW_2_DEPTH:]] ForcePow2DepthINTEL 2
; CHECK-SPIRV: Decorate [[#STRIDESIZE:]] StridesizeINTEL 4
; CHECK-SPIRV: Decorate [[#WORDSIZE:]] WordsizeINTEL 8
; CHECK-SPIRV: Decorate [[#TRUE_DUAL_PORT:]] TrueDualPortINTEL
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
; CHECK-SPIRV: Variable {{[0-9]+}} [[#BANK_BITS]] {{[0-9]+}}
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
; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[REGISTER]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[MEMORY]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[NUMBANKS]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[BANKWIDTH]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[MAX_PRIVATE_COPIES]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[SINGLEPUMP]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[DOUBLEPUMP]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[MAX_REPLICATES]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[SIMPLE_DUAL_PORT]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[MERGE]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[BANK_BITS]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[FORCE_POW_2_DEPTH]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[STRIDESIZE]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[WORDSIZE]], ptr undef, i32 undef, ptr undef)
; CHECK-LLVM: call void @llvm.var.annotation.p0.p0(ptr %{{[a-zA-Z0-9_]+}}, ptr [[TRUE_DUAL_PORT]], ptr undef, i32 undef, ptr undef)

%"example_type" = type { i32 }

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
  %0 = alloca %"example_type", align 4
  call void @llvm.var.annotation.p0.p1(ptr %0, ptr addrspace(1) @register_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_memory_attr() {
entry:
  %0 = alloca %"example_type", align 4
  call void @llvm.var.annotation.p0.p1(ptr %0, ptr addrspace(1) @memory_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_numbanks_attr() {
entry:
  %0 = alloca %"example_type", align 4
  call void @llvm.var.annotation.p0.p1(ptr %0, ptr addrspace(1) @numbanks_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_bankwidth_attr() {
entry:
  %0 = alloca %"example_type", align 4
  call void @llvm.var.annotation.p0.p1(ptr %0, ptr addrspace(1) @bankwidth_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_max_private_copies_attr() {
entry:
  %0 = alloca %"example_type", align 4
  call void @llvm.var.annotation.p0.p1(ptr %0, ptr addrspace(1) @max_private_copies_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_singlepump_attr() {
entry:
  %0 = alloca %"example_type", align 4
  call void @llvm.var.annotation.p0.p1(ptr %0, ptr addrspace(1) @singlepump_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_doublepump_attr() {
entry:
  %0 = alloca %"example_type", align 4
  call void @llvm.var.annotation.p0.p1(ptr %0, ptr addrspace(1) @doublepump_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_max_replicates_attr() {
entry:
  %0 = alloca %"example_type", align 4
  call void @llvm.var.annotation.p0.p1(ptr %0, ptr addrspace(1) @max_replicates_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_simple_dual_port_attr() {
entry:
  %0 = alloca %"example_type", align 4
  call void @llvm.var.annotation.p0.p1(ptr %0, ptr addrspace(1) @simple_dual_port_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_merge_attr() {
entry:
  %0 = alloca %"example_type", align 4
  call void @llvm.var.annotation.p0.p1(ptr %0, ptr addrspace(1) @merge_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_bankbits_attr() {
entry:
  %0 = alloca %"example_type", align 4
  call void @llvm.var.annotation.p0.p1(ptr %0, ptr addrspace(1) @bankbits_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_force_pow_2_depth_attr() {
entry:
  %0 = alloca %"example_type", align 4
  call void @llvm.var.annotation.p0.p1(ptr %0, ptr addrspace(1) @force_pow_2_depth_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_stride_size_attr() {
entry:
  %0 = alloca %"example_type", align 4
  call void @llvm.var.annotation.p0.p1(ptr %0, ptr addrspace(1) @stride_size_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_word_size_attr() {
entry:
  %0 = alloca %"example_type", align 4
  call void @llvm.var.annotation.p0.p1(ptr %0, ptr addrspace(1) @word_size_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

define spir_func void @test_fpga_true_dual_port_attr() {
entry:
  %0 = alloca %"example_type", align 4
  call void @llvm.var.annotation.p0.p1(ptr %0, ptr addrspace(1) @true_dual_port_attr, ptr addrspace(1) null, i32 0, ptr addrspace(1) null)
  ret void
}

declare void @llvm.var.annotation.p0.p1(ptr, ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1))
