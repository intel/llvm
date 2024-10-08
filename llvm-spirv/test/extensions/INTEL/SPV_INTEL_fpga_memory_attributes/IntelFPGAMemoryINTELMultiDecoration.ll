; Test checks that MemoryINTEL decoration can be applied twice on a single 
; variable

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_memory_attributes -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability FPGAMemoryAttributesINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_fpga_memory_attributes"
; CHECK-SPIRV: Decorate [[#empty:]] MemoryINTEL "DEFAULT" 
; CHECK-SPIRV-NOT: Decorate [[#]] MemoryINTEL "DEFAULT" 
; CHECK-SPIRV: Decorate [[#mlab:]] MemoryINTEL "MLAB"
; CHECK-SPIRV-NOT: Decorate [[#]] MemoryINTEL "DEFAULT" 
; CHECK-SPIRV: Decorate [[#block_ram:]] MemoryINTEL "BLOCK_RAM"


; ModuleID = '/nfs/site/disks/swuser_work_aradzikh/external/llvm-intel/sycl/test/check_device_code/fpga_mem_local.cpp'
source_filename = "fpga_mem_local.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"fpga_mem" = type { [10 x i32] }

@.str.1 = private unnamed_addr addrspace(1) constant [17 x i8] c"{5826:\22DEFAULT\22}\00", section "llvm.metadata"
@.str.2 = private unnamed_addr addrspace(1) constant [30 x i8] c"{5826:\22DEFAULT\22}{5826:\22MLAB\22}\00", section "llvm.metadata"
@.str.3 = private unnamed_addr addrspace(1) constant [35 x i8] c"{5826:\22DEFAULT\22}{5826:\22BLOCK_RAM\22}\00", section "llvm.metadata"

; CHECK-LLVM: [[empty_annot:]] = private unnamed_addr constant [17 x i8] c"{memory:DEFAULT}\00", align 1
; CHECK-LLVM: [[mlab_annot:]] = private unnamed_addr constant [14 x i8] c"{memory:MLAB}\00", align 1
; CHECK-LLVM: [[block_ram_annot:]] = private unnamed_addr constant [19 x i8] c"{memory:BLOCK_RAM}\00", align 1

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @kernel(ptr addrspace(4) %out) {
entry:
  %empty.i = alloca %"fpga_mem", align 4
  %mlab.i = alloca %"fpga_mem", align 4
  %block_ram.i = alloca %"fpga_mem", align 4
  %empty.ascast.i = addrspacecast ptr %empty.i to ptr addrspace(4)
  %mlab.ascast.i = addrspacecast ptr %mlab.i to ptr addrspace(4)
  %block_ram.ascast.i = addrspacecast ptr %block_ram.i to ptr addrspace(4)

  call void @llvm.var.annotation.p4.p1(ptr addrspace(4) %empty.ascast.i, ptr addrspace(1) @.str.1, ptr addrspace(1) undef, i32 undef, ptr addrspace(1) undef)
  ; CHECK-SPV-IR: call void @llvm.var.annotation{{.*}}(ptr addrspace(4) %empty{{.*}}, ptr [[empty_annot]]
  ; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr addrspace(4) %empty{{.*}}, ptr [[empty_annot]]
  call void @llvm.var.annotation.p4.p1(ptr addrspace(4) %mlab.ascast.i, ptr addrspace(1) @.str.2, ptr addrspace(1) undef, i32 undef, ptr addrspace(1) undef)
  ; CHECK-SPV-IR: call void @llvm.var.annotation{{.*}}(ptr addrspace(4) %mlab{{.*}}, ptr [[mlab_annot]]
  ; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr addrspace(4) %mlab{{.*}}, ptr [[mlab_annot]]
  call void @llvm.var.annotation.p4.p1(ptr addrspace(4) %block_ram.ascast.i, ptr addrspace(1) @.str.3, ptr addrspace(1) undef, i32 undef, ptr addrspace(1) undef)
  ; CHECK-SPV-IR: call void @llvm.var.annotation{{.*}}(ptr addrspace(4) %block_ram{{.*}}, ptr [[block_ram_annot]]
  ; CHECK-LLVM: call void @llvm.var.annotation{{.*}}(ptr addrspace(4) %block_ram{{.*}}, ptr [[block_ram_annot]]

  %l1 = load i32, ptr addrspace(4) %empty.ascast.i, align 4
  %l2 = load i32, ptr addrspace(4) %mlab.ascast.i, align 4
  %l3 = load i32, ptr addrspace(4) %block_ram.ascast.i, align 4

  %add1 = add nsw i32 %l1, %l2
  %add2 = add nsw i32 %add1, %l3
  store i32 %add2, ptr addrspace(4) %out, align 4

  ret void
}

declare void @llvm.var.annotation.p4.p1(ptr addrspace(4), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1))
