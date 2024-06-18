; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_fpga_cluster_attributes -spirv-text -o - %t.bc | FileCheck --check-prefix CHECK-SPIRV %s
; RUN: llvm-spirv --spirv-ext=+SPV_INTEL_fpga_cluster_attributes %t.bc -o %t.spv
; spirv-val %t.spv
; RUN: llvm-spirv -r --spirv-ext=+SPV_INTEL_fpga_cluster_attributes %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck --check-prefix CHECK-LLVM %s 

target triple = "spir64-unknown-unknown"

; CHECK-SPIRV-DAG: Capability FPGAClusterAttributesINTEL
; CHECK-SPIRV-DAG: Capability FPGAClusterAttributesV2INTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_fpga_cluster_attributes"
; CHECK-SPIRV-DAG: Decorate [[#STALLENABLE_DEC:]] StallEnableINTEL
; CHECK-SPIRV-DAG: Decorate [[#STALLFREE_DEC:]] StallFreeINTEL
; CHECK-SPIRV: Function {{[0-9]+}} [[#STALLENABLE_DEC]] {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV: Function {{[0-9]+}} [[#STALLFREE_DEC]] {{[0-9]+}} {{[0-9]+}}
; CHECK-LLVM: define spir_func void @test_fpga_stallenable_attr() {{.*}} !stall_enable [[STALL_MD:![0-9]+]]
; CHECK-LLVM: define spir_func void @test_fpga_stallfree_attr() {{.*}} !stall_free [[STALL_MD]]
; CHECK-LLVM: [[STALL_MD]] = !{i32 1}


define spir_func void @test_fpga_stallenable_attr() !stall_enable !0 {
entry:
  ret void
}

define spir_func void @test_fpga_stallfree_attr() !stall_free !1 {
entry:
  ret void
}

!0 = !{ i32 1 }
!1 = !{ i32 1 }
