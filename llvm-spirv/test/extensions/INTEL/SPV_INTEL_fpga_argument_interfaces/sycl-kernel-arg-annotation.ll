; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-ext=-all,+SPV_INTEL_fpga_argument_interfaces,+SPV_INTEL_fpga_buffer_location -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$_ZTS4MyIP = comdat any

; Function Attrs: convergent mustprogress norecurse
define weak_odr dso_local spir_kernel void @_ZTS4MyIP(ptr addrspace(4) noundef %_arg_p) #0 comdat !kernel_arg_buffer_location !1587 !spirv.ParameterDecorations !1588
; CHECK-LLVM-DAG:  !spirv.ParameterDecorations ![[PARMDECOR:[0-9]+]]
{
entry:
		ret void
}

!1587 = !{i32 -1}
!1588 = !{!1589}
!1589 = !{!1590, !1591, !1592, !1593, !1594, !1595, !1596, !1597, !1598, !1599, !1600, !1601}
!1590 = !{i32 44, i32 4}
!1591 = !{i32 6177, i32 32}
!1592 = !{i32 5921, i32 10}
!1593 = !{i32 6175, i32 1}
!1594 = !{i32 6178, i32 64}
!1595 = !{i32 6179, i32 1}
!1596 = !{i32 6181, i32 3}
!1597 = !{i32 6180, i32 2}
!1598 = !{i32 6176, i32 1}
!1599 = !{i32 6183, i32 1}
!1600 = !{i32 19, i32 1}
!1601 = !{i32 6182, i32 5}

; CHECK-LLVM-DAG: ![[PARMDECOR]] = !{![[ARG:[0-9]+]]}
; CHECK-LLVM-DAG: ![[ARG]] = !{![[ALIGN:[0-9]+]], ![[AWIDTH:[0-9]+]], ![[BL:[0-9]+]], ![[CONDUIT:[0-9]+]], ![[DWIDTH:[0-9]+]], ![[LATENCY:[0-9]+]], ![[MAXBURST:[0-9]+]], ![[RWMODE:[0-9]+]], ![[REGMAP:[0-9]+]], ![[STABLE:[0-9]+]], ![[STRICT:[0-9]+]], ![[WAITREQ:[0-9]+]]}

; CHECK: ![[ALIGN]]   = !{i32 44, i32 4}
; CHECK: ![[AWIDTH]]  = !{i32 6177, i32 32}
; CHECK: ![[BL]]      = !{i32 5921, i32 10}
; CHECK: ![[CONDUIT]] = !{i32 6175, i32 1}
; CHECK: ![[DWIDTH]]  = !{i32 6178, i32 64}
; CHECK: ![[LATENCY]] = !{i32 6179, i32 1}
; CHECK: ![[MAXBURST]] = !{i32 6181, i32 3}
; CHECK: ![[RWMODE]]  = !{i32 6180, i32 2}
; CHECK: ![[REGMAP]]  = !{i32 6176, i32 1}
; CHECK: ![[STABLE]]  = !{i32 6183, i32 1}
; CHECK: ![[STRICT]]  = !{i32 19, i32 1}
; CHECK: ![[WAITREQ]] = !{i32 6182, i32 5}

; CHECK-SPIRV: Capability FPGABufferLocationINTEL
; CHECK-SPIRV: Capability FPGAArgumentInterfacesINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_fpga_argument_interfaces"
; CHECK-SPIRV: Extension "SPV_INTEL_fpga_buffer_location"
; CHECK-SPIRV-DAG:  Name [[IDS:[0-9]+]] "_arg_p"
; CHECK-SPIRV-DAG:  Name [[ID:[0-9]+]] "_arg_p"
; CHECK-SPIRV:  Decorate [[ID]] Alignment 4
; CHECK-SPIRV:  Decorate [[ID]] MMHostInterfaceAddressWidthINTEL 32
; CHECK-SPIRV:  Decorate [[ID]] BufferLocationINTEL 10
; CHECK-SPIRV:  Decorate [[ID]] ConduitKernelArgumentINTEL
; CHECK-SPIRV:  Decorate [[ID]] MMHostInterfaceDataWidthINTEL 64
; CHECK-SPIRV:  Decorate [[ID]] MMHostInterfaceLatencyINTEL 1
; CHECK-SPIRV:  Decorate [[ID]] MMHostInterfaceMaxBurstINTEL 3
; CHECK-SPIRV:  Decorate [[ID]] MMHostInterfaceReadWriteModeINTEL 2
; CHECK-SPIRV:  Decorate [[ID]] RegisterMapKernelArgumentINTEL
; CHECK-SPIRV:  Decorate [[ID]] StableKernelArgumentINTEL
; CHECK-SPIRV:  Decorate [[ID]] Restrict
; CHECK-SPIRV:  Decorate [[ID]] MMHostInterfaceWaitRequestINTEL 5
