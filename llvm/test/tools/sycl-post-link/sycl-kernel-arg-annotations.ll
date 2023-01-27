; RUN: sycl-post-link --device-globals --ir-output-only -S %s -o %t.ll
; RUN: FileCheck %s -input-file=%t.ll
;
; TODO: Remove --device-globals once other features start using compile-time
;       properties.
;
; Tests the translation of "sycl-kernel-arg-attribute" to "spirv.ParameterDecorations" metadata

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64_fpga-unknown-unknown"

$singleArg = comdat any
$multiArg = comdat any

; Function Attrs: convergent mustprogress norecurse
define weak_odr dso_local spir_kernel void @singleArg(i32 addrspace(4)* noundef align 4 "sycl-alignment"="4" "sycl-awidth"="32" "sycl-buffer-location"="10" "sycl-conduit" "sycl-dwidth"="64" "sycl-latency"="1" "sycl-maxburst"="3" "sycl-read-write-mode"="2" "sycl-register-map" "sycl-stable" "sycl-strict" "sycl-wait-request"="5" %_arg_p) #0 comdat !kernel_arg_buffer_location !1587
; CHECK-DAG: !spirv.ParameterDecorations ![[PARMDECOR_CASE1:[0-9]+]]
{

entry:
		ret void
}

define weak_odr dso_local spir_kernel void @multiArg(i32 addrspace(4)* noundef align 4 "sycl-alignment"="8" %_arg_a, i32 addrspace(4)* noundef %_arg_b, i32 addrspace(4)* noundef align 4 "sycl-awidth"="64" %_arg_c) #0 comdat !kernel_arg_buffer_location !1588
; CHECK-DAG: !spirv.ParameterDecorations ![[PARMDECOR_CASE2:[0-9]+]]
{

entry:
		ret void
}

!1587 = !{i32 -1}
!1588 = !{i32 -1, i32 -1, i32 -1}
; CHECK-DAG: ![[PARMDECOR_CASE1]] = !{![[ARG:[0-9]+]]}
; CHECK-DAG: ![[ARG]] = !{![[ALIGN:[0-9]+]], ![[AWIDTH:[0-9]+]], ![[BL:[0-9]+]], ![[CONDUIT:[0-9]+]], ![[DWIDTH:[0-9]+]], ![[LATENCY:[0-9]+]], ![[MAXBURST:[0-9]+]], ![[RWMODE:[0-9]+]], ![[REGMAP:[0-9]+]], ![[STABLE:[0-9]+]], ![[STRICT:[0-9]+]], ![[WAITREQ:[0-9]+]]}

; CHECK: ![[ALIGN]]   = !{i32 6182, i32 4}
; CHECK: ![[AWIDTH]]  = !{i32 6177, i32 32}
; CHECK: ![[BL]]      = !{i32 5921, i32 10}
; CHECK: ![[CONDUIT]] = !{i32 6175, i32 1}
; CHECK: ![[DWIDTH]]  = !{i32 6178, i32 64}
; CHECK: ![[LATENCY]] = !{i32 6179, i32 1}
; CHECK: ![[MAXBURST]] = !{i32 6181, i32 3}
; CHECK: ![[RWMODE]]  = !{i32 6180, i32 2}
; CHECK: ![[REGMAP]]  = !{i32 6176, i32 1}
; CHECK: ![[STABLE]]  = !{i32 6184, i32 1}
; CHECK: ![[STRICT]]  = !{i32 19, i32 1}
; CHECK: ![[WAITREQ]] = !{i32 6183, i32 5}

; CHECK-DAG: ![[PARMDECOR_CASE2]] = !{![[ARG1:[0-9]+]], ![[ARG2:[0-9]+]], ![[ARG3:[0-9]+]]}
; CHECK-DAG: ![[ARG1]] = !{![[ALIGN1:[0-9]+]]}
; CHECK: ![[ALIGN1]]   = !{i32 6182, i32 8}
; CHECK-DAG: ![[ARG2]] = !{}
; CHECK-DAG: ![[ARG3]] = !{![[AWIDTH3:[0-9]+]]}
; CHECK: ![[AWIDTH3]]  = !{i32 6177, i32 64}
