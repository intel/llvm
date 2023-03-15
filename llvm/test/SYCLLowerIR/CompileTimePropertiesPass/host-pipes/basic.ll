; RUN: opt -passes=compile-time-properties %s -S | FileCheck %s --check-prefix CHECK-IR

; This test is intended to check that CompileTimePropertiesPass adds all the required
; metadata nodes to host pipe vars decorated with the "sycl-host-pipe" attribute

source_filename = "basic.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64_fpga-unknown-unknown"

%struct.BasicKernel = type { i8 }

$_ZN4sycl3_V13ext5intel12experimental9host_pipeI9D2HPipeIDiNS1_6oneapi12experimental10propertiesISt5tupleIJEEEEE6__pipeE = comdat any

@_ZN4sycl3_V13ext5intel12experimental9host_pipeI9D2HPipeIDiNS1_6oneapi12experimental10propertiesISt5tupleIJEEEEE6__pipeE = linkonce_odr dso_local addrspace(1) constant %struct.BasicKernel zeroinitializer, comdat, align 1 #0
; CHECK-IR: @_ZN4sycl3_V13ext5intel12experimental9host_pipeI9D2HPipeIDiNS1_6oneapi12experimental10propertiesISt5tupleIJEEEEE6__pipeE = linkonce_odr dso_local addrspace(1) constant %struct.BasicKernel zeroinitializer, comdat, align 1, !spirv.Decorations ![[#MN0:]]

attributes #0 = { "sycl-host-pipe" "sycl-device-global-size"="4" "sycl-unique-id"="_ZN4sycl3_V13ext5intel12experimental9host_pipeI9H2DPipeIDiNS1_6oneapi12experimental10propertiesISt5tupleIJEEEEE6__pipeE" }

; Ensure that the generated metadata nodes are correct
; CHECK-IR-DAG: ![[#MN0]] = !{![[#MN1:]]}
; CHECK-IR-DAG: ![[#MN1]] = !{i32 6147, i32 2, !"_ZN4sycl3_V13ext5intel12experimental9host_pipeI9H2DPipeIDiNS1_6oneapi12experimental10propertiesISt5tupleIJEEEEE6__pipeE"}
