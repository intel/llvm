; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext\
; RUN: -passes=print-sycl-module-info\
; RUN: --sycl-info-path %S/kernel-info.yaml -disable-output %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Test scenario: Test if the analysis and the printing pass work correctly
; by checking that loading the info from the file in the analysis and 
; subsequently printing the module info matches the file content.

; CHECK-LABEL: Kernels:
; CHECK-NEXT:   - KernelName:      _ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E9KernelOne
; CHECK-NEXT:     Args:
; CHECK-NEXT:       Kinds:           [ Accessor, StdLayout, StdLayout, StdLayout, Accessor,
; CHECK-NEXT:                          StdLayout, StdLayout, StdLayout, Accessor, StdLayout,
; CHECK-NEXT:                          StdLayout, StdLayout ]
; CHECK-NEXT:       Mask:            [ 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1 ]
; CHECK-NEXT:     BinInfo:
; CHECK-NEXT:       Format:          SPIRV
; CHECK-NEXT:       AddressBits:     0
; CHECK-NEXT:       BinarySize:      10612
; CHECK-NEXT:   - KernelName:      _ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E9KernelTwo
; CHECK-NEXT:     Args:
; CHECK-NEXT:       Kinds:           [ Accessor, StdLayout, StdLayout, StdLayout, Accessor,
; CHECK-NEXT:                          StdLayout, StdLayout, StdLayout, Accessor, StdLayout,
; CHECK-NEXT:                          StdLayout, StdLayout ]
; CHECK-NEXT:       Mask:            [ 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1 ]
; CHECK-NEXT:     BinInfo:
; CHECK-NEXT:       Format:          SPIRV
; CHECK-NEXT:       AddressBits:     0
; CHECK-NEXT:       BinarySize:      10612
