; RUN: not sycl-post-link -split=auto -ir-output-only -sycl-rdc=false %s 2>&1 | FileCheck %s
; Verify -ir-output-only and -sycl-rdc=false together throw an error
; CHECK: error: -sycl-rdc can't be used with -ir-output-only
