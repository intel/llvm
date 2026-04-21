; RUN: sycl-post-link -properties --device-globals -S < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.ll --check-prefix CHECK-IR
;
; Test checks that all device_global variables in llvm.compiler.used are removed
; but any other values stay in.

source_filename = "test_global_variable.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.cl::sycl::ext::oneapi::device_global.0" = type { ptr addrspace(4) }
%class.anon.0 = type { i8 }

; CHECK-IR: @llvm.compiler.used = appending global [1 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL16NotADeviceGlobal to ptr addrspace(4))]
@llvm.compiler.used = appending global [3 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (%"class.cl::sycl::ext::oneapi::device_global.0" addrspace(1)* @_ZL7dg_int1 to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL16NotADeviceGlobal to ptr addrspace(4)), ptr addrspace(4) addrspacecast (%"class.cl::sycl::ext::oneapi::device_global.0" addrspace(1)* @_ZL7dg_int2 to ptr addrspace(4))]

@_ZL7dg_int1 = internal addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.0" zeroinitializer, align 8, !spirv.Decorations !0 #0
@_ZL7dg_int2 = internal addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.0" zeroinitializer, align 8, !spirv.Decorations !4 #1

@_ZL16NotADeviceGlobal = internal addrspace(1) constant i8 zeroinitializer

attributes #0 = { "sycl-device-global-size"="4" "sycl-device-image-scope"="false" "sycl-host-access"="1" "sycl-implement-in-csr"="true" "sycl-init-mode"="0" "sycl-unique-id"="6da74a122db9f35d____ZL7dg_int1" }
attributes #1 = { "sycl-device-global-size"="4" "sycl-implement-in-csr"="false" "sycl-init-mode"="1" "sycl-unique-id"="7da74a1187b9f35d____ZL7dg_int2" }
attributes #5 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #6 = { convergent nounwind }

!llvm.dependent-libraries = !{!8}
!llvm.module.flags = !{!9, !10}
!opencl.spir.version = !{!11}
!spirv.Source = !{!12}
!llvm.ident = !{!13}

!0 = !{!1, !2, !3}
!1 = !{i32 6149, i32 1}
!2 = !{i32 6148, i32 0}
!3 = !{i32 6147, i32 1, !"6da74a122db9f35d____ZL7dg_int1"}
!4 = !{!5, !6, !7}
!5 = !{i32 6149, i32 0}
!6 = !{i32 6148, i32 1}
!7 = !{i32 6147, i32 2, !"7da74a1187b9f35d____ZL7dg_int2"}
!8 = !{!"libcpmt"}
!9 = !{i32 1, !"wchar_size", i32 2}
!10 = !{i32 7, !"frame-pointer", i32 2}
!11 = !{i32 1, i32 2}
!12 = !{i32 4, i32 100000}
!13 = !{!"clang version 14.0.0"}
