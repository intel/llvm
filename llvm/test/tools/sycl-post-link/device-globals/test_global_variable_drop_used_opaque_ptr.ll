; RUN: sycl-post-link -properties --device-globals -S < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.ll --check-prefix CHECK-IR
;
; Test checks that llvm.compiler.used is removed when all values in it are
; device_global. Likewise it checks that device_global variables that have no
; uses after it are dropped too. This case is using opaque pointers.

source_filename = "test_global_variable.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.cl::sycl::ext::oneapi::device_global.0" = type { ptr addrspace(4) }
%"class.cl::sycl::ext::oneapi::device_global.1" = type { i8 }
%class.anon.0 = type { i8 }

; CHECK-IR-NOT: @llvm.compiler.used =
@llvm.compiler.used = appending global [4 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL7dg_int1 to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL7dg_int2 to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL8dg_bool4 to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL7no_dg_int1 to ptr addrspace(4))]

@_ZL7dg_int1 = weak_odr addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.0" zeroinitializer, align 8, !spirv.Decorations !0 #0
@_ZL7dg_int2 = weak_odr addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.0" zeroinitializer, align 8, !spirv.Decorations !4 #1
@_ZL8dg_bool3 = weak_odr addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.1" zeroinitializer, align 1, !spirv.Decorations !8 #2
@_ZL8dg_bool4 = weak_odr addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.1" zeroinitializer, align 1, !spirv.Decorations !10 #3
@_ZL7no_dg_int1 = weak_odr addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.0" zeroinitializer, align 8, !spirv.Decorations !19 #4

; CHECK-IR: @_ZL7dg_int1 =
; CHECK-IR: @_ZL7dg_int2 =
; CHECK-IR: @_ZL8dg_bool3 =
; CHECK-IR: @_ZL8dg_bool4 =
; CHECK-IR-NOT: @_ZL7no_dg_int1 =

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define weak_odr spir_func void @_ZZ4mainENKUlvE_clEv(ptr addrspace(4) align 1 dereferenceable_or_null(1) %this) #5 align 2 {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  %this.addr.ascast = addrspacecast ptr %this.addr to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr addrspace(4) %this.addr.ascast, align 8
  %this1 = load ptr addrspace(4), ptr addrspace(4) %this.addr.ascast, align 8
  %call1 = call spir_func align 4 dereferenceable(4) ptr addrspace(4) @_ZNK2cl4sycl3ext6oneapi13device_globalIiJNS2_8PropertyIXadsoKcL_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_IXadsoS5_L_ZL5Name2EEEXadsoS5_L_ZL6Value2EEEEENS4_IXadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IXadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv(ptr addrspace(4) align 8 dereferenceable_or_null(8) addrspacecast (ptr addrspace(1) @_ZL7dg_int1 to ptr addrspace(4))) #6
  %call2 = call spir_func align 4 dereferenceable(4) ptr addrspace(4) @_ZNK2cl4sycl3ext6oneapi13device_globalIiJNS2_8PropertyIXadsoKcL_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_IXadsoS5_L_ZL5Name2EEEXadsoS5_L_ZL6Value2EEEEENS4_IXadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IXadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv(ptr addrspace(4) align 8 dereferenceable_or_null(8) addrspacecast (ptr addrspace(1) @_ZL7dg_int2 to ptr addrspace(4))) #6
  %call3 = call spir_func align 1 dereferenceable(1) ptr addrspace(4) @_ZNK2cl4sycl3ext6oneapi13device_globalIbJNS2_8PropertyIXadsoKcL_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_IXadsoS5_L_ZL5Name2EEEXadsoS5_L_ZL6Value2EEEEENS4_IXadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IXadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv(ptr addrspace(4) align 1 dereferenceable_or_null(1) addrspacecast (ptr addrspace(1) @_ZL8dg_bool3 to ptr addrspace(4))) #6
  %call4 = call spir_func align 1 dereferenceable(1) ptr addrspace(4) @_ZNK2cl4sycl3ext6oneapi13device_globalIbJNS2_8PropertyIXadsoKcL_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_IXadsoS5_L_ZL5Name2EEEXadsoS5_L_ZL6Value2EEEEENS4_IXadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IXadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv(ptr addrspace(4) align 1 dereferenceable_or_null(1) addrspacecast (ptr addrspace(1) @_ZL8dg_bool4 to ptr addrspace(4))) #6
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
declare spir_func align 4 dereferenceable(4) ptr addrspace(4) @_ZNK2cl4sycl3ext6oneapi13device_globalIiJNS2_8PropertyIXadsoKcL_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_IXadsoS5_L_ZL5Name2EEEXadsoS5_L_ZL6Value2EEEEENS4_IXadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IXadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv(ptr addrspace(4) align 8 dereferenceable_or_null(8)) #5 align 2

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
declare spir_func align 1 dereferenceable(1) ptr addrspace(4) @_ZNK2cl4sycl3ext6oneapi13device_globalIbJNS2_8PropertyIXadsoKcL_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_IXadsoS5_L_ZL5Name2EEEXadsoS5_L_ZL6Value2EEEEENS4_IXadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IXadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv(ptr addrspace(4) align 1 dereferenceable_or_null(1)) #5 align 2

attributes #0 = { "sycl-device-global-size"="4" "sycl-device-image-scope"="false" "sycl-host-access"="1" "sycl-implement-in-csr"="true" "sycl-init-mode"="0" "sycl-unique-id"="6da74a122db9f35d____ZL7dg_int1" }
attributes #1 = { "sycl-device-global-size"="4" "sycl-implement-in-csr"="false" "sycl-init-mode"="1" "sycl-unique-id"="7da74a1187b9f35d____ZL7dg_int2" }
attributes #2 = { "sycl-device-global-size"="1" "sycl-device-image-scope"="true" "sycl-host-access"="0" "sycl-implement-in-csr" "sycl-init-mode"="0" "sycl-unique-id"="9d329ad59055e972____ZL8dg_bool3" }
attributes #3 = { "sycl-device-global-size"="1" "sycl-device-image-scope" "sycl-host-access"="2" "sycl-unique-id"="dda2bad52c45c432____ZL8dg_bool4" }
attributes #4 = { "sycl-device-global-size"="4" "sycl-device-image-scope"="false" "sycl-host-access"="1" "sycl-implement-in-csr"="true" "sycl-init-mode"="0" "sycl-unique-id"="6da74a122db9f35d____ZL7no_dg_int1" }
attributes #5 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #6 = { convergent nounwind }

!llvm.dependent-libraries = !{!13}
!llvm.module.flags = !{!14, !15}
!opencl.spir.version = !{!16}
!spirv.Source = !{!17}
!llvm.ident = !{!18}

!0 = !{!1, !2, !3}
!1 = !{i32 6149, i32 1}
!2 = !{i32 6148, i32 0}
!3 = !{i32 6147, i32 1, !"6da74a122db9f35d____ZL7dg_int1"}
!4 = !{!5, !6, !7}
!5 = !{i32 6149, i32 0}
!6 = !{i32 6148, i32 1}
!7 = !{i32 6147, i32 2, !"7da74a1187b9f35d____ZL7dg_int2"}
!8 = !{!1, !2, !9}
!9 = !{i32 6147, i32 0, !"9d329ad59055e972____ZL8dg_bool3"}
!10 = !{!11}
!11 = !{i32 6147, i32 2, !"dda2bad52c45c432____ZL8dg_bool4"}
!12 = !{i32 6147, i32 1, !"6da74a122db9f35d____ZL7no_dg_int1"}
!13 = !{!"libcpmt"}
!14 = !{i32 1, !"wchar_size", i32 2}
!15 = !{i32 7, !"frame-pointer", i32 2}
!16 = !{i32 1, i32 2}
!17 = !{i32 4, i32 100000}
!18 = !{!"clang version 14.0.0"}
!19 = !{!1, !2, !12}

; CHECK-IR: !"6da74a122db9f35d____ZL7dg_int1"
; CHECK-IR: !"7da74a1187b9f35d____ZL7dg_int2"
; CHECK-IR: !"9d329ad59055e972____ZL8dg_bool3"
; CHECK-IR: !"dda2bad52c45c432____ZL8dg_bool4"
; CHECK-IR-NOT: !"6da74a122db9f35d____ZL7no_dg_int1"
