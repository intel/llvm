; RUN: sycl-post-link -properties --device-globals --emit-program-metadata -S < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.prop --check-prefix CHECK-PROP

; This test is intended to check that the global_id_mapping program metadata properties are
; generated for device_global variables in a program.

source_filename = "test_global_variable.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.cl::sycl::ext::oneapi::device_global.0" = type { i32 addrspace(4)* }
%"class.cl::sycl::ext::oneapi::device_global.1" = type { i8 }
%class.anon.0 = type { i8 }

@_ZL7dg_int1 = weak_odr addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.0" zeroinitializer, align 8 #0
@_ZL7dg_int2 = weak_odr addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.0" zeroinitializer, align 8 #1
@_ZL8dg_bool3 = weak_odr addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.1" zeroinitializer, align 1 #2
@_ZL8dg_bool4 = weak_odr addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.1" zeroinitializer, align 1 #3
@_ZL7no_dg_int1 = weak_odr addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.0" zeroinitializer, align 8 #6

define weak_odr spir_func void @_ZZ4mainENKUlvE_clEv(%class.anon.0 addrspace(4)* align 1 dereferenceable_or_null(1) %this) #4 align 2 {
entry:
  %this.addr = alloca %class.anon.0 addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %class.anon.0 addrspace(4)** %this.addr to %class.anon.0 addrspace(4)* addrspace(4)*
  store %class.anon.0 addrspace(4)* %this, %class.anon.0 addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %class.anon.0 addrspace(4)*, %class.anon.0 addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %call1 = call spir_func align 4 dereferenceable(4) i32 addrspace(4)* @_ZNK2cl4sycl3ext6oneapi13device_globalIiJNS2_8PropertyIXadsoKcL_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_IXadsoS5_L_ZL5Name2EEEXadsoS5_L_ZL6Value2EEEEENS4_IXadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IXadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv(%"class.cl::sycl::ext::oneapi::device_global.0" addrspace(4)* align 8 dereferenceable_or_null(8) addrspacecast (%"class.cl::sycl::ext::oneapi::device_global.0" addrspace(1)* @_ZL7dg_int1 to %"class.cl::sycl::ext::oneapi::device_global.0" addrspace(4)*)) #5
  %call2 = call spir_func align 4 dereferenceable(4) i32 addrspace(4)* @_ZNK2cl4sycl3ext6oneapi13device_globalIiJNS2_8PropertyIXadsoKcL_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_IXadsoS5_L_ZL5Name2EEEXadsoS5_L_ZL6Value2EEEEENS4_IXadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IXadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv(%"class.cl::sycl::ext::oneapi::device_global.0" addrspace(4)* align 8 dereferenceable_or_null(8) addrspacecast (%"class.cl::sycl::ext::oneapi::device_global.0" addrspace(1)* @_ZL7dg_int2 to %"class.cl::sycl::ext::oneapi::device_global.0" addrspace(4)*)) #5
  %call3 = call spir_func align 1 dereferenceable(1) i8 addrspace(4)* @_ZNK2cl4sycl3ext6oneapi13device_globalIbJNS2_8PropertyIXadsoKcL_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_IXadsoS5_L_ZL5Name2EEEXadsoS5_L_ZL6Value2EEEEENS4_IXadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IXadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv(%"class.cl::sycl::ext::oneapi::device_global.1" addrspace(4)* align 1 dereferenceable_or_null(1) addrspacecast (%"class.cl::sycl::ext::oneapi::device_global.1" addrspace(1)* @_ZL8dg_bool3 to %"class.cl::sycl::ext::oneapi::device_global.1" addrspace(4)*)) #5
  %call4 = call spir_func align 1 dereferenceable(1) i8 addrspace(4)* @_ZNK2cl4sycl3ext6oneapi13device_globalIbJNS2_8PropertyIXadsoKcL_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_IXadsoS5_L_ZL5Name2EEEXadsoS5_L_ZL6Value2EEEEENS4_IXadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IXadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv(%"class.cl::sycl::ext::oneapi::device_global.1" addrspace(4)* align 1 dereferenceable_or_null(1) addrspacecast (%"class.cl::sycl::ext::oneapi::device_global.1" addrspace(1)* @_ZL8dg_bool4 to %"class.cl::sycl::ext::oneapi::device_global.1" addrspace(4)*)) #5
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
declare spir_func align 4 dereferenceable(4) i32 addrspace(4)* @_ZNK2cl4sycl3ext6oneapi13device_globalIiJNS2_8PropertyIXadsoKcL_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_IXadsoS5_L_ZL5Name2EEEXadsoS5_L_ZL6Value2EEEEENS4_IXadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IXadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv(%"class.cl::sycl::ext::oneapi::device_global.0" addrspace(4)* align 8 dereferenceable_or_null(8) %this) #4 align 2

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
declare spir_func align 1 dereferenceable(1) i8 addrspace(4)* @_ZNK2cl4sycl3ext6oneapi13device_globalIbJNS2_8PropertyIXadsoKcL_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_IXadsoS5_L_ZL5Name2EEEXadsoS5_L_ZL6Value2EEEEENS4_IXadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IXadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv(%"class.cl::sycl::ext::oneapi::device_global.1" addrspace(4)* align 1 dereferenceable_or_null(1) %this) #4 align 2

attributes #0 = { "sycl-unique-id"="6da74a122db9f35d____ZL7dg_int1" "sycl-device-image-scope"="false" "sycl-host-access"="1" "sycl-implement-in-csr"="true" "sycl-init-mode"="0" "sycl-device-global-size"="4" }
attributes #1 = { "sycl-unique-id"="7da74a1187b9f35d____ZL7dg_int2" "sycl-implement-in-csr"="false" "sycl-init-mode"="1" "sycl-device-global-size"="4" }
attributes #2 = { "sycl-unique-id"="9d329ad59055e972____ZL8dg_bool3" "sycl-device-image-scope"="true" "sycl-host-access"="0" "sycl-implement-in-csr" "sycl-init-mode"="0" "sycl-device-global-size"="1" }
attributes #3 = { "sycl-unique-id"="dda2bad52c45c432____ZL8dg_bool4" "sycl-device-image-scope" "sycl-host-access"="2" "sycl-device-global-size"="1" }
attributes #4 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #5 = { convergent nounwind }
; no sycl-device-global-size attribute, this is not a device global variable but it contains compile-time properties,
; a metadata node will be generated.
attributes #6 = { "sycl-unique-id"="6da74a122db9f35d____ZL7no_dg_int1" "sycl-device-image-scope"="false" "sycl-host-access"="1" "sycl-implement-in-csr"="true" "sycl-init-mode"="0" }

!llvm.dependent-libraries = !{!0}
!llvm.module.flags = !{!1, !2}
!opencl.spir.version = !{!3}
!spirv.Source = !{!4}
!llvm.ident = !{!5}

!0 = !{!"libcpmt"}
!1 = !{i32 1, !"wchar_size", i32 2}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 1, i32 2}
!4 = !{i32 4, i32 100000}
!5 = !{!"clang version 14.0.0"}

; The encoding of global_id_mapping program metadata entries contain:
;  1. 8 bytes denoting the bit-size of the byte array, here 64 bits or 8 bytes.
;  2. N bytes with the name of the corresponding device_global variable, where
;     N is the value in the previous 8 bytes.
;
; CHECK-PROP: [SYCL/program metadata]
; CHECK-PROP-NEXT: 6da74a122db9f35d____ZL7dg_int1@global_id_mapping=2|YBAAAAAAAAwXax0Nkd2Xp5GdxA
; CHECK-PROP-NEXT: 7da74a1187b9f35d____ZL7dg_int2@global_id_mapping=2|YBAAAAAAAAwXax0Nkd2Xp5GdyA
; CHECK-PROP-NEXT: 9d329ad59055e972____ZL8dg_bool3@global_id_mapping=2|gBAAAAAAAAwXaxEOkd2Xi92bsND
; CHECK-PROP-NEXT: dda2bad52c45c432____ZL8dg_bool4@global_id_mapping=2|gBAAAAAAAAwXaxEOkd2Xi92bsRD
;
; The variable is not a device global one and must be ignored.
; CHECK-PROP-NOT: 6da74a122db9f35d____ZL7no_dg_int1@global_id_mapping
