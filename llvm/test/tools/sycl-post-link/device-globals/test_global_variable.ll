; RUN: sycl-post-link --device-globals -S %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.prop --check-prefix CHECK-PROP
; RUN: FileCheck %s -input-file=%t.files_0.ll --check-prefix CHECK-IR
; RUN: sycl-post-link --ir-output-only --device-globals %s -S -o - | FileCheck %s --check-prefix CHECK-IR

; This test is intended to check that DeviceGlobalPass adds all the required
; metadata nodes to every device global variable as well as the required
; properties in the 'SYCL/device globals' property set and handles the
; 'device_image_scope' attribute written in any allowed form.

source_filename = "test_global_variable.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.cl::sycl::ext::oneapi::device_global.0" = type { i32 addrspace(4)* }
%"class.cl::sycl::ext::oneapi::device_global.1" = type { i8 }
%class.anon.0 = type { i8 }

@_ZL7dg_int1 = internal addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.0" zeroinitializer, align 8 #0
; CHECK-IR: @_ZL7dg_int1 = internal addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.0" zeroinitializer, align 8, !spirv.Decorations ![[#MN0:]]
@_ZL7dg_int2 = internal addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.0" zeroinitializer, align 8 #1
; CHECK-IR: @_ZL7dg_int2 = internal addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.0" zeroinitializer, align 8, !spirv.Decorations ![[#MN4:]]
@_ZL8dg_bool3 = internal addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.1" zeroinitializer, align 1 #2
; CHECK-IR: @_ZL8dg_bool3 = internal addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.1" zeroinitializer, align 1, !spirv.Decorations ![[#MN8:]]
@_ZL8dg_bool4 = internal addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.1" zeroinitializer, align 1 #3
; CHECK-IR: @_ZL8dg_bool4 = internal addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.1" zeroinitializer, align 1, !spirv.Decorations ![[#MN10:]]
@_ZL7no_dg_int1 = internal addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.0" zeroinitializer, align 8 #6
; CHECK-IR: @_ZL7no_dg_int1 = internal addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.0" zeroinitializer, align 8, !spirv.Decorations ![[#MN12:]]

define internal spir_func void @_ZZ4mainENKUlvE_clEv(%class.anon.0 addrspace(4)* align 1 dereferenceable_or_null(1) %this) #4 align 2 {
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

attributes #0 = { "sycl-unique-id"="6da74a122db9f35d____ZL7dg_int1" "device_image_scope"="false" "host_access"="1" "implement_in_csr"="true" "init_mode"="0" "sycl-device-global-size"="4" }
attributes #1 = { "sycl-unique-id"="7da74a1187b9f35d____ZL7dg_int2" "implement_in_csr"="false" "init_mode"="1" "sycl-device-global-size"="4" }
attributes #2 = { "sycl-unique-id"="9d329ad59055e972____ZL8dg_bool3" "device_image_scope"="true" "host_access"="0" "implement_in_csr" "init_mode"="0" "sycl-device-global-size"="1" }
attributes #3 = { "sycl-unique-id"="dda2bad52c45c432____ZL8dg_bool4" "device_image_scope" "host_access"="2" "sycl-device-global-size"="1" }
attributes #4 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #5 = { convergent nounwind }
; no sycl-device-global-size attribute, this is not a device global variable but it contains compile-time properties,
; a metadata node will be generated.
attributes #6 = { "sycl-unique-id"="6da74a122db9f35d____ZL7no_dg_int1" "device_image_scope"="false" "host_access"="1" "implement_in_csr"="true" "init_mode"="0" }

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

; Ensure that the generated metadata nodes are correct
; CHECK-IR-DAG: ![[#MN0]] = !{![[#MN1:]], ![[#MN2:]], ![[#MN3:]]}
; CHECK-IR-DAG: ![[#MN1]] = !{i32 6149, i32 1}
; CHECK-IR-DAG: ![[#MN2]] = !{i32 6148, i32 0}
; CHECK-IR-DAG: ![[#MN3]] = !{i32 6147, i32 1, !"6da74a122db9f35d____ZL7dg_int1"}

; CHECK-IR-DAG: ![[#MN4]] = !{![[#MN5:]], ![[#MN6:]], ![[#MN7:]]}
; CHECK-IR-DAG: ![[#MN5]] = !{i32 6149, i32 0}
; CHECK-IR-DAG: ![[#MN6]] = !{i32 6148, i32 1}
; CHECK-IR-DAG: ![[#MN7]] = !{i32 6147, i32 2, !"7da74a1187b9f35d____ZL7dg_int2"}

; CHECK-IR-DAG: ![[#MN8]] = !{![[#MN1]], ![[#MN2]], ![[#MN9:]]}
; CHECK-IR-DAG: ![[#MN9]] = !{i32 6147, i32 0, !"9d329ad59055e972____ZL8dg_bool3"}

; CHECK-IR-DAG: ![[#MN10]] = !{![[#MN11:]]}
; CHECK-IR-DAG: ![[#MN11]] = !{i32 6147, i32 2, !"dda2bad52c45c432____ZL8dg_bool4"}

; For not a device global variable, only actually present compile-time
; properties are handled
; CHECK-IR-DAG: ![[#MN12]] = !{![[#MN1]], ![[#MN2]]}

; Ensure that the default values are correct.
; ABAAAAAAAAABAAAAAxxxxx is decoded to
; "40 00 00 00 00 00 00 00 | 04 00 00 00 | 00 | xx xx xx" which consists of:
;  1. 8 bytes denoting the bit-size of the byte array, here 64 bits or 8 bytes.
;  2. 4 bytes with the value of the 32-bit uint32_t integer with the size of the
;     underlying type of the device global variable. Its value being 4.
;  3. 1 byte with the value of the 8-bit uint8_t integer with the flag that
;     the device global variable has the 'device_image_scope' property.
;     Its value being 0, no property.
;  4. Any 3 bytes used as padding to align the structure to 8 bytes.
;
; ABAAAAAAAAQAAAAABxxxxx is decoded to
; "40 00 00 00 00 00 00 00 | 01 00 00 00 | 01 | xx xx xx" which consists of:
;  1. 8 bytes denoting the bit-size of the byte array, here 64 bits or 8 bytes.
;  2. 4 bytes with the value of the 32-bit uint32_t integer with the size of the
;     underlying type of the device global variable. Its value being 1.
;  3. 1 byte with the value of the 8-bit uint8_t integer with the flag that
;     the device global variable has the 'device_image_scope' property.
;     Its value being 1, property is present.
;  4. Any 3 bytes used as padding to align the structure to 8 bytes.
;
; CHECK-PROP: [SYCL/device globals]
; CHECK-PROP-NEXT: 6da74a122db9f35d____ZL7dg_int1=2|ABAAAAAAAAABAAAAA
; CHECK-PROP-NEXT: 7da74a1187b9f35d____ZL7dg_int2=2|ABAAAAAAAAABAAAAA
; CHECK-PROP-NEXT: 9d329ad59055e972____ZL8dg_bool3=2|ABAAAAAAAAQAAAAAB
; CHECK-PROP-NEXT: dda2bad52c45c432____ZL8dg_bool4=2|ABAAAAAAAAQAAAAAB
;
; The variable is not a device global one and must be ignored
; CHECK-PROP-NOT: 6da74a122db9f35d____ZL7no_dg_int1
