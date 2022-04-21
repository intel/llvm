; RUN: sycl-post-link --device-globals -S %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.prop --check-prefix CHECK-PROP

; This test is intended to check that sycl-post-link doesn't add the header for
; the empty SYCL/device globals property set.

source_filename = "test_global_variable.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.cl::sycl::ext::oneapi::device_global.0" = type { i32 addrspace(4)* }
%class.anon.0 = type { i8 }

@_ZL7dg_int1 = internal addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global.0" zeroinitializer, align 8 #0

define internal spir_func void @_ZZ4mainENKUlvE_clEv(%class.anon.0 addrspace(4)* align 1 dereferenceable_or_null(1) %this) #1 align 2 {
entry:
  %this.addr = alloca %class.anon.0 addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %class.anon.0 addrspace(4)** %this.addr to %class.anon.0 addrspace(4)* addrspace(4)*
  store %class.anon.0 addrspace(4)* %this, %class.anon.0 addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %class.anon.0 addrspace(4)*, %class.anon.0 addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %call1 = call spir_func align 4 dereferenceable(4) i32 addrspace(4)* @_ZNK2cl4sycl3ext6oneapi13device_globalIiJNS2_8PropertyIXadsoKcL_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_IXadsoS5_L_ZL5Name2EEEXadsoS5_L_ZL6Value2EEEEENS4_IXadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IXadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv(%"class.cl::sycl::ext::oneapi::device_global.0" addrspace(4)* align 8 dereferenceable_or_null(8) addrspacecast (%"class.cl::sycl::ext::oneapi::device_global.0" addrspace(1)* @_ZL7dg_int1 to %"class.cl::sycl::ext::oneapi::device_global.0" addrspace(4)*)) #2
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
declare spir_func align 4 dereferenceable(4) i32 addrspace(4)* @_ZNK2cl4sycl3ext6oneapi13device_globalIiJNS2_8PropertyIXadsoKcL_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_IXadsoS5_L_ZL5Name2EEEXadsoS5_L_ZL6Value2EEEEENS4_IXadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IXadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv(%"class.cl::sycl::ext::oneapi::device_global.0" addrspace(4)* align 8 dereferenceable_or_null(8) %this) #1 align 2

; no the sycl-device-global-size attribute, this is not a device global variable
attributes #0 = { "sycl-unique-id"="6da74a122db9f35d____ZL7dg_int1" "device_image_scope"="false" "host_access"="1" "implement_in_csr"="true" "init_mode"="0" }
attributes #1 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind }

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

; CHECK-PROP-NOT: [SYCL/device globals]
