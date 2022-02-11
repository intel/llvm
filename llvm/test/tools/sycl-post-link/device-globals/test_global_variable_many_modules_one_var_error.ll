; RUN: not sycl-post-link --device-globals --split=source %s -o %t.files.table 2>&1 | FileCheck %s

; This test is intended to check that sycl-post-link does not allow to use a
; single device global variable with the 'device_image_scope' property from
; multiple device images.

; CHECK: sycl-post-link: device_global variable 'dg_int2' with property "device_image_scope" is contained in more than one device image.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.cl::sycl::ext::oneapi::device_global" = type { i32 }
%"class.cl::sycl::detail::accessor_common" = type { i8 }

$_ZTSZ7kernel1RN2cl4sycl5queueEEUlvE_ = comdat any
$_ZTSZ7kernel2RN2cl4sycl5queueEEUlvE_ = comdat any
$_ZTSZ7kernel3RN2cl4sycl5queueEEUlvE_ = comdat any
$_ZTSZ7kernel4RN2cl4sycl5queueEEUlvE_ = comdat any

$dg_int2 = comdat any
@dg_int2 = linkonce_odr dso_local addrspace(1) constant %"class.cl::sycl::ext::oneapi::device_global" zeroinitializer, comdat, align 4 #0

; Third kernel that uses no device-global variables
define weak_odr dso_local spir_kernel void @_ZTSZ7kernel3RN2cl4sycl5queueEEUlvE_() #4 comdat !kernel_arg_buffer_location !6 {
entry:
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define weak_odr dso_local spir_kernel void @_ZTSZ7kernel4RN2cl4sycl5queueEEUlvE_() #3 comdat !kernel_arg_buffer_location !6 {
entry:
  %0 = alloca %"class.cl::sycl::detail::accessor_common", align 1
  %1 = addrspacecast %"class.cl::sycl::detail::accessor_common"* %0 to %"class.cl::sycl::detail::accessor_common" addrspace(4)*
  call spir_func void @_ZZ7kernel1RN2cl4sycl5queueEENKUlvE_clEv(%"class.cl::sycl::detail::accessor_common" addrspace(4)* align 1 dereferenceable_or_null(1) %1) #5
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define weak_odr dso_local spir_kernel void @_ZTSZ7kernel1RN2cl4sycl5queueEEUlvE_() #2 comdat !kernel_arg_buffer_location !6 {
entry:
  %0 = alloca %"class.cl::sycl::detail::accessor_common", align 1
  %1 = addrspacecast %"class.cl::sycl::detail::accessor_common"* %0 to %"class.cl::sycl::detail::accessor_common" addrspace(4)*
  call spir_func void @_ZZ7kernel1RN2cl4sycl5queueEENKUlvE_clEv(%"class.cl::sycl::detail::accessor_common" addrspace(4)* align 1 dereferenceable_or_null(1) %1) #5
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define internal spir_func void @_ZZ7kernel1RN2cl4sycl5queueEENKUlvE_clEv(%"class.cl::sycl::detail::accessor_common" addrspace(4)* align 1 dereferenceable_or_null(1) %this) #1 align 2 {
entry:
  %this.addr = alloca %"class.cl::sycl::detail::accessor_common" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class.cl::sycl::detail::accessor_common" addrspace(4)** %this.addr to %"class.cl::sycl::detail::accessor_common" addrspace(4)* addrspace(4)*
  store %"class.cl::sycl::detail::accessor_common" addrspace(4)* %this, %"class.cl::sycl::detail::accessor_common" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %"class.cl::sycl::detail::accessor_common" addrspace(4)*, %"class.cl::sycl::detail::accessor_common" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  call spir_func void @_Z14kernel1_level1v() #5
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local spir_func void @_Z14kernel1_level1v() #1 {
entry:
  %dg_int_ptr = alloca %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)*, align 8
  %dg_int_ptr.ascast = addrspacecast %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)** %dg_int_ptr to %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* addrspace(4)*
  store %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* addrspacecast (%"class.cl::sycl::ext::oneapi::device_global" addrspace(1)* @dg_int2 to %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)*), %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* addrspace(4)* %dg_int_ptr.ascast, align 8
  %0 = load %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)*, %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* addrspace(4)* %dg_int_ptr.ascast, align 8
  %call = call spir_func align 4 dereferenceable(4) i32 addrspace(4)* @_ZNK2cl4sycl3ext6oneapi13device_globalIiJNS2_8PropertyIPKcXadsoS5_L_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_I11host_accessXadsoS5_L_ZL5Name2EEELS8_1EEENS4_IS6_XadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IS6_XadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv(%"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* align 4 dereferenceable_or_null(4) %0) #7
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define internal spir_func align 4 dereferenceable(4) i32 addrspace(4)* @_ZNK2cl4sycl3ext6oneapi13device_globalIiJNS2_8PropertyIPKcXadsoS5_L_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_I11host_accessXadsoS5_L_ZL5Name2EEELS8_1EEENS4_IS6_XadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IS6_XadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv(%"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* align 4 dereferenceable_or_null(4) %this) #1 align 2 {
entry:
  %retval = alloca i32 addrspace(4)*, align 8
  %this.addr = alloca %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)*, align 8
  %retval.ascast = addrspacecast i32 addrspace(4)** %retval to i32 addrspace(4)* addrspace(4)*
  %this.addr.ascast = addrspacecast %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)** %this.addr to %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* addrspace(4)*
  store %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* %this, %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)*, %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %val = getelementptr inbounds %"class.cl::sycl::ext::oneapi::device_global", %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* %this1, i32 0, i32 0
  ret i32 addrspace(4)* %val
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define weak_odr dso_local spir_kernel void @_ZTSZ7kernel2RN2cl4sycl5queueEEUlvE_() #2 comdat !kernel_arg_buffer_location !6 {
entry:
  %0 = alloca %"class.cl::sycl::detail::accessor_common", align 1
  %1 = addrspacecast %"class.cl::sycl::detail::accessor_common"* %0 to %"class.cl::sycl::detail::accessor_common" addrspace(4)*
  call spir_func void @_ZZ7kernel2RN2cl4sycl5queueEENKUlvE_clEv(%"class.cl::sycl::detail::accessor_common" addrspace(4)* align 1 dereferenceable_or_null(1) %1) #5
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define internal spir_func void @_ZZ7kernel2RN2cl4sycl5queueEENKUlvE_clEv(%"class.cl::sycl::detail::accessor_common" addrspace(4)* align 1 dereferenceable_or_null(1) %this) #1 align 2 {
entry:
  %this.addr = alloca %"class.cl::sycl::detail::accessor_common" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class.cl::sycl::detail::accessor_common" addrspace(4)** %this.addr to %"class.cl::sycl::detail::accessor_common" addrspace(4)* addrspace(4)*
  store %"class.cl::sycl::detail::accessor_common" addrspace(4)* %this, %"class.cl::sycl::detail::accessor_common" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %"class.cl::sycl::detail::accessor_common" addrspace(4)*, %"class.cl::sycl::detail::accessor_common" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %call = call spir_func align 4 dereferenceable(4) i32 addrspace(4)* @_ZNK2cl4sycl3ext6oneapi13device_globalIiJNS2_8PropertyIPKcXadsoS5_L_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_I11host_accessXadsoS5_L_ZL5Name2EEELS8_1EEENS4_IS6_XadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IS6_XadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv.2(%"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* align 4 dereferenceable_or_null(4) addrspacecast (%"class.cl::sycl::ext::oneapi::device_global" addrspace(1)* @dg_int2 to %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)*)) #6
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define internal spir_func align 4 dereferenceable(4) i32 addrspace(4)* @_ZNK2cl4sycl3ext6oneapi13device_globalIiJNS2_8PropertyIPKcXadsoS5_L_ZL5Name1EEEXadsoS5_L_ZL6Value1EEEEENS4_I11host_accessXadsoS5_L_ZL5Name2EEELS8_1EEENS4_IS6_XadsoS5_L_ZL5Name3EEEXadsoS5_L_ZL6Value3EEEEENS4_IS6_XadsoS5_L_ZL5Name4EEEXadsoS5_L_ZL6Value4EEEEEEE3getEv.2(%"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* align 4 dereferenceable_or_null(4) %this) #1 align 2 {
entry:
  %retval = alloca i32 addrspace(4)*, align 8
  %this.addr = alloca %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)*, align 8
  %retval.ascast = addrspacecast i32 addrspace(4)** %retval to i32 addrspace(4)* addrspace(4)*
  %this.addr.ascast = addrspacecast %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)** %this.addr to %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* addrspace(4)*
  store %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* %this, %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)*, %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %val = getelementptr inbounds %"class.cl::sycl::ext::oneapi::device_global", %"class.cl::sycl::ext::oneapi::device_global" addrspace(4)* %this1, i32 0, i32 0
  ret i32 addrspace(4)* %val
}

attributes #0 = { "sycl-unique-id"="dg_int2" "device_image_scope"="true" "host_access"="1" "implement_in_csr"="true" "init_mode"="0" "sycl-device-global-size"="4" }
attributes #1 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="test_global_variable_1.cpp" "uniform-work-group-size"="true" }
attributes #3 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="test_global_variable_2.cpp" "uniform-work-group-size"="true" }
attributes #4 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="test_global_variable_main.cpp" "uniform-work-group-size"="true" }
attributes #5 = { convergent }
attributes #6 = { convergent nounwind }
attributes #7 = { nobuiltin allocsize(0) "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dependent-libraries = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}
!llvm.module.flags = !{!4, !5}

!0 = !{!"libcpmt"}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 14.0.0"}
!4 = !{i32 1, !"wchar_size", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{}
