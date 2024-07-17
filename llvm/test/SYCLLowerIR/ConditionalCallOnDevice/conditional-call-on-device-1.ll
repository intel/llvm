; RUN: opt -passes=sycl-conditional-call-on-device -sycl-conditional-call-on-device-unique-prefix="PREFIX" < %s -S | FileCheck %s

%class.anon = type { ptr addrspace(4) }
%"struct.std::integer_sequence.3" = type { i8 }

define internal spir_func void @call_if_on_device_conditionally_helper(ptr noundef byval(%class.anon) align 8 %fn, ptr noundef byval(%"struct.std::integer_sequence.3") align 1 %0) #2 !srcloc !0 {
entry:
  %agg.tmp = alloca %class.anon, align 8
  %fn.ascast = addrspacecast ptr %fn to ptr addrspace(4)
  call spir_func void @call_if_on_device_conditionally(ptr noundef byval(%class.anon) align 8 %agg.tmp, i32 noundef -2, i32 noundef 251660032) #9
  ret void
}

; CHECK-NOT: call spir_func void @call_if_on_device_conditionally(
; CHECK: call spir_func void @call_if_on_device_conditionally_PREFIX_1(ptr @CallableFunc, ptr %agg.tmp, i32 -2, i32 251660032)

define internal spir_func void @call_if_on_device_conditionally(ptr noundef byval(%class.anon) align 8 %fn, i32 noundef %0, i32 noundef %1) #7 !srcloc !1 {
entry:
  %fn.ascast = addrspacecast ptr %fn to ptr addrspace(4)
  call spir_func void @CallableFunc(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %fn.ascast) #9
  ret void
}

; CHECK-NOT: define internal spir_func void @call_if_on_device_conditionally(

define internal spir_func void @CallableFunc(ptr addrspace(4) noundef align 8 dereferenceable_or_null(8) %this) #6 align 2 !srcloc !2 {
entry:
  ret void
}

; CHECK: declare spir_func void @call_if_on_device_conditionally_PREFIX_1(ptr, ptr, i32, i32)

attributes #2 = { convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #6 = { convergent inlinehint mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #7 = { convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-call-if-on-device-conditionally"="true" }
attributes #9 = { convergent nounwind }

!0 = !{i32 74241}
!1 = !{i32 69449}
!2 = !{i32 835}
