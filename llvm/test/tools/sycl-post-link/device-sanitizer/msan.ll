; This test checks that the post-link tool properly generates "sanUsed=msan"
; in [SYCL/misc properties], and fixes the attributes and metadata of @__MsanKernelMetadata

; RUN: sycl-post-link -properties -split=kernel -symbols -S < %s -o %t.table

; RUN: FileCheck %s -input-file=%t_0.prop --check-prefix CHECK-PROP
; CHECK-PROP: [SYCL/misc properties]
; CHECK-PROP: sanUsed=2|gAAAAAAAAAQbzFmb

; RUN: FileCheck %s -input-file=%t_0.ll --check-prefix CHECK-IR

; ModuleID = 'check_call.cpp'
source_filename = "check_call.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E8MyKernel = comdat any

@__msan_kernel = internal addrspace(1) constant [55 x i8] c"_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E8MyKernel\00"
@__MsanKernelMetadata = appending dso_local local_unnamed_addr addrspace(1) global [1 x { i64, i64 }] [{ i64, i64 } { i64 ptrtoint (ptr addrspace(1) @__msan_kernel to i64), i64 54 }] #0
; CHECK-IR: @__MsanKernelMetadata {{.*}} !spirv.Decorations
@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__asan_func = internal addrspace(2) constant [106 x i8] c"typeinfo name for main::'lambda'(sycl::_V1::handler&)::operator()(sycl::_V1::handler&) const::MyKernelR_4\00"

; Function Attrs: mustprogress norecurse nounwind sanitize_memory uwtable
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E8MyKernel(ptr addrspace(1) noundef align 4 %_arg_array) local_unnamed_addr #1 comdat !srcloc !6 !kernel_arg_buffer_location !7 !sycl_fixed_targets !8 {
entry:
  %0 = load i32, ptr addrspace(1) %_arg_array, align 4
  %1 = ptrtoint ptr addrspace(1) %_arg_array to i64
  %2 = call i64 @__msan_get_shadow(i64 %1, i32 1)
  %3 = inttoptr i64 %2 to ptr addrspace(1)
  %_msld = load i32, ptr addrspace(1) %3, align 4
  %arrayidx3.i = getelementptr inbounds i8, ptr addrspace(1) %_arg_array, i64 4
  %4 = load i32, ptr addrspace(1) %arrayidx3.i, align 4
  %5 = ptrtoint ptr addrspace(1) %arrayidx3.i to i64
  %6 = call i64 @__msan_get_shadow(i64 %5, i32 1)
  %7 = inttoptr i64 %6 to ptr addrspace(1)
  %_msld2 = load i32, ptr addrspace(1) %7, align 4
  %_msprop = sext i32 %_msld2 to i64
  %conv.i = sext i32 %4 to i64
  %_mscmp = icmp ne i32 %_msld, 0
  %_mscmp3 = icmp ne i64 %_msprop, 0
  %_msor = or i1 %_mscmp, %_mscmp3
  %8 = zext i1 %_msor to i8
  call void @__msan_maybe_warning_1(i8 zeroext %8, i32 zeroext 0, ptr addrspace(2) null, i32 0, ptr addrspace(2) @__asan_func)
  %call.i = tail call spir_func noundef i64 @_Z3fooix(i32 noundef %0, i64 noundef %conv.i) #4
  %conv4.i = trunc i64 %call.i to i32
  %9 = ptrtoint ptr addrspace(1) %_arg_array to i64
  %10 = call i64 @__msan_get_shadow(i64 %9, i32 1)
  %11 = inttoptr i64 %10 to ptr addrspace(1)
  store i32 0, ptr addrspace(1) %11, align 4
  store i32 %conv4.i, ptr addrspace(1) %_arg_array, align 4
  ret void
}

; Function Attrs: mustprogress noinline norecurse nounwind sanitize_memory uwtable
define linkonce_odr dso_local spir_func noundef i64 @_Z3fooix(i32 noundef %data1, i64 noundef %data2) local_unnamed_addr #2  {
entry:
  %conv = sext i32 %data1 to i64
  %add = add nsw i64 %data2, %conv
  ret i64 %add
}

declare i64 @__msan_get_shadow(i64, i32)
declare void @__msan_maybe_warning_1(i8, i32, ptr addrspace(2), i32, ptr addrspace(2))

attributes #0 = { "sycl-device-global-size"="16" "sycl-device-image-scope" "sycl-host-access"="0" "sycl-unique-id"="_Z20__MsanKernelMetadata" }
attributes #1 = { mustprogress norecurse nounwind sanitize_memory uwtable "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="check_call.cpp" "sycl-single-task" "uniform-work-group-size"="true" }
attributes #2 = { mustprogress noinline norecurse nounwind sanitize_memory uwtable "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1, !2}
!opencl.spir.version = !{!3}
!spirv.Source = !{!4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 2}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 1, i32 2}
!4 = !{i32 4, i32 100000}
!5 = !{!"clang version 19.0.0git (https://github.com/intel/llvm f8eada76c08c6a5e6c5842842ac5b98fa72669be)"}
!6 = !{i32 563}
!7 = !{i32 -1}
!8 = !{}
