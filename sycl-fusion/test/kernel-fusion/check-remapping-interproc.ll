; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext\
; RUN: -passes=sycl-kernel-fusion -S %s | FileCheck %s

; This tests checks that SPIR-V builtins are correctly remapped when fusing
;  kernels with different ND-ranges.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"


; Function Attrs: alwaysinline nounwind
declare spir_func void @__itt_offload_wi_start_wrapper() #3

; Function Attrs: alwaysinline nounwind
declare spir_func void @__itt_offload_wi_finish_wrapper() #3

declare spir_func i64 @_Z25__spirv_BuiltInGlobalSizei(i32) #4
declare spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32) #4
declare spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32) #4
declare spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32) #4
declare spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32) #4
declare spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32) #4
declare spir_func i64 @_Z27__spirv_BuiltInGlobalOffseti(i32) #4

define spir_func void @foo(i32 %x) {
entry:
  %0 = call spir_func i64 @_Z25__spirv_BuiltInGlobalSizei(i32 %x) #4
  %1 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 %x) #4
  %2 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 %x) #4
  %3 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 %x) #4
  %4 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 %x) #4
  %5 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 %x) #4
  %6 = call spir_func i64 @_Z27__spirv_BuiltInGlobalOffseti(i32 %x) #4
  ret void
}

define spir_func i32 @bar(i32 %x) {
entry:
  %cmp = icmp ule i32 %x, 1
  br i1 %cmp, label %return, label %if.end

if.end:
  %sub = sub i32 %x, 1
  %call = call i32 @bar(i32 %sub)
  %mul = mul i32 %x, %call
  br label %return

return:
  %res = phi i32 [%x, %entry], [%sub, %if.end]
  ret i32 %res
}

define spir_func void @baz(i32 %x) {
entry:
  ret void
}

; Function Attrs: nounwind
define spir_kernel void @KernelOne(i32 %x) #2 !kernel_arg_addr_space !6 !kernel_arg_access_qual !6 !kernel_arg_type !6 !kernel_arg_type_qual !6 !kernel_arg_base_type !6 !kernel_arg_name !6 !work_group_size_hint !11 {
entry:
  call spir_func void @__itt_offload_wi_start_wrapper() #3
  %0 = call spir_func i64 @_Z25__spirv_BuiltInGlobalSizei(i32 %x) #4
  %1 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 %x) #4
  %2 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 %x) #4
  %3 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 %x) #4
  %4 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 %x) #4
  %5 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 %x) #4
  %6 = call spir_func i64 @_Z27__spirv_BuiltInGlobalOffseti(i32 %x) #4
  call spir_func void @foo(i32 %x)
  %y = call spir_func i32 @bar(i32 %x)
  call spir_func void @baz(i32 %y)
  call spir_func void @__itt_offload_wi_finish_wrapper() #3
  ret void
}

declare !sycl.kernel.fused !13 !sycl.kernel.nd-ranges !15 !sycl.kernel.nd-range !24 void @fused_kernel()

; CHECK-LABEL: define spir_kernel void @fused_0(
; CHECK-SAME:                                   i32 %[[X0:.*]], i32 %[[X1:.*]], i32 %[[X2:.*]])
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %[[GID:.*]] = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS:.*]]
; CHECK-NEXT:    %[[CMP:.*]] = icmp ult i64 %[[GID]], 42
; CHECK-NEXT:    br i1 %[[CMP]], label %[[CALL:.*]], label %[[EXIT:.*]]
; CHECK-EMPTY:
; CHECK-NEXT:  [[CALL]]:
; CHECK-NEXT:    call spir_func i64 @__global_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS:.*]]
; CHECK-NEXT:    call spir_func i64 @__group_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__local_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__local_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__global_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__num_work_groups_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__global_offset_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func void @foo.2(i32 %[[X0]])
; CHECK-NEXT:    %[[Y0:.*]] = call spir_func i32 @bar.3(i32 %[[X0]])
; CHECK-NEXT:    call spir_func void @baz(i32 %[[Y0]])
; CHECK-NEXT:    br label %[[EXIT]]
; CHECK-EMPTY:
; CHECK-NEXT:  [[EXIT]]:
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrierjjj
; CHECK-NEXT:    %[[GID:.*]] = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS:.*]]
; CHECK-NEXT:    %[[CMP:.*]] = icmp ult i64 %[[GID]], 8
; CHECK-NEXT:    br i1 %[[CMP]], label %[[CALL:.*]], label %[[EXIT:.*]]
; CHECK-EMPTY:
; CHECK-NEXT:  [[CALL]]:
; CHECK-NEXT:    call spir_func i64 @__global_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X1]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__group_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X1]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__local_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X1]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__local_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X1]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__global_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X1]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__num_work_groups_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X1]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__global_offset_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X1]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func void @foo.6(i32 %[[X1]])
; CHECK-NEXT:    %[[Y1:.*]] = call spir_func i32 @bar.7(i32 %[[X1]])
; CHECK-NEXT:    call spir_func void @baz(i32 %[[Y1]])
; CHECK-NEXT:    br label %[[EXIT]]
; CHECK-EMPTY:
; CHECK-NEXT:  [[EXIT]]:
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrierjjj
; CHECK-NEXT:    call spir_func i64 @__global_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X2]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__group_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X2]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__local_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X2]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__local_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X2]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__global_id_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X2]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__num_work_groups_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X2]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__global_offset_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X2]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func void @foo.10(i32 %[[X2]])
; CHECK-NEXT:    %[[Y2:.*]] = call spir_func i32 @bar.11(i32 %[[X2]])
; CHECK-NEXT:    call spir_func void @baz(i32 %[[Y2]])
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }

; CHECK-LABEL: define spir_func void @foo.2(
; CHECK-SAME:                               i32 %[[X:.*]]) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call spir_func i64 @__global_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X]]) #[[ATTRS]]
; CHECK-NEXT:   call spir_func i64 @__group_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X]]) #[[ATTRS]]
; CHECK-NEXT:   call spir_func i64 @__local_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X]]) #[[ATTRS]]
; CHECK-NEXT:   call spir_func i64 @__local_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X]]) #[[ATTRS]]
; CHECK-NEXT:   call spir_func i64 @__global_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X]]) #[[ATTRS]]
; CHECK-NEXT:   call spir_func i64 @__num_work_groups_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X]]) #[[ATTRS]]
; CHECK-NEXT:   call spir_func i64 @__global_offset_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X]]) #[[ATTRS]]
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK-LABEL: define spir_func i32 @bar.3(
; CHECK:        call spir_func i32 @bar.3

; CHECK-LABEL: define spir_func void @foo.6(
; CHECK-SAME:                               i32 %[[X:.*]]) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call spir_func i64 @__global_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X]]) #[[ATTRS]]
; CHECK-NEXT:   call spir_func i64 @__group_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X]]) #[[ATTRS]]
; CHECK-NEXT:   call spir_func i64 @__local_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X]]) #[[ATTRS]]
; CHECK-NEXT:   call spir_func i64 @__local_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X]]) #[[ATTRS]]
; CHECK-NEXT:   call spir_func i64 @__global_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X]]) #[[ATTRS]]
; CHECK-NEXT:   call spir_func i64 @__num_work_groups_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X]]) #[[ATTRS]]
; CHECK-NEXT:   call spir_func i64 @__global_offset_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(i32 %[[X]]) #[[ATTRS]]
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

declare !sycl.kernel.fused !31 !sycl.kernel.nd-ranges !25 !sycl.kernel.nd-range !24 void @fused_kernel_1D()

; CHECK-LABEL: define spir_kernel void @fused_1(
; CHECK-SAME:                                   i32 %[[X0:.*]], i32 %[[X1:.*]], i32 %[[X2:.*]])
; CHECK-NEXT:   entry:
; CHECK-NEXT:     %[[GID:.*]] = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS:.*]]
; CHECK-NEXT:     %[[CMP:.*]] = icmp ult i64 %[[GID]], 20
; CHECK-NEXT:     br i1 %[[CMP]], label %[[CALL:.*]], label %[[EXIT:.*]]
; CHECK-EMPTY:
; CHECK-NEXT:   [[CALL]]
; CHECK-NEXT:     call spir_func i64 @__global_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__group_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__local_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__global_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__num_work_groups_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__global_offset_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func void @foo.14(i32 %[[X0]])
; CHECK-NEXT:     %[[Y0:.*]] = call spir_func i32 @bar.15(i32 %[[X0]])
; CHECK-NEXT:     call spir_func void @baz(i32 %[[Y0]])
; CHECK-NEXT:     br label %[[EXIT]]
; CHECK-EMPTY:
; CHECK-NEXT:   [[EXIT]]:
; CHECK-NEXT:     call spir_func void @_Z22__spirv_ControlBarrierjjj
; CHECK-NEXT:     %[[GID:.*]] = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS:.*]]
; CHECK-NEXT:     %[[CMP:.*]] = icmp ult i64 %[[GID]], 10
; CHECK-NEXT:     br i1 %[[CMP]], label %[[CALL:.*]], label %[[EXIT:.*]]
; CHECK-EMPTY:
; CHECK-NEXT:   [[CALL]]:
; CHECK-NEXT:     call spir_func i64 @__global_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X1]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__group_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X1]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__local_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X1]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__local_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X1]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__global_id_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X1]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__num_work_groups_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X1]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__global_offset_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X1]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func void @foo.18(i32 %[[X1]])
; CHECK-NEXT:     %[[Y1:.*]] = call spir_func i32 @bar.19(i32 %[[X1]])
; CHECK-NEXT:     call spir_func void @baz(i32 %[[Y1]])
; CHECK-NEXT:     label %[[EXIT]]
; CHECK-EMPTY:
; CHECK-NEXT:   [[EXIT]]:
; CHECK-NEXT:     call spir_func void @_Z22__spirv_ControlBarrierjjj
; CHECK-NEXT:     %[[GID:.*]] = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS:.*]]
; CHECK-NEXT:     %[[CMP:.*]] = icmp ult i64 %[[GID]], 20
; CHECK-NEXT:     br i1 %[[CMP]], label %[[CALL:.*]], label %[[EXIT:.*]]
; CHECK-EMPTY:
; CHECK-NEXT:   [[CALL]]:
; CHECK-NEXT:     call spir_func i64 @__global_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X2]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__group_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X2]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X2]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__local_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X2]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__global_id_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X2]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__num_work_groups_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X2]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func i64 @__global_offset_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(i32 %[[X2]]) #[[ATTRS]]
; CHECK-NEXT:     call spir_func void @foo.14(i32 %[[X2]])
; CHECK-NEXT:     %[[Y2:.*]] = call spir_func i32 @bar.15(i32 %[[X2]])
; CHECK-NEXT:     call spir_func void @baz(i32 %[[Y2]])
; CHECK-NEXT:     br label %[[EXIT]]
; CHECK-EMPTY:
; CHECK-NEXT:   [[EXIT]]:
; CHECK-NEXT:     ret void
; CHECK-NEXT:   }

; This should be the last test

declare !sycl.kernel.fused !41 !sycl.kernel.nd-ranges !42 !sycl.kernel.nd-range !43 void @fused_kernel_homogeneous()

; CHECK-LABEL: define spir_kernel void @fused_3
; CHECK-NOT:   remapper

attributes #2 = { nounwind }
attributes #3 = { alwaysinline nounwind }
attributes #4 = { willreturn nounwind }

; CHECK:       attributes #[[ATTRS]] = { alwaysinline nounwind }

!6 = !{}
!11 = !{i32 64, i32 1, i32 1}
!12 = !{!"_arg_y"}
!13 = !{!"fused_0", !14}
!14 = !{!"KernelOne", !"KernelOne", !"KernelOne"}
!15 = !{!16, !17, !18}
!16 = !{i32 3, !19, !20, !21}
!17 = !{i32 2, !22, !20, !21}
!18 = !{i32 1, !23, !20, !21}
!19 = !{i64 2, i64 3, i64 7}
!20 = !{i64 2, i64 1, i64 1}
!21 = !{i64 0, i64 0, i64 0}
!22 = !{i64 2, i64 4, i64 1}
!23 = !{i64 48, i64 1, i64 1}
!24 = !{i32 3, !23, !20, !21}
!25 = !{!26, !27, !26}
!26 = !{i32 1, !28, !29, !21}
!27 = !{i32 1, !30, !29, !21}
!28 = !{i64 20, i64 1, i64 1}
!29 = !{i64 10, i64 1, i64 1}
!30 = !{i64 10, i64 1, i64 1}
!31 = !{!"fused_1", !14}
!32 = !{!"fused_2", !14}
!33 = !{!34, !35, !36}
!34 = !{i32 3, !37, !38, !21}
!35 = !{i32 3, !39, !38, !21}
!36 = !{i32 3, !40, !38, !21}
!37 = !{i64 60, i64 60, i64 60}
!38 = !{i64 2, i64 3, i64 20}
!39 = !{i64 2, i64 6, i64 40}
!40 = !{i64 6, i64 30, i64 60}
!41 = !{!"fused_3", !14}
!42 = !{!43, !43, !43}
!43 = !{i32 3, !44, !45, !46}
!44 = !{i64 100, i64 100, i64 100}
!45 = !{i64 10, i64 10, i64 10}
!46 = !{i64 0, i64 0, i64 0}
!47 = !{
  !"KernelOne",
  !{!"Accessor", !"StdLayout", !"StdLayout", !"StdLayout", !"Accessor",
    !"StdLayout", !"StdLayout", !"StdLayout", !"Accessor", !"StdLayout",
    !"StdLayout", !"StdLayout"},
  !{i8 1, i8 0, i8 0, i8 1, i8 1, i8 0, i8 0, i8 1, i8 1, i8 0, i8 0, i8 1},
  !{!"work_group_size_hint", i32 1, i32 1, i32 64}
}
!sycl.moduleinfo = !{!47}
