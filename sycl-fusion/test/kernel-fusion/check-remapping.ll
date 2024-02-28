; RUN: opt -load-pass-plugin %shlibdir/SYCLKernelFusion%shlibext\
; RUN: -passes=sycl-kernel-fusion -S %s\
; RUN: | FileCheck %s

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
; CHECK-NEXT:    call spir_func i64 @__global_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__group_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__local_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__local_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__global_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__num_work_groups_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:    call spir_func i64 @__global_offset_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(i32 %[[X0]]) #[[ATTRS]]
; CHECK-NEXT:    br label %[[EXIT]]
; CHECK-EMPTY:
; CHECK-NEXT:  [[EXIT]]:
; CHECK-NEXT:    call spir_func void @_Z22__spirv_ControlBarrierjjj
; CHECK-NEXT:    %[[GID:.*]] = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
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
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__global_linear_id_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                   ) #[[ATTRS]] {
; CHECK-NEXT:    entry:
; CHECK-NEXT:     %0 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 2)
; CHECK-NEXT:     %1 = mul i64 %0, 1
; CHECK-NEXT:     %2 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 1)
; CHECK-NEXT:     %3 = mul i64 %2, 1
; CHECK-NEXT:     %4 = add i64 %3, %1
; CHECK-NEXT:     %5 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0)
; CHECK-NEXT:     %6 = add i64 %5, %4
; CHECK-NEXT:     ret i64 %6
; CHECK-NEXT:   }

; CHECK-LABEL: define internal spir_func i64 @__global_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                     i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = extractelement <3 x i64> <i64 7, i64 3, i64 2>, i32 %0
; CHECK-NEXT:    ret i64 %1
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__group_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                  i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    %2 = udiv i64 %1, 6
; CHECK-NEXT:    %3 = udiv i64 %2, 1
; CHECK-NEXT:    %4 = insertelement <3 x i64> zeroinitializer, i64 %3, i32 0
; CHECK-NEXT:    %5 = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    %6 = udiv i64 %5, 2
; CHECK-NEXT:    %7 = urem i64 %6, 3
; CHECK-NEXT:    %8 = udiv i64 %7, 1
; CHECK-NEXT:    %9 = insertelement <3 x i64> %4, i64 %8, i32 1
; CHECK-NEXT:    %10 = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    %11 = urem i64 %10, 2
; CHECK-NEXT:    %12 = udiv i64 %11, 2
; CHECK-NEXT:    %13 = insertelement <3 x i64> %9, i64 %12, i32 2
; CHECK-NEXT:    %14 = extractelement <3 x i64> %13, i32 %0
; CHECK-NEXT:    ret i64 %14
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__local_size_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                    i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = extractelement <3 x i64> <i64 1, i64 1, i64 2>, i32 %0
; CHECK-NEXT:    ret i64 %1
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__local_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                  i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    %2 = udiv i64 %1, 6
; CHECK-NEXT:    %3 = urem i64 %2, 1
; CHECK-NEXT:    %4 = insertelement <3 x i64> zeroinitializer, i64 %3, i32 0
; CHECK-NEXT:    %5 = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    %6 = udiv i64 %5, 2
; CHECK-NEXT:    %7 = urem i64 %6, 3
; CHECK-NEXT:    %8 = urem i64 %7, 1
; CHECK-NEXT:    %9 = insertelement <3 x i64> %4, i64 %8, i32 1
; CHECK-NEXT:    %10 = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    %11 = urem i64 %10, 2
; CHECK-NEXT:    %12 = urem i64 %11, 2
; CHECK-NEXT:    %13 = insertelement <3 x i64> %9, i64 %12, i32 2
; CHECK-NEXT:    %14 = extractelement <3 x i64> %13, i32 %0
; CHECK-NEXT:    ret i64 %14
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__global_id_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                   i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    %2 = udiv i64 %1, 6
; CHECK-NEXT:    %3 = insertelement <3 x i64> zeroinitializer, i64 %2, i32 0
; CHECK-NEXT:    %4 = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    %5 = udiv i64 %4, 2
; CHECK-NEXT:    %6 = urem i64 %5, 3
; CHECK-NEXT:    %7 = insertelement <3 x i64> %3, i64 %6, i32 1
; CHECK-NEXT:    %8 = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    %9 = urem i64 %8, 2
; CHECK-NEXT:    %10 = insertelement <3 x i64> %7, i64 %9, i32 2
; CHECK-NEXT:    %11 = extractelement <3 x i64> %10, i32 %0
; CHECK-NEXT:    ret i64 %11
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__num_work_groups_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                         i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = extractelement <3 x i64> <i64 7, i64 3, i64 1>, i32 %0
; CHECK-NEXT:    ret i64 %1
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__global_offset_remapper_3_2_3_7_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                       i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = extractelement <3 x i64> zeroinitializer, i32 %0
; CHECK-NEXT:    ret i64 %1
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__global_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                     i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = extractelement <3 x i64> <i64 4, i64 2, i64 1>, i32 %0
; CHECK-NEXT:    ret i64 %1
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__group_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                  i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    %2 = udiv i64 %1, 2
; CHECK-NEXT:    %3 = udiv i64 %2, 1
; CHECK-NEXT:    %4 = insertelement <3 x i64> zeroinitializer, i64 %3, i32 0
; CHECK-NEXT:    %5 = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    %6 = udiv i64 %5, 1
; CHECK-NEXT:    %7 = urem i64 %6, 2
; CHECK-NEXT:    %8 = udiv i64 %7, 2
; CHECK-NEXT:    %9 = insertelement <3 x i64> %4, i64 %8, i32 1
; CHECK-NEXT:    %10 = extractelement <3 x i64> %9, i32 %0
; CHECK-NEXT:    ret i64 %10
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__local_size_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                    i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = extractelement <3 x i64> <i64 1, i64 2, i64 1>, i32 %0
; CHECK-NEXT:    ret i64 %1
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__local_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                  i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    %2 = udiv i64 %1, 2
; CHECK-NEXT:    %3 = urem i64 %2, 1
; CHECK-NEXT:    %4 = insertelement <3 x i64> zeroinitializer, i64 %3, i32 0
; CHECK-NEXT:    %5 = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    %6 = udiv i64 %5, 1
; CHECK-NEXT:    %7 = urem i64 %6, 2
; CHECK-NEXT:    %8 = urem i64 %7, 2
; CHECK-NEXT:    %9 = insertelement <3 x i64> %4, i64 %8, i32 1
; CHECK-NEXT:    %10 = extractelement <3 x i64> %9, i32 %0
; CHECK-NEXT:    ret i64 %10
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__global_id_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                   i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    %2 = udiv i64 %1, 2
; CHECK-NEXT:    %3 = insertelement <3 x i64> zeroinitializer, i64 %2, i32 0
; CHECK-NEXT:    %4 = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
; CHECK-NEXT:    %5 = udiv i64 %4, 1
; CHECK-NEXT:    %6 = urem i64 %5, 2
; CHECK-NEXT:    %7 = insertelement <3 x i64> %3, i64 %6, i32 1
; CHECK-NEXT:    %8 = extractelement <3 x i64> %7, i32 %0
; CHECK-NEXT:    ret i64 %8
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__num_work_groups_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                         i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = extractelement <3 x i64> <i64 4, i64 1, i64 1>, i32 %0
; CHECK-NEXT:    ret i64 %1
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__global_offset_remapper_2_2_4_1_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                       i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = extractelement <3 x i64> zeroinitializer, i32 %0
; CHECK-NEXT:    ret i64 %1
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__global_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                      i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = extractelement <3 x i64> <i64 48, i64 1, i64 1>, i32 %0
; CHECK-NEXT:    ret i64 %1
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__local_size_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                     i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = extractelement <3 x i64> <i64 2, i64 1, i64 1>, i32 %0
; CHECK-NEXT:    ret i64 %1
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__num_work_groups_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                          i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = extractelement <3 x i64> <i64 24, i64 1, i64 1>, i32 %0
; CHECK-NEXT:    ret i64 %1
; CHECK-NEXT:  }

; CHECK-LABEL: define internal spir_func i64 @__global_offset_remapper_1_48_1_1_2_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                        i32 %0) #[[ATTRS]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %1 = extractelement <3 x i64> zeroinitializer, i32 %0
; CHECK-NEXT:    ret i64 %1
; CHECK-NEXT:  }

declare !sycl.kernel.fused !31 !sycl.kernel.nd-ranges !25 !sycl.kernel.nd-range !24 void @fused_kernel_1D()

; CHECK-LABEL: define spir_kernel void @fused_1(
; CHECK-SAME:                                   i32 %[[X0:.*]], i32 %[[X1:.*]], i32 %[[X2:.*]])
; CHECK-NEXT:   entry:
; CHECK-NEXT:     %[[GID:.*]] = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
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
; CHECK-NEXT:     br label %[[EXIT]]
; CHECK-EMPTY:
; CHECK-NEXT:   [[EXIT]]:
; CHECK-NEXT:     call spir_func void @_Z22__spirv_ControlBarrierjjj
; CHECK-NEXT:     %[[GID:.*]] = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
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
; CHECK-NEXT:     label %[[EXIT]]
; CHECK-EMPTY:
; CHECK-NEXT:   [[EXIT]]:
; CHECK-NEXT:     call spir_func void @_Z22__spirv_ControlBarrierjjj
; CHECK-NEXT:     %[[GID:.*]] = call spir_func i64 @__global_linear_id_3_48_1_1_2_1_1() #[[ATTRS]]
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
; CHECK-NEXT:     br label %[[EXIT]]
; CHECK-EMPTY:
; CHECK-NEXT:   [[EXIT]]:
; CHECK-NEXT:     ret void
; CHECK-NEXT:   }

; CHECK-LABEL: define internal spir_func i64 @__global_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(
; CHECK-SAME:                                                                                       i32 %0) #[[ATTRS]] {
; CHECK-NEXT:   entry:
; CHECK-NEXT:     %1 = extractelement <3 x i64> <i64 20, i64 1, i64 1>, i32 %0
; CHECK-NEXT:     ret i64 %1
; CHECK-NEXT:   }

; CHECK-LABEL: define internal spir_func i64 @__local_size_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(
; CHECK-SAMEE:                                                                                     i32 %0) #[[ATTRS]] {
; CHECK-NEXT:   entry:
; CHECK-NEXT:     %1 = extractelement <3 x i64> <i64 10, i64 1, i64 1>, i32 %0
; CHECK-NEXT:     ret i64 %1
; CHECK-NEXT:   }

; CHECK-LABEL: define internal spir_func i64 @__num_work_groups_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(
; CHECK-SAMEE:                                                                                          i32 %0) #[[ATTRS]] {
; CHECK-NEXT:   entry:
; CHECK-NEXT:     %1 = extractelement <3 x i64> <i64 2, i64 1, i64 1>, i32 %0
; CHECK-NEXT:     ret i64 %1
; CHECK-NEXT:   }

; CHECK-LABEL: define internal spir_func i64 @__global_offset_remapper_1_20_1_1_10_1_1_3_48_1_1_2_1_1(
; CHECK-SAMEE:                                                                                        i32 %0) #[[ATTRS]] {
; CHECK-NEXT:   entry:
; CHECK-NEXT:     %1 = extractelement <3 x i64> zeroinitializer, i32 %0
; CHECK-NEXT:     ret i64 %1
; CHECK-NEXT:   }

; CHECK-LABEL: define internal spir_func i64 @__global_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1(
; CHECK-SAMEE:                                                                                      i32 %0) #[[ATTRS]] {
; CHECK-NEXT:   entry:
; CHECK-NEXT:     %1 = extractelement <3 x i64> <i64 10, i64 1, i64 1>, i32 %0
; CHECK-NEXT:     ret i64 %1
; CHECK-NEXT:   }

; CHECK-LABEL: define internal spir_func i64 @__local_size_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1(
; CHECK-SAMEE:                                                                                     i32 %0) #[[ATTRS]] {
; CHECK-NEXT:   entry:
; CHECK-NEXT:     %1 = extractelement <3 x i64> <i64 10, i64 1, i64 1>, i32 %0
; CHECK-NEXT:     ret i64 %1
; CHECK-NEXT:   }

; CHECK-LABEL: define internal spir_func i64 @__num_work_groups_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1(
; CHECK-SAMEE:                                                                                          i32 %0) #[[ATTRS]] {
; CHECK-NEXT:   entry:
; CHECK-NEXT:     %1 = extractelement <3 x i64> <i64 1, i64 1, i64 1>, i32 %0
; CHECK-NEXT:     ret i64 %1
; CHECK-NEXT:   }

; CHECK-LABEL: define internal spir_func i64 @__global_offset_remapper_1_10_1_1_10_1_1_3_48_1_1_2_1_1(
; CHECK-SAMEE:                                                                                        i32 %0) #[[ATTRS]] {
; CHECK-NEXT:   entry:
; CHECK-NEXT:     %1 = extractelement <3 x i64> zeroinitializer, i32 %0
; CHECK-NEXT:     ret i64 %1
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
