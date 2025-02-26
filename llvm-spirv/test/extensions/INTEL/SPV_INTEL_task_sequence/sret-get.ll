; Test translation of __spirv_TaskSequenceGetINTEL with a sret (structure return) parameter.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_task_sequence -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: TypeTaskSequenceINTEL [[#TypeTS:]]
; CHECK-SPIRV: TypeStruct [[#TypeStruct:]]

; CHECK-SPIRV: TaskSequenceCreateINTEL [[#TypeTS]] [[#TSCreateId:]]
; CHECK-SPIRV: TaskSequenceGetINTEL [[#TypeStruct]] [[#Get:]] [[#TSCreateId]]
; CHECK-SPIRV: Store [[#]] [[#Get]]

; CHECK-LLVM: %struct.FunctionPacket = type <{ float, i8, [3 x i8] }>

; CHECK-LLVM: %[[TSCreate:[a-z0-9.]+]] = call spir_func target("spirv.TaskSequenceINTEL") @_Z66__spirv_TaskSequenceCreateINTEL_RPU3AS125__spirv_TaskSequenceINTELPiiiii(ptr @_Z4multii, i32 -1, i32 -1, i32 1, i32 32)
; CHECK-LLVM: %[[Var:[a-z0-9.]+]] = alloca %struct.FunctionPacket
; CHECK-LLVM: %[[Ptr:[a-z0-9.]+]] = addrspacecast ptr %[[Var]] to ptr addrspace(4)
; CHECK-LLVM: call spir_func void @_Z28__spirv_TaskSequenceGetINTEL{{.*}}(ptr addrspace(4) sret(%struct.FunctionPacket) %[[Ptr]], target("spirv.TaskSequenceINTEL") %[[TSCreate]])

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$_ZTS8MyKernel = comdat any

%struct.FunctionPacket = type <{ float, i8, [3 x i8] }>

; Function Attrs: convergent mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTS8MyKernel(ptr addrspace(1) noundef align 4 %_arg_in, ptr addrspace(1) noundef align 4 %_arg_res) local_unnamed_addr #0 comdat !srcloc !5 !kernel_arg_buffer_location !6 !sycl_fixed_targets !7 !sycl_kernel_omit_args !8 !stall_enable !9 {
entry:
  %call.i1 = call spir_func noundef target("spirv.TaskSequenceINTEL") @_Z31__spirv_TaskSequenceCreateINTELIN4sycl3_V13ext5intel12experimental13task_sequenceIL_Z4multiiENS2_6oneapi12experimental10propertiesISt5tupleIJNS7_14property_valueINS4_13pipelined_keyEJSt17integral_constantIiLin1EEEEENSA_INS4_16fpga_cluster_keyEJSC_INS4_25fpga_cluster_options_enumELSG_1EEEEENSA_INS4_12balanced_keyEJEEENSA_INS4_23invocation_capacity_keyEJSC_IjLj10EEEEENSA_INS4_21response_capacity_keyEJSC_IjLj17EEEEEEEEEEEiJiiEEmPT_PFT0_DpT1_Ejjit(ptr noundef nonnull @_Z4multii, i32 noundef -1, i32 noundef -1, i32 noundef 1, i32 noundef 32) #3
  br label %for.body10.i

for.body10.i:                                     ; preds = %entry, %for.body10.i
  %i5.0.i9 = phi i32 [ 0, %entry ], [ %inc14.i, %for.body10.i ]
  %ref.tmp.i = alloca %struct.FunctionPacket, align 4
  %ref.tmp.ascast.i = addrspacecast ptr %ref.tmp.i to ptr addrspace(4)
  call spir_func void @_Z28__spirv_TaskSequenceGetINTELI14FunctionPacketET_PN5__spv25__spirv_TaskSequenceINTELE(ptr addrspace(4) dead_on_unwind writable sret(%struct.FunctionPacket) align 4 %ref.tmp.ascast.i, target("spirv.TaskSequenceINTEL") noundef %call.i1) #8
  %inc14.i = add nuw nsw i32 %i5.0.i9, 1
  %exitcond.not = icmp eq i32 %inc14.i, 16384
  br i1 %exitcond.not, label %_ZZ4mainENKUlvE_clEv.exit, label %for.body10.i, !llvm.loop !10

_ZZ4mainENKUlvE_clEv.exit:                        ; preds = %for.body10.i
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define linkonce_odr dso_local spir_func noundef i32 @_Z4multii(i32 noundef %a, i32 noundef %b) #1 !srcloc !12 {
entry:
  %mul = mul nsw i32 %b, %a
  ret i32 %mul
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.TaskSequenceINTEL") @_Z31__spirv_TaskSequenceCreateINTELIN4sycl3_V13ext5intel12experimental13task_sequenceIL_Z4multiiENS2_6oneapi12experimental10propertiesISt5tupleIJNS7_14property_valueINS4_13pipelined_keyEJSt17integral_constantIiLin1EEEEENSA_INS4_16fpga_cluster_keyEJSC_INS4_25fpga_cluster_options_enumELSG_1EEEEENSA_INS4_12balanced_keyEJEEENSA_INS4_23invocation_capacity_keyEJSC_IjLj10EEEEENSA_INS4_21response_capacity_keyEJSC_IjLj17EEEEEEEEEEEiJiiEEmPT_PFT0_DpT1_Ejjit(ptr noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef zeroext) local_unnamed_addr #2

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z28__spirv_TaskSequenceGetINTELI14FunctionPacketET_PN5__spv25__spirv_TaskSequenceINTELE(ptr addrspace(4) dead_on_unwind writable sret(%struct.FunctionPacket) align 4, target("spirv.TaskSequenceINTEL") noundef) local_unnamed_addr #2

attributes #0 = { convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-fpga-cluster"="1" "sycl-module-id"="test.cpp" "sycl-optlevel"="2" "sycl-single-task" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-optlevel"="2" }
attributes #2 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent nounwind }


!opencl.spir.version = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!spirv.Source = !{!1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1}
!llvm.ident = !{!2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2}
!llvm.module.flags = !{!3, !4}
!sycl.specialization-constants = !{}
!sycl.specialization-constants-default-values = !{}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 18.0.0git (https://github.com/bowenxue-intel/llvm.git bb1121cb47589e94ab65b81971a298b9d2c21ff6)"}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{i32 5445863}
!6 = !{i32 -1, i32 -1}
!7 = !{}
!8 = !{i1 false, i1 false}
!9 = !{i1 true}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{i32 5445350}
