; RUN: opt -S -sycllowerwglocalmemory < %s | FileCheck --check-prefixes=CHECK,CHECK-CALL %s

; CHECK-CALL-NOT: __sycl_allocateLocalMemory

; CHECK: [[WGLOCALMEM_1:@WGLocalMem.*]] = internal addrspace(3) global [128 x i8] undef, align 4
; CHECK: [[WGLOCALMEM_2:@WGLocalMem.*]] = internal addrspace(3) global [4 x i8] undef, align 4
; CHECK: [[WGLOCALMEM_3:@WGLocalMem.*]] = internal addrspace(3) global [256 x i8] undef, align 8

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

%"class.range" = type { %"class.array" }
%"class.array" = type { [1 x i64] }
%"class.id" = type { %"class.array" }

$_ZTS7KernelA = comdat any

$_ZTS7KernelB = comdat any

@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInWorkgroupId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTS7KernelA(i32 addrspace(1)* %_arg_, %"class.range"* byval(%"class.range") align 8 %_arg_1, %"class.range"* byval(%"class.range") align 8 %_arg_2, %"class.id"* byval(%"class.id") align 8 %_arg_3, float addrspace(1)* %_arg_4, %"class.range"* byval(%"class.range") align 8 %_arg_6, %"class.range"* byval(%"class.range") align 8 %_arg_7, %"class.id"* byval(%"class.id") align 8 %_arg_8) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !4 {
entry:
  %0 = getelementptr inbounds %"class.id", %"class.id"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast i64* %0 to i64 addrspace(4)*
  %2 = load i64, i64 addrspace(4)* %1, align 8
  %add.ptr.i = getelementptr inbounds i32, i32 addrspace(1)* %_arg_, i64 %2
  %3 = getelementptr inbounds %"class.id", %"class.id"* %_arg_8, i64 0, i32 0, i32 0, i64 0
  %4 = addrspacecast i64* %3 to i64 addrspace(4)*
  %5 = load i64, i64 addrspace(4)* %4, align 8
  %add.ptr.i32 = getelementptr inbounds float, float addrspace(1)* %_arg_4, i64 %5
  %6 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupId to <3 x i64> addrspace(4)*), align 32
  %7 = extractelement <3 x i64> %6, i64 0
  %8 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInLocalInvocationId to <3 x i64> addrspace(4)*), align 32
  %ptridx.i16.i = getelementptr inbounds i32, i32 addrspace(1)* %add.ptr.i, i64 %7
  %ptridx.ascast.i17.i = addrspacecast i32 addrspace(1)* %ptridx.i16.i to i32 addrspace(4)*
  %call.i12.i = tail call spir_func i8 addrspace(3)* @__sycl_allocateLocalMemory(i64 128, i64 4) #2
  ; CHECK: i8 addrspace(3)* getelementptr inbounds ([128 x i8], [128 x i8] addrspace(3)* [[WGLOCALMEM_1]], i32 0, i32 0)
  %shift = shufflevector <3 x i64> %8, <3 x i64> poison, <3 x i32> <i32 undef, i32 2, i32 undef>
  %9 = or <3 x i64> %shift, %8
  %shift56 = shufflevector <3 x i64> %9, <3 x i64> poison, <3 x i32> <i32 1, i32 undef, i32 undef>
  %10 = or <3 x i64> %shift56, %8
  %11 = extractelement <3 x i64> %10, i64 0
  %12 = icmp eq i64 %11, 0
  br i1 %12, label %if.then.i13.i, label %exit.i

if.then.i13.i:                                    ; preds = %entry
  %13 = bitcast i8 addrspace(3)* %call.i12.i to i32 addrspace(3)*
  %arrayidx.i.i.i = addrspacecast i32 addrspace(3)* %13 to i32 addrspace(4)*
  store i32 1, i32 addrspace(4)* %arrayidx.i.i.i, align 4, !tbaa !5
  %14 = load i32, i32 addrspace(4)* %ptridx.ascast.i17.i, align 4, !tbaa !5
  %inc2.i.i.i = add nsw i32 %14, 1
  store i32 %inc2.i.i.i, i32 addrspace(4)* %ptridx.ascast.i17.i, align 4, !tbaa !5
  br label %exit.i

exit.i: ; preds = %if.then.i13.i, %entry
  tail call void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272) #2
  %call.i.i = tail call spir_func i8 addrspace(3)* @__sycl_allocateLocalMemory(i64 4, i64 4) #2
  br i1 %12, label %if.then.i.i, label %"_ZZZ4mainENK3.exit"

if.then.i.i:                                      ; preds = %exit.i
  %ptridx.ascast.i.i = addrspacecast float addrspace(1)* %add.ptr.i32 to float addrspace(4)*
  %15 = bitcast i8 addrspace(3)* %call.i.i to float addrspace(3)*
  ; CHECK: i8 addrspace(3)* getelementptr inbounds ([4 x i8], [4 x i8] addrspace(3)* [[WGLOCALMEM_2]], i32 0, i32 0)
  %16 = addrspacecast float addrspace(3)* %15 to float addrspace(4)*
  %17 = load float, float addrspace(4)* %ptridx.ascast.i.i, align 4, !tbaa !9
  store float %17, float addrspace(4)* %16, align 4, !tbaa !9
  br label %"_ZZZ4mainENK3.exit"

"_ZZZ4mainENK3.exit": ; preds = %exit.i, %if.then.i.i
  tail call void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272) #2
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i8 addrspace(3)* @__sycl_allocateLocalMemory(i64, i64) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local void @_Z22__spirv_ControlBarrierjjj(i32, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTS7KernelB(i64 addrspace(1)* %_arg_, %"class.range"* byval(%"class.range") align 8 %_arg_1, %"class.range"* byval(%"class.range") align 8 %_arg_2, %"class.id"* byval(%"class.id") align 8 %_arg_3) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !11 {
entry:
  %0 = getelementptr inbounds %"class.id", %"class.id"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast i64* %0 to i64 addrspace(4)*
  %2 = load i64, i64 addrspace(4)* %1, align 8
  %add.ptr.i = getelementptr inbounds i64, i64 addrspace(1)* %_arg_, i64 %2
  %call.i.i = tail call spir_func i8 addrspace(3)* @__sycl_allocateLocalMemory(i64 256, i64 8) #2
  ; CHECK: i8 addrspace(3)* getelementptr inbounds ([256 x i8], [256 x i8] addrspace(3)* [[WGLOCALMEM_3]], i32 0, i32 0)
  %3 = bitcast i8 addrspace(3)* %call.i.i to i64 addrspace(3)*
  %arrayidx.i = addrspacecast i64 addrspace(3)* %3 to i64 addrspace(4)*
  %4 = load i64, i64 addrspace(4)* %arrayidx.i, align 8, !tbaa !12
  %ptridx.ascast.i.i = addrspacecast i64 addrspace(1)* %add.ptr.i to i64 addrspace(4)*
  store i64 %4, i64 addrspace(4)* %ptridx.ascast.i.i, align 8, !tbaa !12
  ret void
}

attributes #0 = { convergent norecurse "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 13.0.0"}
!4 = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"float", !7, i64 0}
!11 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!12 = !{!13, !13, i64 0}
!13 = !{!"long", !7, i64 0}
