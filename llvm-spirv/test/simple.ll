; RUN: llvm-as < %s | llvm-spirv -spirv-text -o %t
; RUN: FileCheck < %t %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Support of doubles is required.
; CHECK: Capability Float64
; CHECK: "fun01"
; Function Attrs: nounwind
define spir_kernel void @fun01(ptr addrspace(1) noalias %a, ptr addrspace(1) %b, i32 %c) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 !reqd_work_group_size !6 {
entry:
  %a.addr = alloca ptr addrspace(1), align 8
  %b.addr = alloca ptr addrspace(1), align 8
  %c.addr = alloca i32, align 4
  store ptr addrspace(1) %a, ptr %a.addr, align 8
  store ptr addrspace(1) %b, ptr %b.addr, align 8
  store i32 %c, ptr %c.addr, align 4
  %0 = load ptr addrspace(1), ptr %b.addr, align 8
  %1 = load i32, ptr addrspace(1) %0, align 4
  %2 = load ptr addrspace(1), ptr %a.addr, align 8
  store i32 %1, ptr addrspace(1) %2, align 4
  %3 = load ptr addrspace(1), ptr %b.addr, align 8
  %cmp = icmp ugt ptr addrspace(1) %3, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %4 = load ptr addrspace(1), ptr %a.addr, align 8
  store i32 2, ptr addrspace(1) %4, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; CHECK: "fun02"
; Function Attrs: nounwind
define spir_kernel void @fun02(ptr addrspace(1) %a, ptr addrspace(1) %b, i32 %c) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !8 !kernel_arg_base_type !9 !kernel_arg_type_qual !10 !vec_type_hint !11 {
entry:
  %a.addr = alloca ptr addrspace(1), align 8
  %b.addr = alloca ptr addrspace(1), align 8
  %c.addr = alloca i32, align 4
  store ptr addrspace(1) %a, ptr %a.addr, align 8
  store ptr addrspace(1) %b, ptr %b.addr, align 8
  store i32 %c, ptr %c.addr, align 4
  %0 = load i32, ptr %c.addr, align 4
  %idxprom = sext i32 %0 to i64
  %1 = load ptr addrspace(1), ptr %b.addr, align 8
  %arrayidx = getelementptr inbounds double, ptr addrspace(1) %1, i64 %idxprom
  %2 = load double, ptr addrspace(1) %arrayidx, align 8
  %3 = load i32, ptr %c.addr, align 4
  %idxprom1 = sext i32 %3 to i64
  %4 = load ptr addrspace(1), ptr %a.addr, align 8
  %arrayidx2 = getelementptr inbounds double, ptr addrspace(1) %4, i64 %idxprom1
  store double %2, ptr addrspace(1) %arrayidx2, align 8
  ret void
}

; CHECK: "test_builtin"
; Function Attrs: nounwind
define spir_func void @test_builtin(ptr addrspace(1) %in, ptr addrspace(1) %out) #0 {
entry:
  %in.addr = alloca ptr addrspace(1), align 8
  %out.addr = alloca ptr addrspace(1), align 8
  %n = alloca i32, align 4
  store ptr addrspace(1) %in, ptr %in.addr, align 8
  store ptr addrspace(1) %out, ptr %out.addr, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #2
  %conv = trunc i64 %call to i32
  store i32 %conv, ptr %n, align 4
  %0 = load i32, ptr %n, align 4
  %idxprom = sext i32 %0 to i64
  %1 = load ptr addrspace(1), ptr %in.addr, align 8
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %idxprom
  %2 = load i32, ptr addrspace(1) %arrayidx, align 4
  %call1 = call spir_func i32 @_Z3absi(i32 %2) #2
  %3 = load i32, ptr %n, align 4
  %idxprom2 = sext i32 %3 to i64
  %4 = load ptr addrspace(1), ptr %out.addr, align 8
  %arrayidx3 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %idxprom2
  store i32 %call1, ptr addrspace(1) %arrayidx3, align 4
  ret void
}

; CHECK-NOT: "_Z13get_global_idj"
; Function Attrs: nounwind readnone
declare spir_func i64 @_Z13get_global_idj(i32) #1

; CHECK-NOT: "_Z3absi"
; Function Attrs: nounwind readnone
declare spir_func i32 @_Z3absi(i32) #1

; CHECK: "myabs"
; Function Attrs: nounwind
define spir_func i32 @myabs(i32 %x) #0 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %call = call spir_func i32 @_Z3absi(i32 %0) #2
  ret i32 %call
}

; CHECK: "test_function_call"
; Function Attrs: nounwind
define spir_func void @test_function_call(ptr addrspace(1) %in, ptr addrspace(1) %out) #0 {
entry:
  %in.addr = alloca ptr addrspace(1), align 8
  %out.addr = alloca ptr addrspace(1), align 8
  %n = alloca i32, align 4
  store ptr addrspace(1) %in, ptr %in.addr, align 8
  store ptr addrspace(1) %out, ptr %out.addr, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #2
  %conv = trunc i64 %call to i32
  store i32 %conv, ptr %n, align 4
  %0 = load i32, ptr %n, align 4
  %idxprom = sext i32 %0 to i64
  %1 = load ptr addrspace(1), ptr %in.addr, align 8
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %idxprom
  %2 = load i32, ptr addrspace(1) %arrayidx, align 4
  %call1 = call spir_func i32 @myabs(i32 %2)
  %3 = load i32, ptr %n, align 4
  %idxprom2 = sext i32 %3 to i64
  %4 = load ptr addrspace(1), ptr %out.addr, align 8
  %arrayidx3 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %idxprom2
  store i32 %call1, ptr addrspace(1) %arrayidx3, align 4
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!12}
!opencl.ocl.version = !{!12}
!opencl.used.extensions = !{!13}
!opencl.used.optional.core.features = !{!14}
!opencl.compiler.options = !{!13}

!1 = !{i32 1, i32 1, i32 0}
!2 = !{!"none", !"none", !"none"}
!3 = !{!"int*", !"int*", !"int"}
!4 = !{!"int*", !"int*", !"int"}
!5 = !{!"restrict", !"const", !""}
!6 = !{i32 1, i32 2, i32 3}
!8 = !{!"double*", !"double*", !"int"}
!9 = !{!"double*", !"double*", !"int"}
!10 = !{!"", !"", !""}
!11 = !{double undef, i32 1}
!12 = !{i32 1, i32 2}
!13 = !{}
!14 = !{!"cl_doubles"}
