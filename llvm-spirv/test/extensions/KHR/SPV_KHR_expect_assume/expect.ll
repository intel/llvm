; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_expect_assume -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV-NO-EXT
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM-NO-EXT

; CHECK-SPIRV: Capability ExpectAssumeKHR
; CHECK-SPIRV: Extension "SPV_KHR_expect_assume"
; CHECK-SPIRV: Name [[FUNPARAM:[0-9]+]] "x.addr"
; CHECK-SPIRV: Name [[VALUE1:[0-9]+]] "conv"
; CHECK-SPIRV: Name [[VALUE2:[0-9]+]] "conv"
; CHECK-SPIRV: TypeInt [[TYPEID:[0-9]+]] 64 0
; CHECK-SPIRV: Constant [[TYPEID]] [[EXPVAL1:[0-9]+]] {{[0-9]+}} {{[0-9]+}}

; CHECK-SPIRV: Function
; CHECK-SPIRV: ExpectKHR [[TYPEID]] [[RES1:[0-9]+]] [[VALUE1]] [[EXPVAL1]]
; CHECK-SPIRV: INotEqual {{[0-9]+}} {{[0-9]+}} [[RES1]] {{[0-9]+}}

; CHECK-SPIRV: Function
; CHECK-SPIRV: FunctionCall {{[0-9]+}} [[FUNRES:[0-9]+]] {{[0-9]+}}
; CHECK-SPIRV: SConvert [[TYPEID]] [[EXPVAL2:[0-9]+]] [[FUNRES]]
; CHECK-SPIRV: ExpectKHR {{[0-9]+}} [[RES2:[0-9]+]] [[VALUE2]] [[EXPVAL2]]
; CHECK-SPIRV: INotEqual {{[0-9]+}} {{[0-9]+}} [[RES2]] {{[0-9]+}}

; CHECK-LLVM: define spir_func i32 @_Z12expect_consti{{.*}}
; CHECK-LLVM: %[[EXP1:[0-9]+]] = load i32, ptr {{.*}}, align 4
; CHECK-LLVM: %[[CONV1:[a-z0-9]+]] = sext i32 %[[EXP1]] to i64
; CHECK-LLVM: %[[RES1:[a-z0-9]+]] = call i64 @llvm.expect.i64(i64 %[[CONV1]], i64 1)
; CHECK-LLVM: %{{.*}} = icmp ne i64 %[[RES1]], 0

; CHECK-LLVM: define spir_func i32 @_Z10expect_funi{{.*}}
; CHECK-LLVM: %[[EXP2:[0-9]+]] = load i32, ptr {{.*}}, align 4
; CHECK-LLVM: %[[CONV2A:[a-z0-9]+]] = sext i32 %[[EXP2]] to i64
; CHECK-LLVM: %[[CALL:[a-z0-9]+]] = call spir_func i32 @_Z3foov()
; CHECK-LLVM: %[[CONV2B:[a-z0-9]+]] = sext i32 %[[CALL]] to i64
; CHECK-LLVM: %[[RES2:[a-z0-9]+]] = call i64 @llvm.expect.i64(i64 %[[CONV2A]], i64 %[[CONV2B]])
; CHECK-LLVM: %{{.*}} = icmp ne i64 %[[RES2]], 0

; CHECK-SPIRV-NO-EXT-NOT: Capability ExpectAssumeKHR
; CHECK-SPIRV-NO-EXT-NOT: Extension "SPV_KHR_expect_assume"
; CHECK-SPIRV-NO-EXT: Function
; CHECK-SPIRV-NO-EXT-NOT: ExpectKHR {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV-NO-EXT: SConvert {{[0-9]+}} [[RES1:[0-9]+]] {{[0-9]+}}
; CHECK-SPIRV-NO-EXT: INotEqual {{[0-9]+}} {{[0-9]+}} [[RES1]] {{[0-9]+}}

; CHECK-SPIRV-NO-EXT: Function
; CHECK-SPIRV-NO-EXT: SConvert {{[0-9]+}} [[RES2:[0-9]+]] {{[0-9]+}}
; CHECK-SPIRV-NO-EXT-NOT: ExpectKHR {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
; CHECK-SPIRV-NO-EXT: INotEqual {{[0-9]+}} {{[0-9]+}} [[RES2]] {{[0-9]+}}

; CHECK-LLVM-NO-EXT: define spir_func i32 @_Z12expect_consti{{.*}}
; CHECK-LLVM-NO-EXT: %[[EXP1:[0-9]+]] = load i32, ptr {{.*}}, align 4
; CHECK-LLVM-NO-EXT: %[[CONV1:[a-z0-9]+]] = sext i32 %[[EXP1]] to i64
; CHECK-LLVM-NO-EXT-NOT: %{{.*}} = call i64 @llvm.expect.i64(i64 %{{.*}}, i64 1)
; CHECK-LLVM-NO-EXT: %{{.*}} = icmp ne i64 %[[CONV1]], 0

; CHECK-LLVM-NO-EXT: define spir_func i32 @_Z10expect_funi{{.*}}
; CHECK-LLVM-NO-EXT: %[[EXP2:[0-9]+]] = load i32, ptr {{.*}}, align 4
; CHECK-LLVM-NO-EXT: %[[CONV2A:[a-z0-9]+]] = sext i32 %[[EXP2]] to i64
; CHECK-LLVM-NO-EXT: %[[CALL:[a-z0-9]+]] = call spir_func i32 @_Z3foov()
; CHECK-LLVM-NO-EXT: %[[CONV2B:[a-z0-9]+]] = sext i32 %[[CALL]] to i64
; CHECK-LLVM-NO-EXT-NOT: %{{.*}} = call i64 @llvm.expect.i64(i64 %[[CONV2A]], i64 %[[CONV2B]])
; CHECK-LLVM-NO-EXT: %{{.*}} = icmp ne i64 %[[CONV2A]], 0

; ModuleID = 'expect.cpp'
source_filename = "expect.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

%"class._ZTSZ4mainE3$_0.anon" = type { i8 }

; Function Attrs: norecurse nounwind
define spir_kernel void @_ZTSZ4mainE15kernel_function() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  %0 = alloca %"class._ZTSZ4mainE3$_0.anon", align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr %0) #6
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  call spir_func void @"_ZZ4mainENK3$_0clEv"(ptr addrspace(4) %1)
  call void @llvm.lifetime.end.p0(i64 1, ptr %0) #6
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: inlinehint norecurse nounwind
define internal spir_func void @"_ZZ4mainENK3$_0clEv"(ptr addrspace(4) %this) #2 align 2 {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  store ptr addrspace(4) %this, ptr %this.addr, align 8, !tbaa !5
  %this1 = load ptr addrspace(4), ptr %this.addr, align 8
  call void @llvm.lifetime.start.p0(i64 4, ptr %a) #6
  %call = call spir_func i32 @_Z12expect_consti(i32 1)
  store i32 %call, ptr %a, align 4, !tbaa !9
  call void @llvm.lifetime.start.p0(i64 4, ptr %b) #6
  %call2 = call spir_func i32 @_Z10expect_funi(i32 2)
  store i32 %call2, ptr %b, align 4, !tbaa !9
  call void @llvm.lifetime.end.p0(i64 4, ptr %b) #6
  call void @llvm.lifetime.end.p0(i64 4, ptr %a) #6
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: norecurse nounwind
define spir_func i32 @_Z12expect_consti(i32 %x) #3 {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4, !tbaa !9
  %0 = load i32, ptr %x.addr, align 4, !tbaa !9
  %conv = sext i32 %0 to i64
  %expval = call i64 @llvm.expect.i64(i64 %conv, i64 1)
  %tobool = icmp ne i64 %expval, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 0, ptr %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
  %1 = load i32, ptr %x.addr, align 4, !tbaa !9
  store i32 %1, ptr %retval, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %2 = load i32, ptr %retval, align 4
  ret i32 %2
}

; Function Attrs: norecurse nounwind
define spir_func i32 @_Z10expect_funi(i32 %x) #3 {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4, !tbaa !9
  %0 = load i32, ptr %x.addr, align 4, !tbaa !9
  %conv = sext i32 %0 to i64
  %call = call spir_func i32 @_Z3foov()
  %conv1 = sext i32 %call to i64
  %expval = call i64 @llvm.expect.i64(i64 %conv, i64 %conv1)
  %tobool = icmp ne i64 %expval, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 0, ptr %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
  %1 = load i32, ptr %x.addr, align 4, !tbaa !9
  store i32 %1, ptr %retval, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %2 = load i32, ptr %retval, align 4
  ret i32 %2
}

; Function Attrs: nounwind readnone willreturn
declare i64 @llvm.expect.i64(i64, i64) #4

declare spir_func i32 @_Z3foov() #5

attributes #0 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "sycl-module-id"="expect.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { inlinehint norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readnone willreturn }
attributes #5 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 11.0.0"}
!4 = !{}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !7, i64 0}
