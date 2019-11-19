; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_memory_attributes -o %t.spv
; RUN: llvm-spirv %t.spv --spirv-ext=+SPV_INTEL_fpga_memory_attributes -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv -spirv-text -r %t.spt -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
;
; TODO: add a bunch of different tests for --spirv-ext option

; CHECK-SPIRV: Capability FPGAMemoryAttributesINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_fpga_memory_attributes"
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 1 RegisterINTEL
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 0 MemoryINTEL "DEFAULT"
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 3 MemoryINTEL "DEFAULT"
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 2 MemoryINTEL "MLAB"
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 0 NumbanksINTEL 2
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 0 NumbanksINTEL 4
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 3 BankwidthINTEL 8
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 4 MaxPrivateCopiesINTEL 4
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 5 SinglepumpINTEL
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 6 DoublepumpINTEL
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 8 MaxReplicatesINTEL 4
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 9 SimpleDualPortINTEL
; CHECK-SPIRV: MemberDecorate {{[0-9]+}} 7 MergeINTEL "foobar" "width"

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

%class.anon = type { i8 }
%struct.foo = type { i32, i32, i32, i32, i8, i32, i32, i32, i32, i32 }

%struct._ZTSZ20field_addrspace_castvE5state.state = type { [8 x i32] }

; CHECK-LLVM: [[STR1:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}{numbanks:4}
; CHECK-LLVM: [[STR2:@[0-9_.]+]] = {{.*}}{register:1}
; CHECK-LLVM: [[STR3:@[0-9_.]+]] = {{.*}}{memory:MLAB}
; CHECK-LLVM: [[STR4:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}{bankwidth:8}
; CHECK-LLVM: [[STR5:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}{max_private_copies:4}
; CHECK-LLVM: [[STR6:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}{pump:1}
; CHECK-LLVM: [[STR7:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}{pump:2}
; CHECK-LLVM: [[STR8:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}{merge:foobar:width}
; CHECK-LLVM: [[STR9:@[0-9_.]+]] = {{.*}}{max_replicates:4}
; CHECK-LLVM: [[STR10:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}{simple_dual_port:1}
; CHECK-LLVM: [[STR11:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}{numbanks:2}
@.str = private unnamed_addr constant [29 x i8] c"{memory:DEFAULT}{numbanks:4}\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [16 x i8] c"test_struct.cpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr constant [13 x i8] c"{register:1}\00", section "llvm.metadata"
@.str.3 = private unnamed_addr constant [14 x i8] c"{memory:MLAB}\00", section "llvm.metadata"
@.str.4 = private unnamed_addr constant [30 x i8] c"{memory:DEFAULT}{bankwidth:8}\00", section "llvm.metadata"
@.str.5 = private unnamed_addr constant [39 x i8] c"{memory:DEFAULT}{max_private_copies:4}\00", section "llvm.metadata"
@.str.6 = private unnamed_addr constant [25 x i8] c"{memory:DEFAULT}{pump:1}\00", section "llvm.metadata"
@.str.7 = private unnamed_addr constant [25 x i8] c"{memory:DEFAULT}{pump:2}\00", section "llvm.metadata"
@.str.8 = private unnamed_addr constant [37 x i8] c"{memory:DEFAULT}{merge:foobar:width}\00", section "llvm.metadata"
@.str.9 = private unnamed_addr constant [19 x i8] c"{max_replicates:4}\00", section "llvm.metadata"
@.str.10 = private unnamed_addr constant [37 x i8] c"{memory:DEFAULT}{simple_dual_port:1}\00", section "llvm.metadata"
@.str.11 = private unnamed_addr constant [29 x i8] c"{memory:DEFAULT}{numbanks:2}\00", section "llvm.metadata"

; Function Attrs: nounwind
define spir_kernel void @_ZTSZ4mainE15kernel_function() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  %0 = alloca %class.anon, align 1
  %1 = bitcast %class.anon* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %1) #4
  call spir_func void @"_ZZ4mainENK3$_0clEv"(%class.anon* %0)
  %2 = bitcast %class.anon* %0 to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %2) #4
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: inlinehint nounwind
define internal spir_func void @"_ZZ4mainENK3$_0clEv"(%class.anon* %this) #2 align 2 {
entry:
  %this.addr = alloca %class.anon*, align 8
  store %class.anon* %this, %class.anon** %this.addr, align 8, !tbaa !5
  %this1 = load %class.anon*, %class.anon** %this.addr, align 8
  call spir_func void @_Z3barv()
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind
define spir_func void @_Z3barv() #3 {
entry:
  %s1 = alloca %struct.foo, align 4
  %0 = bitcast %struct.foo* %s1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 20, i8* %0) #4
  ; CHECK-LLVM: %[[FIELD1:.*]] = getelementptr inbounds %struct.foo, %struct.foo* %{{[a-zA-Z0-9]+}}, i32 0, i32 0
  ; CHECK-LLVM: %[[CAST1:.*]] = bitcast{{.*}}%[[FIELD1]]
  ; CHECK-LLVM: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST1]]{{.*}}[[STR1]]
  %f1 = getelementptr inbounds %struct.foo, %struct.foo* %s1, i32 0, i32 0
  %1 = bitcast i32* %f1 to i8*
  %2 = call i8* @llvm.ptr.annotation.p0i8(i8* %1, i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i32 0, i32 0), i32 2)
  %3 = bitcast i8* %2 to i32*
  store i32 0, i32* %3, align 4, !tbaa !9
  ; CHECK-LLVM: %[[FIELD2:.*]] = getelementptr inbounds %struct.foo, %struct.foo* %{{[a-zA-Z0-9]+}}, i32 0, i32 1
  ; CHECK-LLVM: %[[CAST2:.*]] = bitcast{{.*}}%[[FIELD2]]
  ; CHECK-LLVM: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST2]]{{.*}}[[STR2]]
  %f2 = getelementptr inbounds %struct.foo, %struct.foo* %s1, i32 0, i32 1
  %4 = bitcast i32* %f2 to i8*
  %5 = call i8* @llvm.ptr.annotation.p0i8(i8* %4, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.2, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i32 0, i32 0), i32 3)
  %6 = bitcast i8* %5 to i32*
  store i32 0, i32* %6, align 4, !tbaa !12
  ; CHECK-LLVM: %[[FIELD3:.*]] = getelementptr inbounds %struct.foo, %struct.foo* %{{[a-zA-Z0-9]+}}, i32 0, i32 2
  ; CHECK-LLVM: %[[CAST3:.*]] = bitcast{{.*}}%[[FIELD3]]
  ; CHECK-LLVM: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST3]]{{.*}}[[STR3]]
  %f3 = getelementptr inbounds %struct.foo, %struct.foo* %s1, i32 0, i32 2
  %7 = bitcast i32* %f3 to i8*
  %8 = call i8* @llvm.ptr.annotation.p0i8(i8* %7, i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.3, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i32 0, i32 0), i32 4)
  %9 = bitcast i8* %8 to i32*
  store i32 0, i32* %9, align 4, !tbaa !13
  ; CHECK-LLVM: %[[FIELD4:.*]] = getelementptr inbounds %struct.foo, %struct.foo* %{{[a-zA-Z0-9]+}}, i32 0, i32 3
  ; CHECK-LLVM: %[[CAST4:.*]] = bitcast{{.*}}%[[FIELD4]]
  ; CHECK-LLVM: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST4]]{{.*}}[[STR4]]
  %f4 = getelementptr inbounds %struct.foo, %struct.foo* %s1, i32 0, i32 3
  %10 = bitcast i32* %f4 to i8*
  %11 = call i8* @llvm.ptr.annotation.p0i8(i8* %10, i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.4, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i32 0, i32 0), i32 5)
  %12 = bitcast i8* %11 to i32*
  store i32 0, i32* %12, align 4, !tbaa !14
  ; CHECK-LLVM: %[[FIELD5:.*]] = getelementptr inbounds %struct.foo, %struct.foo* %{{[a-zA-Z0-9]+}}, i32 0, i32 4
  ; CHECK-LLVM: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[FIELD5]]{{.*}}[[STR5]]
  %f5 = getelementptr inbounds %struct.foo, %struct.foo* %s1, i32 0, i32 4
  %13 = call i8* @llvm.ptr.annotation.p0i8(i8* %f5, i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.5, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i32 0, i32 0), i32 6)
  store i8 0, i8* %13, align 4, !tbaa !15
  ; CHECK-LLVM: %[[FIELD6:.*]] = getelementptr inbounds %struct.foo, %struct.foo* %{{[a-zA-Z0-9]+}}, i32 0, i32 5
  ; CHECK-LLVM: %[[CAST6:.*]] = bitcast{{.*}}%[[FIELD6]]
  ; CHECK-LLVM: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST6]]{{.*}}[[STR6]]
  %f6 = getelementptr inbounds %struct.foo, %struct.foo* %s1, i32 0, i32 5
  %14 = bitcast i32* %f6 to i8*
  %15 = call i8* @llvm.ptr.annotation.p0i8(i8* %14, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.6, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i32 0, i32 0), i32 7)
  %16 = bitcast i8* %15 to i32*
  store i32 0, i32* %16, align 4, !tbaa !16
  ; CHECK-LLVM: %[[FIELD7:.*]] = getelementptr inbounds %struct.foo, %struct.foo* %{{[a-zA-Z0-9]+}}, i32 0, i32 6
  ; CHECK-LLVM: %[[CAST7:.*]] = bitcast{{.*}}%[[FIELD7]]
  ; CHECK-LLVM: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST7]]{{.*}}[[STR7]]
  %f7 = getelementptr inbounds %struct.foo, %struct.foo* %s1, i32 0, i32 6
  %17 = bitcast i32* %f7 to i8*
  %18 = call i8* @llvm.ptr.annotation.p0i8(i8* %17, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.7, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i32 0, i32 0), i32 8)
  %19 = bitcast i8* %18 to i32*
  store i32 0, i32* %19, align 4, !tbaa !17
  ; CHECK-LLVM: %[[FIELD8:.*]] = getelementptr inbounds %struct.foo, %struct.foo* %{{[a-zA-Z0-9]+}}, i32 0, i32 7
  ; CHECK-LLVM: %[[CAST8:.*]] = bitcast{{.*}}%[[FIELD8]]
  ; CHECK-LLVM: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST8]]{{.*}}[[STR8]]
  %f8 = getelementptr inbounds %struct.foo, %struct.foo* %s1, i32 0, i32 7
  %20 = bitcast i32* %f8 to i8*
  %21 = call i8* @llvm.ptr.annotation.p0i8(i8* %20, i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.8, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i32 0, i32 0), i32 9)
  %22 = bitcast i8* %21 to i32*
  store i32 0, i32* %22, align 4, !tbaa !18
  ; CHECK-LLVM: %[[FIELD9:.*]] = getelementptr inbounds %struct.foo, %struct.foo* %{{[a-zA-Z0-9]+}}, i32 0, i32 8
  ; CHECK-LLVM: %[[CAST9:.*]] = bitcast{{.*}}%[[FIELD9]]
  ; CHECK-LLVM: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST9]]{{.*}}[[STR9]]
  %f9 = getelementptr inbounds %struct.foo, %struct.foo* %s1, i32 0, i32 8
  %23 = bitcast i32* %f9 to i8*
  %24 = call i8* @llvm.ptr.annotation.p0i8(i8* %23, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.9, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i32 0, i32 0), i32 10)
  %25 = bitcast i8* %24 to i32*
  store i32 0, i32* %25, align 4, !tbaa !19
  ; CHECK-LLVM: %[[FIELD10:.*]] = getelementptr inbounds %struct.foo, %struct.foo* %{{[a-zA-Z0-9]+}}, i32 0, i32 9
  ; CHECK-LLVM: %[[CAST10:.*]] = bitcast{{.*}}%[[FIELD10]]
  ; CHECK-LLVM: call i8* @llvm.ptr.annotation.p0i8{{.*}}%[[CAST10]]{{.*}}[[STR10]]
  %f10 = getelementptr inbounds %struct.foo, %struct.foo* %s1, i32 0, i32 9
  %26 = bitcast i32* %f10 to i8*
  %27 = call i8* @llvm.ptr.annotation.p0i8(i8* %26, i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.10, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i32 0, i32 0), i32 11)
  %28 = bitcast i8* %27 to i32*
  store i32 0, i32* %28, align 4, !tbaa !20
  %29 = bitcast %struct.foo* %s1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 40, i8* %29) #4
  ret void
}

define spir_func void @_Z20field_addrspace_castv() #3 {
entry:
  %state_var = alloca %struct._ZTSZ20field_addrspace_castvE5state.state, align 4
  %0 = bitcast %struct._ZTSZ20field_addrspace_castvE5state.state* %state_var to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* %0) #4
  %1 = addrspacecast %struct._ZTSZ20field_addrspace_castvE5state.state* %state_var to %struct._ZTSZ20field_addrspace_castvE5state.state addrspace(4)*
  call spir_func void @_ZZ20field_addrspace_castvEN5stateC2Ev(%struct._ZTSZ20field_addrspace_castvE5state.state addrspace(4)* %1)
  %mem = getelementptr inbounds %struct._ZTSZ20field_addrspace_castvE5state.state, %struct._ZTSZ20field_addrspace_castvE5state.state* %state_var, i32 0, i32 0
  ; CHECK-LLVM: %[[GEP:.*]] = getelementptr inbounds %struct._ZTSZ20field_addrspace_castvE5state.state, %struct._ZTSZ20field_addrspace_castvE5state.state* %state_var, i32 0, i32 0
  ; CHECK-LLVM: %[[CAST11:.*]] = bitcast [8 x i32]* %[[GEP:.*]] to i8*
  ; CHECK-LLVM: %{{[0-9]+}} = call i8* @llvm.ptr.annotation.p0i8(i8* %[[CAST11]]{{.*}}[[STR11]]
  %2 = bitcast [8 x i32]* %mem to i8*
  %3 = call i8* @llvm.ptr.annotation.p0i8(i8* %2, i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.11, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i32 0, i32 0), i32 24)
  %4 = bitcast i8* %3 to [8 x i32]*
  %arrayidx = getelementptr inbounds [8 x i32], [8 x i32]* %4, i64 0, i64 0
  store i32 42, i32* %arrayidx, align 4, !tbaa !9
  %5 = bitcast %struct._ZTSZ20field_addrspace_castvE5state.state* %state_var to i8*
  call void @llvm.lifetime.end.p0i8(i64 32, i8* %5) #4
  ret void
}

define internal spir_func void @_ZZ20field_addrspace_castvEN5stateC2Ev(%struct._ZTSZ20field_addrspace_castvE5state.state addrspace(4)* %this) unnamed_addr #3 align 2 {
entry:
  %this.addr = alloca %struct._ZTSZ20field_addrspace_castvE5state.state addrspace(4)*, align 8
  %i = alloca i32, align 4
  store %struct._ZTSZ20field_addrspace_castvE5state.state addrspace(4)* %this, %struct._ZTSZ20field_addrspace_castvE5state.state addrspace(4)** %this.addr, align 8, !tbaa !5
  %this1 = load %struct._ZTSZ20field_addrspace_castvE5state.state addrspace(4)*, %struct._ZTSZ20field_addrspace_castvE5state.state addrspace(4)** %this.addr, align 8
  %0 = bitcast i32* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #4
  store i32 0, i32* %i, align 4, !tbaa !9
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load i32, i32* %i, align 4, !tbaa !9
  %cmp = icmp slt i32 %1, 8
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %2 = bitcast i32* %i to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %2) #4
  br label %for.end

for.body:                                         ; preds = %for.cond
  %3 = load i32, i32* %i, align 4, !tbaa !9
  %mem = getelementptr inbounds %struct._ZTSZ20field_addrspace_castvE5state.state, %struct._ZTSZ20field_addrspace_castvE5state.state addrspace(4)* %this1, i32 0, i32 0
  ; FIXME: currently llvm.ptr.annotation is not emitted for c'tors, need to fix it and add a check here
  %4 = bitcast [8 x i32] addrspace(4)* %mem to i8 addrspace(4)*
  %5 = call i8 addrspace(4)* @llvm.ptr.annotation.p4i8(i8 addrspace(4)* %4, i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.11, i32 0, i32 0), i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i32 0, i32 0), i32 24)
  %6 = bitcast i8 addrspace(4)* %5 to [8 x i32] addrspace(4)*
  %7 = load i32, i32* %i, align 4, !tbaa !9
  %idxprom = sext i32 %7 to i64
  %arrayidx = getelementptr inbounds [8 x i32], [8 x i32] addrspace(4)* %6, i64 0, i64 %idxprom
  store i32 %3, i32 addrspace(4)* %arrayidx, align 4, !tbaa !9
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %8 = load i32, i32* %i, align 4, !tbaa !9
  %inc = add nsw i32 %8, 1
  store i32 %inc, i32* %i, align 4, !tbaa !9
  br label %for.cond

for.end:                                          ; preds = %for.cond.cleanup
  ret void
}

; Function Attrs: nounwind
declare i8* @llvm.ptr.annotation.p0i8(i8*, i8*, i8*, i32) #4

; Function Attrs: nounwind
declare i8 addrspace(4)* @llvm.ptr.annotation.p4i8(i8 addrspace(4)*, i8*, i8*, i32) #4

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind optnone noinline "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 9.0.0"}
!4 = !{}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !11, i64 0}
!10 = !{!"_ZTS3foo", !11, i64 0, !11, i64 4, !11, i64 8, !11, i64 12, !11, i64 16, !11, i64 20, !11, i64 24, !11, i64 28, !11, i64 32, !11, i64 36}
!11 = !{!"int", !7, i64 0}
!12 = !{!10, !11, i64 4}
!13 = !{!10, !11, i64 8}
!14 = !{!10, !11, i64 12}
!15 = !{!10, !11, i64 16}
!16 = !{!10, !11, i64 20}
!17 = !{!10, !11, i64 24}
!18 = !{!10, !11, i64 28}
!19 = !{!10, !11, i64 32}
!20 = !{!10, !11, i64 36}

