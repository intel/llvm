; RUN: sycl-post-link --ir-output-only -spec-const=default %s -S -o - | \
; RUN:   FileCheck %s -check-prefixes=CHECK,CHECK-DEF

; This test checks that the post link tool is able to correctly transform
; specialization constant intrinsics for different types in a device code
; compiled for AOT.

%class.specialization_id = type { double }
%class.specialization_id.0 = type { i32 }
%class.specialization_id.1 = type { %struct.ComposConst }
%struct.ComposConst = type { i32, double, %struct.myConst }
%struct.myConst = type { i32, float }
%class.specialization_id.2 = type { %struct.ComposConst2 }
%struct.ComposConst2 = type { i8, %struct.myConst, double }

@id_double = dso_local global %class.specialization_id { double 3.140000e+00 }, align 8
@id_int = dso_local global %class.specialization_id.0 { i32 42 }, align 4
@id_compos = dso_local global %class.specialization_id.1 { %struct.ComposConst { i32 1, double 2.000000e+00, %struct.myConst { i32 13, float 0x4020666660000000 } } }, align 8
@id_compos2 = dso_local global %class.specialization_id.2 { %struct.ComposConst2 { i8 1, %struct.myConst { i32 52, float 0x40479999A0000000 }, double 2.000000e+00 } }, align 8

; check that no longer used globals are removed:
@__builtin_unique_stable_name._Z27get_specialization_constantIL_Z9id_doubleE17specialization_idIdEdET1_v = private unnamed_addr constant [37 x i8] c"_ZTS14name_generatorIL_Z9id_doubleEE\00", align 1
; CHECK-NOT: @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z9id_doubleE17specialization_idIdEdET1_v
@__builtin_unique_stable_name._Z27get_specialization_constantIL_Z6id_intE17specialization_idIiEiET1_v = private unnamed_addr constant [34 x i8] c"_ZTS14name_generatorIL_Z6id_intEE\00", align 1
; CHECK-NOT: @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z6id_intE17specialization_idIiEiET1_v
@__builtin_unique_stable_name._Z27get_specialization_constantIL_Z9id_composE17specialization_idI11ComposConstES1_ET1_v = private unnamed_addr constant [37 x i8] c"_ZTS14name_generatorIL_Z9id_composEE\00", align 1
; CHECK-NOT: @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z9id_composE17specialization_idI11ComposConstES1_ET1_v
@__builtin_unique_stable_name._Z27get_specialization_constantIL_Z10id_compos2E17specialization_idI12ComposConst2ES1_ET1_v = private unnamed_addr constant [39 x i8] c"_ZTS14name_generatorIL_Z10id_compos2EE\00", align 1
; CHECK-NOT: @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z10id_compos2E17specialization_idI12ComposConst2ES1_ET1_v

define dso_local void @_Z4testv() local_unnamed_addr #0 {
entry:
  %call.i = tail call fast double @_Z37__sycl_getScalar2020SpecConstantValueIdET_PKcPvS3_(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z9id_doubleE17specialization_idIdEdET1_v, i64 0, i64 0), i8* bitcast (%class.specialization_id* @id_double to i8*), i8* null)
; CHECK-DEF: %[[GEP:[0-9a-z]+]] = getelementptr i8, i8* null, i32 0
; CHECK-DEF: %[[BITCAST:[0-9a-z]+]] = bitcast i8* %[[GEP]] to double*
; CHECK-DEF: %[[LOAD:[0-9a-z]+]] = load double, double* %[[BITCAST]], align 8
  %call.i3 = tail call i32 @_Z37__sycl_getScalar2020SpecConstantValueIiET_PKcPvS3_(i8* getelementptr inbounds ([34 x i8], [34 x i8]* @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z6id_intE17specialization_idIiEiET1_v, i64 0, i64 0), i8* bitcast (%class.specialization_id.0* @id_int to i8*), i8* null)
; CHECK-DEF: %[[GEP1:[0-9a-z]+]] = getelementptr i8, i8* null, i32 8
; CHECK-DEF: %[[BITCAST1:[0-9a-z]+]] = bitcast i8* %[[GEP1]] to i32*
; CHECK-DEF: %[[LOAD1:[0-9a-z]+]] = load i32, i32* %[[BITCAST1]], align 4
  %call.i4 = tail call fast double @_Z37__sycl_getScalar2020SpecConstantValueIdET_PKcPvS3_(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z9id_doubleE17specialization_idIdEdET1_v, i64 0, i64 0), i8* bitcast (%class.specialization_id* @id_double to i8*), i8* null)
; CHECK-DEF: %[[GEP2:[0-9a-z]+]] = getelementptr i8, i8* null, i32 0
; CHECK-DEF: %[[BITCAST2:[0-9a-z]+]] = bitcast i8* %[[GEP2]] to double*
; CHECK-DEF: %[[LOAD2:[0-9a-z]+]] = load double, double* %[[BITCAST2]], align 8
  ret void
}

define dso_local void @_Z5test2v() local_unnamed_addr #0 {
entry:
  %tmp = alloca %struct.ComposConst, align 8
  %tmp1 = alloca %struct.ComposConst2, align 8
  %tmp2 = alloca %struct.ComposConst, align 8
  %0 = bitcast %struct.ComposConst* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0) #3
  call void @_Z40__sycl_getComposite2020SpecConstantValueI11ComposConstET_PKcPvS4_(%struct.ComposConst* nonnull sret(%struct.ComposConst) align 8 %tmp, i8* getelementptr inbounds ([37 x i8], [37 x i8]* @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z9id_composE17specialization_idI11ComposConstES1_ET1_v, i64 0, i64 0), i8* bitcast (%class.specialization_id.1* @id_compos to i8*), i8* null)
; CHECK: %[[GEP:[0-9a-z]+]] = getelementptr i8, i8* null, i32 12
; CHECK: %[[BITCAST:[0-9a-z]+]] = bitcast i8* %[[GEP]] to %struct.ComposConst*
; CHECK: %[[LOAD:[0-9a-z]+]] = load %struct.ComposConst, %struct.ComposConst* %[[BITCAST]], align 8
; CHECK: store %struct.ComposConst %[[LOAD]], %struct.ComposConst*
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0) #3
  %1 = getelementptr inbounds %struct.ComposConst2, %struct.ComposConst2* %tmp1, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %1) #3
  call void @_Z40__sycl_getComposite2020SpecConstantValueI12ComposConst2ET_PKcPvS4_(%struct.ComposConst2* nonnull sret(%struct.ComposConst2) align 8 %tmp1, i8* getelementptr inbounds ([39 x i8], [39 x i8]* @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z10id_compos2E17specialization_idI12ComposConst2ES1_ET1_v, i64 0, i64 0), i8* getelementptr inbounds (%class.specialization_id.2, %class.specialization_id.2* @id_compos2, i64 0, i32 0, i32 0), i8* null)  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %1) #3
; CHECK: %[[GEP1:[0-9a-z]+]] = getelementptr i8, i8* null, i32 36
; CHECK: %[[BITCAST1:[0-9a-z]+]] = bitcast i8* %[[GEP1]] to %struct.ComposConst2*
; CHECK: %[[LOAD1:[0-9a-z]+]] = load %struct.ComposConst2, %struct.ComposConst2* %[[BITCAST1]], align 8
; CHECK: store %struct.ComposConst2 %[[LOAD1]], %struct.ComposConst2*
  %2 = bitcast %struct.ComposConst* %tmp2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %2) #3
  call void @_Z40__sycl_getComposite2020SpecConstantValueI11ComposConstET_PKcPvS4_(%struct.ComposConst* nonnull sret(%struct.ComposConst) align 8 %tmp2, i8* getelementptr inbounds ([37 x i8], [37 x i8]* @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z9id_composE17specialization_idI11ComposConstES1_ET1_v, i64 0, i64 0), i8* bitcast (%class.specialization_id.1* @id_compos to i8*), i8* null)
; CHECK: %[[GEP2:[0-9a-z]+]] = getelementptr i8, i8* null, i32 12
; CHECK: %[[BITCAST2:[0-9a-z]+]] = bitcast i8* %[[GEP2]] to %struct.ComposConst*
; CHECK: %[[LOAD2:[0-9a-z]+]] = load %struct.ComposConst, %struct.ComposConst* %[[BITCAST2]], align 8
; CHECK: store %struct.ComposConst %[[LOAD2]], %struct.ComposConst*
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %2) #3
  ret void
}
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

declare dso_local double @_Z37__sycl_getScalar2020SpecConstantValueIdET_PKcPvS3_(i8*, i8*, i8*) local_unnamed_addr #1

declare dso_local i32 @_Z37__sycl_getScalar2020SpecConstantValueIiET_PKcPvS3_(i8*, i8*, i8*) local_unnamed_addr #1

declare dso_local void @_Z40__sycl_getComposite2020SpecConstantValueI11ComposConstET_PKcPvS4_(%struct.ComposConst* sret(%struct.ComposConst) align 8, i8*, i8*, i8*) local_unnamed_addr #2

declare dso_local void @_Z40__sycl_getComposite2020SpecConstantValueI12ComposConst2ET_PKcPvS4_(%struct.ComposConst2* sret(%struct.ComposConst2) align 8, i8*, i8*, i8*) local_unnamed_addr #2

attributes #0 = { uwtable mustprogress "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind }

; CHECK: !sycl.specialization-constants = !{![[#ID0:]], ![[#ID1:]], ![[#ID2:]], ![[#ID3:]]}
;
; CHECK: ![[#ID0]] = !{!"_ZTS14name_generatorIL_Z6id_intEE", i32 1, i32 0, i32 4}
; CHECK: ![[#ID1]] = !{!"_ZTS14name_generatorIL_Z9id_doubleEE", i32 0, i32 0, i32 8}
; CHECK: ![[#ID2]] = !{!"_ZTS14name_generatorIL_Z10id_compos2EE", i32 3, i32 0, i32 24
; CHECK: ![[#ID3]] = !{!"_ZTS14name_generatorIL_Z9id_composEE", i32 2, i32 0, i32 24}
