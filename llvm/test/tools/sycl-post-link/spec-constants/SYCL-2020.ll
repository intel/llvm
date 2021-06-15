; RUN: sycl-post-link --ir-output-only -spec-const=default %s -S -o - | \
; RUN:   FileCheck %s -check-prefixes=CHECK,CHECK-DEF
; RUN: sycl-post-link --ir-output-only -spec-const=rt %s -S -o - | \
; RUN:   FileCheck %s -check-prefixes=CHECK,CHECK-RT

; This test checks that the post link tool is able to correctly transform
; SYCL 2020 specialization constant intrinsics for different types in a device
; code.

%class.specialization_id = type { half }
%class.specialization_id.0 = type { i32 }
%class.specialization_id.1 = type { %struct.ComposConst }
%struct.ComposConst = type { i32, double, %struct.myConst }
%struct.myConst = type { i32, float }
%class.specialization_id.2 = type { %struct.ComposConst2 }
%struct.ComposConst2 = type { i8, %struct.myConst, double }

@id_half = dso_local global %class.specialization_id { half 0xH4000 }, align 8
@id_int = dso_local global %class.specialization_id.0 { i32 42 }, align 4
@id_compos = dso_local global %class.specialization_id.1 { %struct.ComposConst { i32 1, double 2.000000e+00, %struct.myConst { i32 13, float 0x4020666660000000 } } }, align 8
@id_compos2 = dso_local global %class.specialization_id.2 { %struct.ComposConst2 { i8 1, %struct.myConst { i32 52, float 0x40479999A0000000 }, double 2.000000e+00 } }, align 8

; check that the following globals are preserved: even though they are won't be
; used in the module anymore, they could still be referenced by debug info
; metadata (specialization_id objects are used as template arguments in SYCL
; specialization constant APIs)
; CHECK: @id_half
; CHECK: @id_int
; CHECK: @id_compos
; CHECK: @id_compos2

@__builtin_unique_stable_name._Z27get_specialization_constantIL_Z9id_halfE17specialization_idIdEdET1_v = private unnamed_addr constant [35 x i8] c"_ZTS14name_generatorIL_Z9id_halfEE\00", align 1
@__builtin_unique_stable_name._Z27get_specialization_constantIL_Z6id_intE17specialization_idIiEiET1_v = private unnamed_addr constant [34 x i8] c"_ZTS14name_generatorIL_Z6id_intEE\00", align 1
@__builtin_unique_stable_name._Z27get_specialization_constantIL_Z9id_composE17specialization_idI11ComposConstES1_ET1_v = private unnamed_addr constant [37 x i8] c"_ZTS14name_generatorIL_Z9id_composEE\00", align 1
@__builtin_unique_stable_name._Z27get_specialization_constantIL_Z10id_compos2E17specialization_idI12ComposConst2ES1_ET1_v = private unnamed_addr constant [39 x i8] c"_ZTS14name_generatorIL_Z10id_compos2EE\00", align 1

; CHECK-LABEL: define dso_local void @_Z4testv
define dso_local void @_Z4testv() local_unnamed_addr #0 {
entry:
  %call.i = tail call fast half @_Z37__sycl_getScalar2020SpecConstantValueIDhET_PKcPvS3_(i8* getelementptr inbounds ([35 x i8], [35 x i8]* @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z9id_halfE17specialization_idIdEdET1_v, i64 0, i64 0), i8* bitcast (%class.specialization_id* @id_half to i8*), i8* null)
; CHECK-DEF: %[[GEP:[0-9a-z]+]] = getelementptr i8, i8* null, i32 0
; CHECK-DEF: %[[BITCAST:[0-9a-z]+]] = bitcast i8* %[[GEP]] to half*
; CHECK-DEF: %[[LOAD:[0-9a-z]+]] = load half, half* %[[BITCAST]], align 2
;
; CHECK-RT: call half @_Z20__spirv_SpecConstantiDh(i32 [[#SCID0:]], half 0xH4000)

  %call.i3 = tail call i32 @_Z37__sycl_getScalar2020SpecConstantValueIiET_PKcPvS3_(i8* getelementptr inbounds ([34 x i8], [34 x i8]* @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z6id_intE17specialization_idIiEiET1_v, i64 0, i64 0), i8* bitcast (%class.specialization_id.0* @id_int to i8*), i8* null)
; CHECK-DEF: %[[GEP1:[0-9a-z]+]] = getelementptr i8, i8* null, i32 2
; CHECK-DEF: %[[BITCAST1:[0-9a-z]+]] = bitcast i8* %[[GEP1]] to i32*
; CHECK-DEF: %[[LOAD1:[0-9a-z]+]] = load i32, i32* %[[BITCAST1]], align 4
;
; CHECK-RT: call i32 @_Z20__spirv_SpecConstantii(i32 [[#SCID1:]], i32 42)

  %call.i4 = tail call fast half @_Z37__sycl_getScalar2020SpecConstantValueIDhET_PKcPvS3_(i8* getelementptr inbounds ([35 x i8], [35 x i8]* @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z9id_halfE17specialization_idIdEdET1_v, i64 0, i64 0), i8* bitcast (%class.specialization_id* @id_half to i8*), i8* null)
; CHECK-DEF: %[[GEP2:[0-9a-z]+]] = getelementptr i8, i8* null, i32 0
; CHECK-DEF: %[[BITCAST2:[0-9a-z]+]] = bitcast i8* %[[GEP2]] to half*
; CHECK-DEF: %[[LOAD2:[0-9a-z]+]] = load half, half* %[[BITCAST2]], align 2
;
; CHECK-RT: call half @_Z20__spirv_SpecConstantiDh(i32 [[#SCID0]], half 0xH4000)
  ret void
}

; CHECK-LABEL: define dso_local void @_Z5test2v
define dso_local void @_Z5test2v() local_unnamed_addr #0 {
entry:
  %tmp = alloca %struct.ComposConst, align 8
  %tmp1 = alloca %struct.ComposConst2, align 8
  %tmp2 = alloca %struct.ComposConst, align 8
  %0 = bitcast %struct.ComposConst* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0) #3
  call void @_Z40__sycl_getComposite2020SpecConstantValueI11ComposConstET_PKcPvS4_(%struct.ComposConst* nonnull sret(%struct.ComposConst) align 8 %tmp, i8* getelementptr inbounds ([37 x i8], [37 x i8]* @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z9id_composE17specialization_idI11ComposConstES1_ET1_v, i64 0, i64 0), i8* bitcast (%class.specialization_id.1* @id_compos to i8*), i8* null)
; CHECK-DEF: %[[GEP:[0-9a-z]+]] = getelementptr i8, i8* null, i32 6
; CHECK-DEF: %[[BITCAST:[0-9a-z]+]] = bitcast i8* %[[GEP]] to %struct.ComposConst*
; CHECK-DEF: %[[C1:[0-9a-z]+]] = load %struct.ComposConst, %struct.ComposConst* %[[BITCAST]], align 8
;
; CHECK-RT: %[[#SE1:]] = call i32 @_Z20__spirv_SpecConstantii(i32 [[#SCID2:]], i32 1)
; CHECK-RT: %[[#SE2:]] = call double @_Z20__spirv_SpecConstantid(i32 [[#SCID3:]], double 2.000000e+00)
; CHECK-RT: %[[#SE3:]] = call i32 @_Z20__spirv_SpecConstantii(i32 [[#SCID4:]], i32 13)
; CHECK-RT: %[[#SE4:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID5:]], float 0x4020666660000000)
; CHECK-RT: %[[#CE1:]] = call %struct.myConst @_Z29__spirv_SpecConstantCompositeif(i32 %[[#SE3]], float %[[#SE4]])
; CHECK-RT: %[[C1:[0-9a-z]+]] = call %struct.ComposConst @_Z29__spirv_SpecConstantCompositeidstruct.myConst(i32 %[[#SE1]], double %[[#SE2]], %struct.myConst %[[#CE1]])
;
; CHECK: store %struct.ComposConst %[[C1]], %struct.ComposConst*

  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0) #3
  %1 = getelementptr inbounds %struct.ComposConst2, %struct.ComposConst2* %tmp1, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %1) #3
  call void @_Z40__sycl_getComposite2020SpecConstantValueI12ComposConst2ET_PKcPvS4_(%struct.ComposConst2* nonnull sret(%struct.ComposConst2) align 8 %tmp1, i8* getelementptr inbounds ([39 x i8], [39 x i8]* @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z10id_compos2E17specialization_idI12ComposConst2ES1_ET1_v, i64 0, i64 0), i8* getelementptr inbounds (%class.specialization_id.2, %class.specialization_id.2* @id_compos2, i64 0, i32 0, i32 0), i8* null)  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %1) #3
; CHECK-DEF: %[[GEP1:[0-9a-z]+]] = getelementptr i8, i8* null, i32 30
; CHECK-DEF: %[[BITCAST1:[0-9a-z]+]] = bitcast i8* %[[GEP1]] to %struct.ComposConst2*
; CHECK-DEF: %[[C2:[0-9a-z]+]] = load %struct.ComposConst2, %struct.ComposConst2* %[[BITCAST1]], align 8
;
; CHECK-RT: %[[#SE1:]] = call i8 @_Z20__spirv_SpecConstantia(i32 [[#SCID6:]], i8 1)
; CHECK-RT: %[[#SE2:]] = call i32 @_Z20__spirv_SpecConstantii(i32 [[#SCID7:]], i32 52)
; CHECK-RT: %[[#SE3:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID8:]], float 0x40479999A0000000)
; CHECK-RT: %[[#CE1:]] = call %struct.myConst @_Z29__spirv_SpecConstantCompositeif(i32 %[[#SE2]], float %[[#SE3]])
; CHECK-RT: %[[#SE4:]] = call double @_Z20__spirv_SpecConstantid(i32 [[#SCID9:]], double 2.000000e+00)
; CHECK-RT: %[[C2:[0-9a-z]+]] = call %struct.ComposConst2 @_Z29__spirv_SpecConstantCompositeastruct.myConstd(i8 %[[#SE1]], %struct.myConst %[[#CE1]], double %[[#SE4]])
;
; CHECK: store %struct.ComposConst2 %[[C2]], %struct.ComposConst2*

  %2 = bitcast %struct.ComposConst* %tmp2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %2) #3
  call void @_Z40__sycl_getComposite2020SpecConstantValueI11ComposConstET_PKcPvS4_(%struct.ComposConst* nonnull sret(%struct.ComposConst) align 8 %tmp2, i8* getelementptr inbounds ([37 x i8], [37 x i8]* @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z9id_composE17specialization_idI11ComposConstES1_ET1_v, i64 0, i64 0), i8* bitcast (%class.specialization_id.1* @id_compos to i8*), i8* null)
; CHECK-DEF: %[[GEP2:[0-9a-z]+]] = getelementptr i8, i8* null, i32 6
; CHECK-DEF: %[[BITCAST2:[0-9a-z]+]] = bitcast i8* %[[GEP2]] to %struct.ComposConst*
; CHECK-DEF: %[[C3:[0-9a-z]+]] = load %struct.ComposConst, %struct.ComposConst* %[[BITCAST2]], align 8
;
; CHECK-RT: %[[#SE1:]] = call i32 @_Z20__spirv_SpecConstantii(i32 [[#SCID2]], i32 1)
; CHECK-RT: %[[#SE2:]] = call double @_Z20__spirv_SpecConstantid(i32 [[#SCID3]], double 2.000000e+00)
; CHECK-RT: %[[#SE3:]] = call i32 @_Z20__spirv_SpecConstantii(i32 [[#SCID4]], i32 13)
; CHECK-RT: %[[#SE4:]] = call float @_Z20__spirv_SpecConstantif(i32 [[#SCID5]], float 0x4020666660000000)
; CHECK-RT: %[[#CE1:]] = call %struct.myConst @_Z29__spirv_SpecConstantCompositeif(i32 %[[#SE3]], float %[[#SE4]])
; CHECK-RT: %[[C3:[0-9a-z]+]] = call %struct.ComposConst @_Z29__spirv_SpecConstantCompositeidstruct.myConst(i32 %[[#SE1]], double %[[#SE2]], %struct.myConst %[[#CE1]])
;
; CHECK: store %struct.ComposConst %[[C3]], %struct.ComposConst*
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %2) #3
  ret void
}
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

declare dso_local double @_Z37__sycl_getScalar2020SpecConstantValueIdET_PKcPvS3_(i8*, i8*, i8*) local_unnamed_addr #1

declare dso_local half @_Z37__sycl_getScalar2020SpecConstantValueIDhET_PKcPvS3_(i8*, i8*, i8*) local_unnamed_addr #1

declare dso_local i32 @_Z37__sycl_getScalar2020SpecConstantValueIiET_PKcPvS3_(i8*, i8*, i8*) local_unnamed_addr #1

declare dso_local void @_Z40__sycl_getComposite2020SpecConstantValueI11ComposConstET_PKcPvS4_(%struct.ComposConst* sret(%struct.ComposConst) align 8, i8*, i8*, i8*) local_unnamed_addr #2

declare dso_local void @_Z40__sycl_getComposite2020SpecConstantValueI12ComposConst2ET_PKcPvS4_(%struct.ComposConst2* sret(%struct.ComposConst2) align 8, i8*, i8*, i8*) local_unnamed_addr #2

attributes #0 = { uwtable mustprogress "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind }

; CHECK: !sycl.specialization-constants = !{![[#ID0:]], ![[#ID1:]], ![[#ID2:]], ![[#ID3:]]}
;
; CHECK-DEF: !sycl.specialization-constants-default-values = !{![[#ID4:]], ![[#ID5:]], ![[#ID6:]], ![[#ID7:]]}
; CHECK-RT-NOT: !sycl.specialization-constants-default-values
;
; CHECK: ![[#ID0]] = !{!"_ZTS14name_generatorIL_Z9id_halfEE", i32 0, i32 0, i32 2}
; CHECK: ![[#ID1]] = !{!"_ZTS14name_generatorIL_Z6id_intEE", i32 1, i32 0, i32 4}
;
; For composite types, the amount of metadata is a bit different between native and emulated spec constants
;
; CHECK-DEF: ![[#ID2]] = !{!"_ZTS14name_generatorIL_Z9id_composEE", i32 2, i32 0, i32 24}
; CHECK-DEF: ![[#ID3]] = !{!"_ZTS14name_generatorIL_Z10id_compos2EE", i32 3, i32 0, i32 24
;
; CHECK-RT: ![[#ID2]] = !{!"_ZTS14name_generatorIL_Z9id_composEE", i32 [[#SCID2]], i32 0, i32 4,
; CHECK-RT-SAME: i32 [[#SCID3]], i32 8, i32 8,
; CHECK-RT-SAME: i32 [[#SCID4]], i32 16, i32 4,
; CHECK-RT-SAME: i32 [[#SCID5]], i32 20, i32 4}
; CHECK-RT: ![[#ID3]] = !{!"_ZTS14name_generatorIL_Z10id_compos2EE", i32 [[#SCID6]], i32 0, i32 1,
; CHECK-RT-SAME: i32 [[#SCID7]], i32 4, i32 4,
; CHECK-RT-SAME: i32 [[#SCID8]], i32 8, i32 4,
; CHECK-RT-SAME: i32 [[#SCID9]], i32 16, i32 8}
;
; CHECK-DEF: ![[#ID4]] = !{half 0xH4000}
; CHECK-DEF: ![[#ID5]] = !{i32 42}
; CHECK-DEF: ![[#ID6]] = !{%struct.ComposConst { i32 1, double 2.000000e+00, %struct.myConst { i32 13, float 0x4020666660000000 } }}
; CHECK-DEF: ![[#ID7]] = !{%struct.ComposConst2 { i8 1, %struct.myConst { i32 52, float 0x40479999A0000000 }, double 2.000000e+00 }}
