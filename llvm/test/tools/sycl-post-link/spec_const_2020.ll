; RUN: sycl-post-link --ir-output-only -spec-const=default %s -S -o - | \
; RUN:   FileCheck %s -check-prefixes=CHECK,CHECK-DEF

; This test checks that the post link tool is able to correctly transform
; specialization constant intrinsics for different types in a device code
; compiled for AOT.

%class.specialization_id = type { double }
%class.specialization_id.0 = type { i32 }

@id_double = dso_local global %class.specialization_id { double 3.140000e+00 }, align 8
@id_int = dso_local global %class.specialization_id.0 { i32 42 }, align 4
; check that no longer used globals are removed:
@__builtin_unique_stable_name._Z27get_specialization_constantIL_Z9id_doubleE17specialization_idIdEdET1_v = private unnamed_addr constant [37 x i8] c"_ZTS14name_generatorIL_Z9id_doubleEE\00", align 1
; CHECK-NOT: @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z9id_doubleE17specialization_idIdEdET1_v
@__builtin_unique_stable_name._Z27get_specialization_constantIL_Z6id_intE17specialization_idIiEiET1_v = private unnamed_addr constant [34 x i8] c"_ZTS14name_generatorIL_Z6id_intEE\00", align 1
; CHECK-NOT: @__builtin_unique_stable_name._Z27get_specialization_constantIL_Z6id_intE17specialization_idIiEiET1_v

define dso_local void @_Z5test2v() local_unnamed_addr #0 {
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

declare dso_local double @_Z37__sycl_getScalar2020SpecConstantValueIdET_PKcPvS3_(i8*, i8*, i8*) local_unnamed_addr #1

declare dso_local i32 @_Z37__sycl_getScalar2020SpecConstantValueIiET_PKcPvS3_(i8*, i8*, i8*) local_unnamed_addr #1

attributes #0 = { uwtable mustprogress "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }