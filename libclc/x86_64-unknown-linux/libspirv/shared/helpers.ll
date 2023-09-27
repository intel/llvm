declare float @llvm.sqrt.f32(float %n)
declare double @llvm.sqrt.f64(double %n)
declare float @llvm.fabs.f32(float %n)
declare double @llvm.fabs.f64(double %n)
declare float @llvm.trunc.f32(float %n)
declare double @llvm.trunc.f64(double %n)
declare float @llvm.ceil.f32(float %n)
declare double @llvm.ceil.f64(double %n)
declare float @llvm.floor.f32(float %n)
declare double @llvm.floor.f64(double %n)
declare float @llvm.round.f32(float %n)
declare double @llvm.round.f64(double %n)
declare float @llvm.rint.f32(float %n)
declare double @llvm.rint.f64(double %n)
declare float @llvm.cos.f32(float %n)
declare double @llvm.cos.f64(double %n)
declare float @llvm.sin.f32(float %n)
declare double @llvm.sin.f64(double %n)
declare float @llvm.exp2.f32(float %n)
declare double @llvm.exp2.f64(double %n)
declare float @llvm.exp.f32(float %n)
declare double @llvm.exp.f64(double %n)
declare float @llvm.log10.f32(float %n)
declare double @llvm.log10.f64(double %n)
declare float @llvm.log.f32(float %n)
declare double @llvm.log.f64(double %n)
declare float @llvm.log2.f32(float %n)
declare double @llvm.log2.f64(double %n)
declare float @llvm.fma.f32(float %n1, float %n2, float %n3)
declare double @llvm.fma.f64(double %n1, double %n2, double %n3)
declare i32 @llvm.ctpop.i32(i32 %n)
declare i8 @llvm.ctpop.i8(i8 %n)

define dso_local float @_Z13__sqrt_helperf(float %x) {
entry:
  %call = call float @llvm.sqrt.f32(float %x) 
  ret float %call
}


define dso_local double @_Z13__sqrt_helperd(double %x) {
entry:
  %call = call double @llvm.sqrt.f64(double %x) 
  ret double %call
}


define dso_local float @_Z13__fabs_helperf(float %x) {
entry:
  %call = call float @llvm.fabs.f32(float %x) 
  ret float %call
}


define dso_local double @_Z13__fabs_helperd(double %x) {
entry:
  %call = call double @llvm.fabs.f64(double %x) 
  ret double %call
}


define dso_local float @_Z14__trunc_helperf(float %x) {
entry:
  %call = call float @llvm.trunc.f32(float %x) 
  ret float %call
}


define dso_local double @_Z14__trunc_helperd(double %x) {
entry:
  %call = call double @llvm.trunc.f64(double %x) 
  ret double %call
}


define dso_local float @_Z13__ceil_helperf(float %x) {
entry:
  %call = call float @llvm.ceil.f32(float %x) 
  ret float %call
}


define dso_local double @_Z13__ceil_helperd(double %x) {
entry:
  %call = call double @llvm.ceil.f64(double %x) 
  ret double %call
}


define dso_local float @_Z14__floor_helperf(float %x) {
entry:
  %call = call float @llvm.floor.f32(float %x) 
  ret float %call
}


define dso_local double @_Z14__floor_helperd(double %x) {
entry:
  %call = call double @llvm.floor.f64(double %x) 
  ret double %call
}


define dso_local float @_Z14__round_helperf(float %x) {
entry:
  %call = call float @llvm.round.f32(float %x) 
  ret float %call
}


define dso_local double @_Z14__round_helperd(double %x) {
entry:
  %call = call double @llvm.round.f64(double %x) 
  ret double %call
}


define dso_local float @_Z13__rint_helperf(float %x) {
entry:
  %call = call float @llvm.rint.f32(float %x) 
  ret float %call
}


define dso_local double @_Z13__rint_helperd(double %x) {
entry:
  %call = call double @llvm.rint.f64(double %x) 
  ret double %call
}


define dso_local float @_Z20__native_sqrt_helperf(float %x) {
entry:
  %call = call float @llvm.sqrt.f32(float %x) 
  ret float %call
}


define dso_local double @_Z20__native_sqrt_helperd(double %x) {
entry:
  %call = call double @llvm.sqrt.f64(double %x) 
  ret double %call
}


define dso_local float @_Z19__native_cos_helperf(float %x) {
entry:
  %call = call float @llvm.cos.f32(float %x) 
  ret float %call
}


define dso_local double @_Z19__native_cos_helperd(double %x) {
entry:
  %call = call double @llvm.cos.f64(double %x) 
  ret double %call
}


define dso_local float @_Z19__native_sin_helperf(float %x) {
entry:
  %call = call float @llvm.sin.f32(float %x) 
  ret float %call
}


define dso_local double @_Z19__native_sin_helperd(double %x) {
entry:
  %call = call double @llvm.sin.f64(double %x) 
  ret double %call
}


define dso_local float @_Z20__native_exp2_helperf(float %x) {
entry:
  %call = call float @llvm.exp2.f32(float %x) 
  ret float %call
}


define dso_local double @_Z20__native_exp2_helperd(double %x) {
entry:
  %call = call double @llvm.exp2.f64(double %x) 
  ret double %call
}


define dso_local float @_Z19__native_exp_helperf(float %x) {
entry:
  %call = call float @llvm.exp.f32(float %x) 
  ret float %call
}


define dso_local double @_Z19__native_exp_helperd(double %x) {
entry:
  %call = call double @llvm.exp.f64(double %x) 
  ret double %call
}


define dso_local float @_Z21__native_log10_helperf(float %x) {
entry:
  %call = call float @llvm.log10.f32(float %x) 
  ret float %call
}


define dso_local double @_Z21__native_log10_helperd(double %x) {
entry:
  %call = call double @llvm.log10.f64(double %x) 
  ret double %call
}


define dso_local float @_Z19__native_log_helperf(float %x) {
entry:
  %call = call float @llvm.log.f32(float %x) 
  ret float %call
}


define dso_local double @_Z19__native_log_helperd(double %x) {
entry:
  %call = call double @llvm.log.f64(double %x) 
  ret double %call
}


define dso_local float @_Z20__native_log2_helperf(float %x) {
entry:
  %call = call float @llvm.log2.f32(float %x) 
  ret float %call
}


define dso_local double @_Z20__native_log2_helperd(double %x) {
entry:
  %call = call double @llvm.log2.f64(double %x) 
  ret double %call
}


define dso_local float @_Z12__fma_helperfff(float %a, float %b, float %c) {
entry:
  %call = call float @llvm.fma.f32(float %a, float %b, float %c) 
  ret float %call
}


define dso_local double @_Z12__fma_helperddd(double %a, double %b, double %c) {
entry:
  %call = call double @llvm.fma.f64(double %a, double %b, double %c) 
  ret double %call
}


define dso_local i32 @_Z17__popcount_helperi(i32 %x) {
entry:
  %call = call i32 @llvm.ctpop.i32(i32 %x) 
  ret i32 %call
}


define dso_local i8 @_Z17__popcount_helpera(i8 %x) {
entry:
  %call = call i8 @llvm.ctpop.i8(i8 %x) 
  ret i8 %call
}

