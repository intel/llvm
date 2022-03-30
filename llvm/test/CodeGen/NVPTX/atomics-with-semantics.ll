; RUN: llc < %s -march=nvptx -mcpu=sm_70 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_70 | FileCheck %s

; CHECK-LABEL: .func test_atomics_scope(
define void @test_atomics_scope(float* %fp, float %f,
                                double* %dfp, double %df,
                                i32* %ip, i32 %i,
                                i32* %uip, i32 %ui,
                                i64* %llp, i64 %ll,
                                float addrspace(1)* %fp1,
                                double addrspace(1)* %dfp1,
                                i32 addrspace(1)* %ip1,
                                i32 addrspace(1)* %uip1,
                                i64 addrspace(1)* %llp1,
                                float addrspace(3)* %fp3,
                                double addrspace(3)* %dfp3,
                                i32 addrspace(3)* %ip3,
                                i32 addrspace(3)* %uip3,
                                i64 addrspace(3)* %llp3) #0 {
entry:


  ; CHECK: atom.acquire.add.s32
  %tmp0 = tail call i32 @llvm.nvvm.atomic.add.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.global.add.s32
  %tmp1 = tail call i32 @llvm.nvvm.atomic.add.global.i.acquire.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.shared.add.s32
  %tmp2 = tail call i32 @llvm.nvvm.atomic.add.shared.i.acquire.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.add.s32
  %tmp3 = tail call i32 @llvm.nvvm.atomic.add.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.global.add.s32
  %tmp4 = tail call i32 @llvm.nvvm.atomic.add.global.i.release.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.shared.add.s32
  %tmp5 = tail call i32 @llvm.nvvm.atomic.add.shared.i.release.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.add.s32
  %tmp6 = tail call i32 @llvm.nvvm.atomic.add.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.global.add.s32
  %tmp7 = tail call i32 @llvm.nvvm.atomic.add.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.shared.add.s32
  %tmp8 = tail call i32 @llvm.nvvm.atomic.add.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.sys.add.s32
  %tmp9 = tail call i32 @llvm.nvvm.atomic.add.gen.i.acquire.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.global.add.s32
  %tmp10 = tail call i32 @llvm.nvvm.atomic.add.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.sys.shared.add.s32
  %tmp11 = tail call i32 @llvm.nvvm.atomic.add.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.sys.add.s32
  %tmp12 = tail call i32 @llvm.nvvm.atomic.add.gen.i.release.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.global.add.s32
  %tmp13 = tail call i32 @llvm.nvvm.atomic.add.global.i.release.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.sys.shared.add.s32
  %tmp14 = tail call i32 @llvm.nvvm.atomic.add.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.sys.add.s32
  %tmp15 = tail call i32 @llvm.nvvm.atomic.add.gen.i.acq.rel.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.global.add.s32
  %tmp16 = tail call i32 @llvm.nvvm.atomic.add.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.sys.shared.add.s32
  %tmp17 = tail call i32 @llvm.nvvm.atomic.add.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.cta.add.s32
  %tmp18 = tail call i32 @llvm.nvvm.atomic.add.gen.i.acquire.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.global.add.s32
  %tmp19 = tail call i32 @llvm.nvvm.atomic.add.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.cta.shared.add.s32
  %tmp20 = tail call i32 @llvm.nvvm.atomic.add.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.cta.add.s32
  %tmp21 = tail call i32 @llvm.nvvm.atomic.add.gen.i.release.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.global.add.s32
  %tmp22 = tail call i32 @llvm.nvvm.atomic.add.global.i.release.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.cta.shared.add.s32
  %tmp23 = tail call i32 @llvm.nvvm.atomic.add.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.cta.add.s32
  %tmp24 = tail call i32 @llvm.nvvm.atomic.add.gen.i.acq.rel.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.global.add.s32
  %tmp25 = tail call i32 @llvm.nvvm.atomic.add.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.cta.shared.add.s32
  %tmp26 = tail call i32 @llvm.nvvm.atomic.add.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.add.u64
  %tmp27 = tail call i64 @llvm.nvvm.atomic.add.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.global.add.u64
  %tmp28 = tail call i64 @llvm.nvvm.atomic.add.global.i.acquire.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.shared.add.u64
  %tmp29 = tail call i64 @llvm.nvvm.atomic.add.shared.i.acquire.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.add.u64
  %tmp30 = tail call i64 @llvm.nvvm.atomic.add.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.global.add.u64
  %tmp31 = tail call i64 @llvm.nvvm.atomic.add.global.i.release.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.shared.add.u64
  %tmp32 = tail call i64 @llvm.nvvm.atomic.add.shared.i.release.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.add.u64
  %tmp33 = tail call i64 @llvm.nvvm.atomic.add.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.global.add.u64
  %tmp34 = tail call i64 @llvm.nvvm.atomic.add.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.shared.add.u64
  %tmp35 = tail call i64 @llvm.nvvm.atomic.add.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.sys.add.u64
  %tmp36 = tail call i64 @llvm.nvvm.atomic.add.gen.i.acquire.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.global.add.u64
  %tmp37 = tail call i64 @llvm.nvvm.atomic.add.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.sys.shared.add.u64
  %tmp38 = tail call i64 @llvm.nvvm.atomic.add.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.sys.add.u64
  %tmp39 = tail call i64 @llvm.nvvm.atomic.add.gen.i.release.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.global.add.u64
  %tmp40 = tail call i64 @llvm.nvvm.atomic.add.global.i.release.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.sys.shared.add.u64
  %tmp41 = tail call i64 @llvm.nvvm.atomic.add.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.sys.add.u64
  %tmp42 = tail call i64 @llvm.nvvm.atomic.add.gen.i.acq.rel.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.global.add.u64
  %tmp43 = tail call i64 @llvm.nvvm.atomic.add.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.sys.shared.add.u64
  %tmp44 = tail call i64 @llvm.nvvm.atomic.add.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.cta.add.u64
  %tmp45 = tail call i64 @llvm.nvvm.atomic.add.gen.i.acquire.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.global.add.u64
  %tmp46 = tail call i64 @llvm.nvvm.atomic.add.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.cta.shared.add.u64
  %tmp47 = tail call i64 @llvm.nvvm.atomic.add.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.cta.add.u64
  %tmp48 = tail call i64 @llvm.nvvm.atomic.add.gen.i.release.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.global.add.u64
  %tmp49 = tail call i64 @llvm.nvvm.atomic.add.global.i.release.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.cta.shared.add.u64
  %tmp50 = tail call i64 @llvm.nvvm.atomic.add.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.cta.add.u64
  %tmp51 = tail call i64 @llvm.nvvm.atomic.add.gen.i.acq.rel.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.global.add.u64
  %tmp52 = tail call i64 @llvm.nvvm.atomic.add.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.cta.shared.add.u64
  %tmp53 = tail call i64 @llvm.nvvm.atomic.add.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.add.f32
  %tmp54 = tail call float @llvm.nvvm.atomic.add.gen.f.acquire.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.acquire.global.add.f32
  %tmp55 = tail call float @llvm.nvvm.atomic.add.global.f.acquire.f32.p1f32(float addrspace(1)* %fp1, float %f);

  ; CHECK: atom.acquire.shared.add.f32
  %tmp56 = tail call float @llvm.nvvm.atomic.add.shared.f.acquire.f32.p3f32(float addrspace(3)* %fp3, float %f);

  ; CHECK: atom.release.add.f32
  %tmp57 = tail call float @llvm.nvvm.atomic.add.gen.f.release.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.release.global.add.f32
  %tmp58 = tail call float @llvm.nvvm.atomic.add.global.f.release.f32.p1f32(float addrspace(1)* %fp1, float %f);

  ; CHECK: atom.release.shared.add.f32
  %tmp59 = tail call float @llvm.nvvm.atomic.add.shared.f.release.f32.p3f32(float addrspace(3)* %fp3, float %f);

  ; CHECK: atom.acq_rel.add.f32
  %tmp60 = tail call float @llvm.nvvm.atomic.add.gen.f.acq.rel.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.acq_rel.global.add.f32
  %tmp61 = tail call float @llvm.nvvm.atomic.add.global.f.acq.rel.f32.p1f32(float addrspace(1)* %fp1, float %f);

  ; CHECK: atom.acq_rel.shared.add.f32
  %tmp62 = tail call float @llvm.nvvm.atomic.add.shared.f.acq.rel.f32.p3f32(float addrspace(3)* %fp3, float %f);

  ; CHECK: atom.acquire.sys.add.f32
  %tmp63 = tail call float @llvm.nvvm.atomic.add.gen.f.acquire.sys.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.acquire.sys.global.add.f32
  %tmp64 = tail call float @llvm.nvvm.atomic.add.global.f.acquire.sys.f32.p1f32(float addrspace(1)* %fp1, float %f);

  ; CHECK: atom.acquire.sys.shared.add.f32
  %tmp65 = tail call float @llvm.nvvm.atomic.add.shared.f.acquire.sys.f32.p3f32(float addrspace(3)* %fp3, float %f);

  ; CHECK: atom.release.sys.add.f32
  %tmp66 = tail call float @llvm.nvvm.atomic.add.gen.f.release.sys.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.release.sys.global.add.f32
  %tmp67 = tail call float @llvm.nvvm.atomic.add.global.f.release.sys.f32.p1f32(float addrspace(1)* %fp1, float %f);

  ; CHECK: atom.release.sys.shared.add.f32
  %tmp68 = tail call float @llvm.nvvm.atomic.add.shared.f.release.sys.f32.p3f32(float addrspace(3)* %fp3, float %f);

  ; CHECK: atom.acq_rel.sys.add.f32
  %tmp69 = tail call float @llvm.nvvm.atomic.add.gen.f.acq.rel.sys.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.acq_rel.sys.global.add.f32
  %tmp70 = tail call float @llvm.nvvm.atomic.add.global.f.acq.rel.sys.f32.p1f32(float addrspace(1)* %fp1, float %f);

  ; CHECK: atom.acq_rel.sys.shared.add.f32
  %tmp71 = tail call float @llvm.nvvm.atomic.add.shared.f.acq.rel.sys.f32.p3f32(float addrspace(3)* %fp3, float %f);

  ; CHECK: atom.acquire.cta.add.f32
  %tmp72 = tail call float @llvm.nvvm.atomic.add.gen.f.acquire.cta.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.acquire.cta.global.add.f32
  %tmp73 = tail call float @llvm.nvvm.atomic.add.global.f.acquire.cta.f32.p1f32(float addrspace(1)* %fp1, float %f);

  ; CHECK: atom.acquire.cta.shared.add.f32
  %tmp74 = tail call float @llvm.nvvm.atomic.add.shared.f.acquire.cta.f32.p3f32(float addrspace(3)* %fp3, float %f);

  ; CHECK: atom.release.cta.add.f32
  %tmp75 = tail call float @llvm.nvvm.atomic.add.gen.f.release.cta.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.release.cta.global.add.f32
  %tmp76 = tail call float @llvm.nvvm.atomic.add.global.f.release.cta.f32.p1f32(float addrspace(1)* %fp1, float %f);

  ; CHECK: atom.release.cta.shared.add.f32
  %tmp77 = tail call float @llvm.nvvm.atomic.add.shared.f.release.cta.f32.p3f32(float addrspace(3)* %fp3, float %f);

  ; CHECK: atom.acq_rel.cta.add.f32
  %tmp78 = tail call float @llvm.nvvm.atomic.add.gen.f.acq.rel.cta.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.acq_rel.cta.global.add.f32
  %tmp79 = tail call float @llvm.nvvm.atomic.add.global.f.acq.rel.cta.f32.p1f32(float addrspace(1)* %fp1, float %f);

  ; CHECK: atom.acq_rel.cta.shared.add.f32
  %tmp80 = tail call float @llvm.nvvm.atomic.add.shared.f.acq.rel.cta.f32.p3f32(float addrspace(3)* %fp3, float %f);

  ; CHECK: atom.acquire.add.f64
  %tmp81 = tail call double @llvm.nvvm.atomic.add.gen.f.acquire.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.acquire.global.add.f64
  %tmp82 = tail call double @llvm.nvvm.atomic.add.global.f.acquire.f64.p1f64(double addrspace(1)* %dfp1, double %df);

  ; CHECK: atom.acquire.shared.add.f64
  %tmp83 = tail call double @llvm.nvvm.atomic.add.shared.f.acquire.f64.p3f64(double addrspace(3)* %dfp3, double %df);

  ; CHECK: atom.release.add.f64
  %tmp84 = tail call double @llvm.nvvm.atomic.add.gen.f.release.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.release.global.add.f64
  %tmp85 = tail call double @llvm.nvvm.atomic.add.global.f.release.f64.p1f64(double addrspace(1)* %dfp1, double %df);

  ; CHECK: atom.release.shared.add.f64
  %tmp86 = tail call double @llvm.nvvm.atomic.add.shared.f.release.f64.p3f64(double addrspace(3)* %dfp3, double %df);

  ; CHECK: atom.acq_rel.add.f64
  %tmp87 = tail call double @llvm.nvvm.atomic.add.gen.f.acq.rel.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.acq_rel.global.add.f64
  %tmp88 = tail call double @llvm.nvvm.atomic.add.global.f.acq.rel.f64.p1f64(double addrspace(1)* %dfp1, double %df);

  ; CHECK: atom.acq_rel.shared.add.f64
  %tmp89 = tail call double @llvm.nvvm.atomic.add.shared.f.acq.rel.f64.p3f64(double addrspace(3)* %dfp3, double %df);

  ; CHECK: atom.acquire.sys.add.f64
  %tmp90 = tail call double @llvm.nvvm.atomic.add.gen.f.acquire.sys.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.acquire.sys.global.add.f64
  %tmp91 = tail call double @llvm.nvvm.atomic.add.global.f.acquire.sys.f64.p1f64(double addrspace(1)* %dfp1, double %df);

  ; CHECK: atom.acquire.sys.shared.add.f64
  %tmp92 = tail call double @llvm.nvvm.atomic.add.shared.f.acquire.sys.f64.p3f64(double addrspace(3)* %dfp3, double %df);

  ; CHECK: atom.release.sys.add.f64
  %tmp93 = tail call double @llvm.nvvm.atomic.add.gen.f.release.sys.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.release.sys.global.add.f64
  %tmp94 = tail call double @llvm.nvvm.atomic.add.global.f.release.sys.f64.p1f64(double addrspace(1)* %dfp1, double %df);

  ; CHECK: atom.release.sys.shared.add.f64
  %tmp95 = tail call double @llvm.nvvm.atomic.add.shared.f.release.sys.f64.p3f64(double addrspace(3)* %dfp3, double %df);

  ; CHECK: atom.acq_rel.sys.add.f64
  %tmp96 = tail call double @llvm.nvvm.atomic.add.gen.f.acq.rel.sys.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.acq_rel.sys.global.add.f64
  %tmp97 = tail call double @llvm.nvvm.atomic.add.global.f.acq.rel.sys.f64.p1f64(double addrspace(1)* %dfp1, double %df);

  ; CHECK: atom.acq_rel.sys.shared.add.f64
  %tmp98 = tail call double @llvm.nvvm.atomic.add.shared.f.acq.rel.sys.f64.p3f64(double addrspace(3)* %dfp3, double %df);

  ; CHECK: atom.acquire.cta.add.f64
  %tmp99 = tail call double @llvm.nvvm.atomic.add.gen.f.acquire.cta.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.acquire.cta.global.add.f64
  %tmp100 = tail call double @llvm.nvvm.atomic.add.global.f.acquire.cta.f64.p1f64(double addrspace(1)* %dfp1, double %df);

  ; CHECK: atom.acquire.cta.shared.add.f64
  %tmp101 = tail call double @llvm.nvvm.atomic.add.shared.f.acquire.cta.f64.p3f64(double addrspace(3)* %dfp3, double %df);

  ; CHECK: atom.release.cta.add.f64
  %tmp102 = tail call double @llvm.nvvm.atomic.add.gen.f.release.cta.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.release.cta.global.add.f64
  %tmp103 = tail call double @llvm.nvvm.atomic.add.global.f.release.cta.f64.p1f64(double addrspace(1)* %dfp1, double %df);

  ; CHECK: atom.release.cta.shared.add.f64
  %tmp104 = tail call double @llvm.nvvm.atomic.add.shared.f.release.cta.f64.p3f64(double addrspace(3)* %dfp3, double %df);

  ; CHECK: atom.acq_rel.cta.add.f64
  %tmp105 = tail call double @llvm.nvvm.atomic.add.gen.f.acq.rel.cta.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.acq_rel.cta.global.add.f64
  %tmp106 = tail call double @llvm.nvvm.atomic.add.global.f.acq.rel.cta.f64.p1f64(double addrspace(1)* %dfp1, double %df);

  ; CHECK: atom.acq_rel.cta.shared.add.f64
  %tmp107 = tail call double @llvm.nvvm.atomic.add.shared.f.acq.rel.cta.f64.p3f64(double addrspace(3)* %dfp3, double %df);

  ; CHECK: atom.acquire.exch.b32
  %tmp108 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.global.exch.b32
  %tmp109 = tail call i32 @llvm.nvvm.atomic.exch.global.i.acquire.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.shared.exch.b32
  %tmp110 = tail call i32 @llvm.nvvm.atomic.exch.shared.i.acquire.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.exch.b32
  %tmp111 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.global.exch.b32
  %tmp112 = tail call i32 @llvm.nvvm.atomic.exch.global.i.release.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.shared.exch.b32
  %tmp113 = tail call i32 @llvm.nvvm.atomic.exch.shared.i.release.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.exch.b32
  %tmp114 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.global.exch.b32
  %tmp115 = tail call i32 @llvm.nvvm.atomic.exch.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.shared.exch.b32
  %tmp116 = tail call i32 @llvm.nvvm.atomic.exch.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.sys.exch.b32
  %tmp117 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.acquire.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.global.exch.b32
  %tmp118 = tail call i32 @llvm.nvvm.atomic.exch.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.sys.shared.exch.b32
  %tmp119 = tail call i32 @llvm.nvvm.atomic.exch.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.sys.exch.b32
  %tmp120 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.release.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.global.exch.b32
  %tmp121 = tail call i32 @llvm.nvvm.atomic.exch.global.i.release.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.sys.shared.exch.b32
  %tmp122 = tail call i32 @llvm.nvvm.atomic.exch.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.sys.exch.b32
  %tmp123 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.acq.rel.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.global.exch.b32
  %tmp124 = tail call i32 @llvm.nvvm.atomic.exch.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.sys.shared.exch.b32
  %tmp125 = tail call i32 @llvm.nvvm.atomic.exch.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.cta.exch.b32
  %tmp126 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.acquire.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.global.exch.b32
  %tmp127 = tail call i32 @llvm.nvvm.atomic.exch.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.cta.shared.exch.b32
  %tmp128 = tail call i32 @llvm.nvvm.atomic.exch.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.cta.exch.b32
  %tmp129 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.release.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.global.exch.b32
  %tmp130 = tail call i32 @llvm.nvvm.atomic.exch.global.i.release.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.cta.shared.exch.b32
  %tmp131 = tail call i32 @llvm.nvvm.atomic.exch.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.cta.exch.b32
  %tmp132 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.acq.rel.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.global.exch.b32
  %tmp133 = tail call i32 @llvm.nvvm.atomic.exch.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.cta.shared.exch.b32
  %tmp134 = tail call i32 @llvm.nvvm.atomic.exch.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.exch.b64
  %tmp135 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.global.exch.b64
  %tmp136 = tail call i64 @llvm.nvvm.atomic.exch.global.i.acquire.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.shared.exch.b64
  %tmp137 = tail call i64 @llvm.nvvm.atomic.exch.shared.i.acquire.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.exch.b64
  %tmp138 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.global.exch.b64
  %tmp139 = tail call i64 @llvm.nvvm.atomic.exch.global.i.release.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.shared.exch.b64
  %tmp140 = tail call i64 @llvm.nvvm.atomic.exch.shared.i.release.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.exch.b64
  %tmp141 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.global.exch.b64
  %tmp142 = tail call i64 @llvm.nvvm.atomic.exch.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.shared.exch.b64
  %tmp143 = tail call i64 @llvm.nvvm.atomic.exch.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.sys.exch.b64
  %tmp144 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.acquire.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.global.exch.b64
  %tmp145 = tail call i64 @llvm.nvvm.atomic.exch.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.sys.shared.exch.b64
  %tmp146 = tail call i64 @llvm.nvvm.atomic.exch.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.sys.exch.b64
  %tmp147 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.release.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.global.exch.b64
  %tmp148 = tail call i64 @llvm.nvvm.atomic.exch.global.i.release.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.sys.shared.exch.b64
  %tmp149 = tail call i64 @llvm.nvvm.atomic.exch.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.sys.exch.b64
  %tmp150 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.acq.rel.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.global.exch.b64
  %tmp151 = tail call i64 @llvm.nvvm.atomic.exch.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.sys.shared.exch.b64
  %tmp152 = tail call i64 @llvm.nvvm.atomic.exch.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.cta.exch.b64
  %tmp153 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.acquire.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.global.exch.b64
  %tmp154 = tail call i64 @llvm.nvvm.atomic.exch.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.cta.shared.exch.b64
  %tmp155 = tail call i64 @llvm.nvvm.atomic.exch.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.cta.exch.b64
  %tmp156 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.release.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.global.exch.b64
  %tmp157 = tail call i64 @llvm.nvvm.atomic.exch.global.i.release.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.cta.shared.exch.b64
  %tmp158 = tail call i64 @llvm.nvvm.atomic.exch.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.cta.exch.b64
  %tmp159 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.acq.rel.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.global.exch.b64
  %tmp160 = tail call i64 @llvm.nvvm.atomic.exch.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.cta.shared.exch.b64
  %tmp161 = tail call i64 @llvm.nvvm.atomic.exch.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.max.s32
  %tmp162 = tail call i32 @llvm.nvvm.atomic.max.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.global.max.s32
  %tmp163 = tail call i32 @llvm.nvvm.atomic.max.global.i.acquire.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.shared.max.s32
  %tmp164 = tail call i32 @llvm.nvvm.atomic.max.shared.i.acquire.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.max.s32
  %tmp165 = tail call i32 @llvm.nvvm.atomic.max.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.global.max.s32
  %tmp166 = tail call i32 @llvm.nvvm.atomic.max.global.i.release.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.shared.max.s32
  %tmp167 = tail call i32 @llvm.nvvm.atomic.max.shared.i.release.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.max.s32
  %tmp168 = tail call i32 @llvm.nvvm.atomic.max.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.global.max.s32
  %tmp169 = tail call i32 @llvm.nvvm.atomic.max.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.shared.max.s32
  %tmp170 = tail call i32 @llvm.nvvm.atomic.max.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.sys.max.s32
  %tmp171 = tail call i32 @llvm.nvvm.atomic.max.gen.i.acquire.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.global.max.s32
  %tmp172 = tail call i32 @llvm.nvvm.atomic.max.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.sys.shared.max.s32
  %tmp173 = tail call i32 @llvm.nvvm.atomic.max.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.sys.max.s32
  %tmp174 = tail call i32 @llvm.nvvm.atomic.max.gen.i.release.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.global.max.s32
  %tmp175 = tail call i32 @llvm.nvvm.atomic.max.global.i.release.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.sys.shared.max.s32
  %tmp176 = tail call i32 @llvm.nvvm.atomic.max.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.sys.max.s32
  %tmp177 = tail call i32 @llvm.nvvm.atomic.max.gen.i.acq.rel.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.global.max.s32
  %tmp178 = tail call i32 @llvm.nvvm.atomic.max.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.sys.shared.max.s32
  %tmp179 = tail call i32 @llvm.nvvm.atomic.max.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.cta.max.s32
  %tmp180 = tail call i32 @llvm.nvvm.atomic.max.gen.i.acquire.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.global.max.s32
  %tmp181 = tail call i32 @llvm.nvvm.atomic.max.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.cta.shared.max.s32
  %tmp182 = tail call i32 @llvm.nvvm.atomic.max.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.cta.max.s32
  %tmp183 = tail call i32 @llvm.nvvm.atomic.max.gen.i.release.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.global.max.s32
  %tmp184 = tail call i32 @llvm.nvvm.atomic.max.global.i.release.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.cta.shared.max.s32
  %tmp185 = tail call i32 @llvm.nvvm.atomic.max.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.cta.max.s32
  %tmp186 = tail call i32 @llvm.nvvm.atomic.max.gen.i.acq.rel.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.global.max.s32
  %tmp187 = tail call i32 @llvm.nvvm.atomic.max.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.cta.shared.max.s32
  %tmp188 = tail call i32 @llvm.nvvm.atomic.max.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.max.s64
  %tmp189 = tail call i64 @llvm.nvvm.atomic.max.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.global.max.s64
  %tmp190 = tail call i64 @llvm.nvvm.atomic.max.global.i.acquire.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.shared.max.s64
  %tmp191 = tail call i64 @llvm.nvvm.atomic.max.shared.i.acquire.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.max.s64
  %tmp192 = tail call i64 @llvm.nvvm.atomic.max.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.global.max.s64
  %tmp193 = tail call i64 @llvm.nvvm.atomic.max.global.i.release.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.shared.max.s64
  %tmp194 = tail call i64 @llvm.nvvm.atomic.max.shared.i.release.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.max.s64
  %tmp195 = tail call i64 @llvm.nvvm.atomic.max.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.global.max.s64
  %tmp196 = tail call i64 @llvm.nvvm.atomic.max.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.shared.max.s64
  %tmp197 = tail call i64 @llvm.nvvm.atomic.max.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.sys.max.s64
  %tmp198 = tail call i64 @llvm.nvvm.atomic.max.gen.i.acquire.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.global.max.s64
  %tmp199 = tail call i64 @llvm.nvvm.atomic.max.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.sys.shared.max.s64
  %tmp200 = tail call i64 @llvm.nvvm.atomic.max.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.sys.max.s64
  %tmp201 = tail call i64 @llvm.nvvm.atomic.max.gen.i.release.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.global.max.s64
  %tmp202 = tail call i64 @llvm.nvvm.atomic.max.global.i.release.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.sys.shared.max.s64
  %tmp203 = tail call i64 @llvm.nvvm.atomic.max.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.sys.max.s64
  %tmp204 = tail call i64 @llvm.nvvm.atomic.max.gen.i.acq.rel.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.global.max.s64
  %tmp205 = tail call i64 @llvm.nvvm.atomic.max.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.sys.shared.max.s64
  %tmp206 = tail call i64 @llvm.nvvm.atomic.max.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.cta.max.s64
  %tmp207 = tail call i64 @llvm.nvvm.atomic.max.gen.i.acquire.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.global.max.s64
  %tmp208 = tail call i64 @llvm.nvvm.atomic.max.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.cta.shared.max.s64
  %tmp209 = tail call i64 @llvm.nvvm.atomic.max.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.cta.max.s64
  %tmp210 = tail call i64 @llvm.nvvm.atomic.max.gen.i.release.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.global.max.s64
  %tmp211 = tail call i64 @llvm.nvvm.atomic.max.global.i.release.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.cta.shared.max.s64
  %tmp212 = tail call i64 @llvm.nvvm.atomic.max.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.cta.max.s64
  %tmp213 = tail call i64 @llvm.nvvm.atomic.max.gen.i.acq.rel.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.global.max.s64
  %tmp214 = tail call i64 @llvm.nvvm.atomic.max.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.cta.shared.max.s64
  %tmp215 = tail call i64 @llvm.nvvm.atomic.max.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.max.u32
  %tmp216 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.global.max.u32
  %tmp217 = tail call i32 @llvm.nvvm.atomic.max.global.ui.acquire.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.shared.max.u32
  %tmp218 = tail call i32 @llvm.nvvm.atomic.max.shared.ui.acquire.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.max.u32
  %tmp219 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.global.max.u32
  %tmp220 = tail call i32 @llvm.nvvm.atomic.max.global.ui.release.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.shared.max.u32
  %tmp221 = tail call i32 @llvm.nvvm.atomic.max.shared.ui.release.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.max.u32
  %tmp222 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.global.max.u32
  %tmp223 = tail call i32 @llvm.nvvm.atomic.max.global.ui.acq.rel.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.shared.max.u32
  %tmp224 = tail call i32 @llvm.nvvm.atomic.max.shared.ui.acq.rel.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.sys.max.u32
  %tmp225 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.acquire.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.global.max.u32
  %tmp226 = tail call i32 @llvm.nvvm.atomic.max.global.ui.acquire.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.sys.shared.max.u32
  %tmp227 = tail call i32 @llvm.nvvm.atomic.max.shared.ui.acquire.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.sys.max.u32
  %tmp228 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.release.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.global.max.u32
  %tmp229 = tail call i32 @llvm.nvvm.atomic.max.global.ui.release.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.sys.shared.max.u32
  %tmp230 = tail call i32 @llvm.nvvm.atomic.max.shared.ui.release.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.sys.max.u32
  %tmp231 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.acq.rel.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.global.max.u32
  %tmp232 = tail call i32 @llvm.nvvm.atomic.max.global.ui.acq.rel.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.sys.shared.max.u32
  %tmp233 = tail call i32 @llvm.nvvm.atomic.max.shared.ui.acq.rel.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.cta.max.u32
  %tmp234 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.acquire.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.global.max.u32
  %tmp235 = tail call i32 @llvm.nvvm.atomic.max.global.ui.acquire.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.cta.shared.max.u32
  %tmp236 = tail call i32 @llvm.nvvm.atomic.max.shared.ui.acquire.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.cta.max.u32
  %tmp237 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.release.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.global.max.u32
  %tmp238 = tail call i32 @llvm.nvvm.atomic.max.global.ui.release.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.cta.shared.max.u32
  %tmp239 = tail call i32 @llvm.nvvm.atomic.max.shared.ui.release.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.cta.max.u32
  %tmp240 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.acq.rel.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.global.max.u32
  %tmp241 = tail call i32 @llvm.nvvm.atomic.max.global.ui.acq.rel.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.cta.shared.max.u32
  %tmp242 = tail call i32 @llvm.nvvm.atomic.max.shared.ui.acq.rel.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.max.u64
  %tmp243 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.global.max.u64
  %tmp244 = tail call i64 @llvm.nvvm.atomic.max.global.ui.acquire.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.shared.max.u64
  %tmp245 = tail call i64 @llvm.nvvm.atomic.max.shared.ui.acquire.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.max.u64
  %tmp246 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.global.max.u64
  %tmp247 = tail call i64 @llvm.nvvm.atomic.max.global.ui.release.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.shared.max.u64
  %tmp248 = tail call i64 @llvm.nvvm.atomic.max.shared.ui.release.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.max.u64
  %tmp249 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.global.max.u64
  %tmp250 = tail call i64 @llvm.nvvm.atomic.max.global.ui.acq.rel.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.shared.max.u64
  %tmp251 = tail call i64 @llvm.nvvm.atomic.max.shared.ui.acq.rel.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.sys.max.u64
  %tmp252 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.acquire.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.global.max.u64
  %tmp253 = tail call i64 @llvm.nvvm.atomic.max.global.ui.acquire.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.sys.shared.max.u64
  %tmp254 = tail call i64 @llvm.nvvm.atomic.max.shared.ui.acquire.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.sys.max.u64
  %tmp255 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.release.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.global.max.u64
  %tmp256 = tail call i64 @llvm.nvvm.atomic.max.global.ui.release.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.sys.shared.max.u64
  %tmp257 = tail call i64 @llvm.nvvm.atomic.max.shared.ui.release.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.sys.max.u64
  %tmp258 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.acq.rel.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.global.max.u64
  %tmp259 = tail call i64 @llvm.nvvm.atomic.max.global.ui.acq.rel.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.sys.shared.max.u64
  %tmp260 = tail call i64 @llvm.nvvm.atomic.max.shared.ui.acq.rel.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.cta.max.u64
  %tmp261 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.acquire.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.global.max.u64
  %tmp262 = tail call i64 @llvm.nvvm.atomic.max.global.ui.acquire.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.cta.shared.max.u64
  %tmp263 = tail call i64 @llvm.nvvm.atomic.max.shared.ui.acquire.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.cta.max.u64
  %tmp264 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.release.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.global.max.u64
  %tmp265 = tail call i64 @llvm.nvvm.atomic.max.global.ui.release.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.cta.shared.max.u64
  %tmp266 = tail call i64 @llvm.nvvm.atomic.max.shared.ui.release.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.cta.max.u64
  %tmp267 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.acq.rel.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.global.max.u64
  %tmp268 = tail call i64 @llvm.nvvm.atomic.max.global.ui.acq.rel.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.cta.shared.max.u64
  %tmp269 = tail call i64 @llvm.nvvm.atomic.max.shared.ui.acq.rel.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.min.s32
  %tmp270 = tail call i32 @llvm.nvvm.atomic.min.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.global.min.s32
  %tmp271 = tail call i32 @llvm.nvvm.atomic.min.global.i.acquire.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.shared.min.s32
  %tmp272 = tail call i32 @llvm.nvvm.atomic.min.shared.i.acquire.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.min.s32
  %tmp273 = tail call i32 @llvm.nvvm.atomic.min.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.global.min.s32
  %tmp274 = tail call i32 @llvm.nvvm.atomic.min.global.i.release.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.shared.min.s32
  %tmp275 = tail call i32 @llvm.nvvm.atomic.min.shared.i.release.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.min.s32
  %tmp276 = tail call i32 @llvm.nvvm.atomic.min.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.global.min.s32
  %tmp277 = tail call i32 @llvm.nvvm.atomic.min.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.shared.min.s32
  %tmp278 = tail call i32 @llvm.nvvm.atomic.min.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.sys.min.s32
  %tmp279 = tail call i32 @llvm.nvvm.atomic.min.gen.i.acquire.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.global.min.s32
  %tmp280 = tail call i32 @llvm.nvvm.atomic.min.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.sys.shared.min.s32
  %tmp281 = tail call i32 @llvm.nvvm.atomic.min.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.sys.min.s32
  %tmp282 = tail call i32 @llvm.nvvm.atomic.min.gen.i.release.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.global.min.s32
  %tmp283 = tail call i32 @llvm.nvvm.atomic.min.global.i.release.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.sys.shared.min.s32
  %tmp284 = tail call i32 @llvm.nvvm.atomic.min.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.sys.min.s32
  %tmp285 = tail call i32 @llvm.nvvm.atomic.min.gen.i.acq.rel.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.global.min.s32
  %tmp286 = tail call i32 @llvm.nvvm.atomic.min.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.sys.shared.min.s32
  %tmp287 = tail call i32 @llvm.nvvm.atomic.min.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.cta.min.s32
  %tmp288 = tail call i32 @llvm.nvvm.atomic.min.gen.i.acquire.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.global.min.s32
  %tmp289 = tail call i32 @llvm.nvvm.atomic.min.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.cta.shared.min.s32
  %tmp290 = tail call i32 @llvm.nvvm.atomic.min.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.cta.min.s32
  %tmp291 = tail call i32 @llvm.nvvm.atomic.min.gen.i.release.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.global.min.s32
  %tmp292 = tail call i32 @llvm.nvvm.atomic.min.global.i.release.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.cta.shared.min.s32
  %tmp293 = tail call i32 @llvm.nvvm.atomic.min.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.cta.min.s32
  %tmp294 = tail call i32 @llvm.nvvm.atomic.min.gen.i.acq.rel.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.global.min.s32
  %tmp295 = tail call i32 @llvm.nvvm.atomic.min.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.cta.shared.min.s32
  %tmp296 = tail call i32 @llvm.nvvm.atomic.min.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.min.s64
  %tmp297 = tail call i64 @llvm.nvvm.atomic.min.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.global.min.s64
  %tmp298 = tail call i64 @llvm.nvvm.atomic.min.global.i.acquire.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.shared.min.s64
  %tmp299 = tail call i64 @llvm.nvvm.atomic.min.shared.i.acquire.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.min.s64
  %tmp300 = tail call i64 @llvm.nvvm.atomic.min.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.global.min.s64
  %tmp301 = tail call i64 @llvm.nvvm.atomic.min.global.i.release.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.shared.min.s64
  %tmp302 = tail call i64 @llvm.nvvm.atomic.min.shared.i.release.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.min.s64
  %tmp303 = tail call i64 @llvm.nvvm.atomic.min.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.global.min.s64
  %tmp304 = tail call i64 @llvm.nvvm.atomic.min.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.shared.min.s64
  %tmp305 = tail call i64 @llvm.nvvm.atomic.min.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.sys.min.s64
  %tmp306 = tail call i64 @llvm.nvvm.atomic.min.gen.i.acquire.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.global.min.s64
  %tmp307 = tail call i64 @llvm.nvvm.atomic.min.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.sys.shared.min.s64
  %tmp308 = tail call i64 @llvm.nvvm.atomic.min.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.sys.min.s64
  %tmp309 = tail call i64 @llvm.nvvm.atomic.min.gen.i.release.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.global.min.s64
  %tmp310 = tail call i64 @llvm.nvvm.atomic.min.global.i.release.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.sys.shared.min.s64
  %tmp311 = tail call i64 @llvm.nvvm.atomic.min.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.sys.min.s64
  %tmp312 = tail call i64 @llvm.nvvm.atomic.min.gen.i.acq.rel.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.global.min.s64
  %tmp313 = tail call i64 @llvm.nvvm.atomic.min.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.sys.shared.min.s64
  %tmp314 = tail call i64 @llvm.nvvm.atomic.min.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.cta.min.s64
  %tmp315 = tail call i64 @llvm.nvvm.atomic.min.gen.i.acquire.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.global.min.s64
  %tmp316 = tail call i64 @llvm.nvvm.atomic.min.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.cta.shared.min.s64
  %tmp317 = tail call i64 @llvm.nvvm.atomic.min.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.cta.min.s64
  %tmp318 = tail call i64 @llvm.nvvm.atomic.min.gen.i.release.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.global.min.s64
  %tmp319 = tail call i64 @llvm.nvvm.atomic.min.global.i.release.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.cta.shared.min.s64
  %tmp320 = tail call i64 @llvm.nvvm.atomic.min.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.cta.min.s64
  %tmp321 = tail call i64 @llvm.nvvm.atomic.min.gen.i.acq.rel.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.global.min.s64
  %tmp322 = tail call i64 @llvm.nvvm.atomic.min.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.cta.shared.min.s64
  %tmp323 = tail call i64 @llvm.nvvm.atomic.min.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.min.u32
  %tmp324 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.global.min.u32
  %tmp325 = tail call i32 @llvm.nvvm.atomic.min.global.ui.acquire.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.shared.min.u32
  %tmp326 = tail call i32 @llvm.nvvm.atomic.min.shared.ui.acquire.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.min.u32
  %tmp327 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.global.min.u32
  %tmp328 = tail call i32 @llvm.nvvm.atomic.min.global.ui.release.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.shared.min.u32
  %tmp329 = tail call i32 @llvm.nvvm.atomic.min.shared.ui.release.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.min.u32
  %tmp330 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.global.min.u32
  %tmp331 = tail call i32 @llvm.nvvm.atomic.min.global.ui.acq.rel.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.shared.min.u32
  %tmp332 = tail call i32 @llvm.nvvm.atomic.min.shared.ui.acq.rel.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.sys.min.u32
  %tmp333 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.acquire.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.global.min.u32
  %tmp334 = tail call i32 @llvm.nvvm.atomic.min.global.ui.acquire.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.sys.shared.min.u32
  %tmp335 = tail call i32 @llvm.nvvm.atomic.min.shared.ui.acquire.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.sys.min.u32
  %tmp336 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.release.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.global.min.u32
  %tmp337 = tail call i32 @llvm.nvvm.atomic.min.global.ui.release.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.sys.shared.min.u32
  %tmp338 = tail call i32 @llvm.nvvm.atomic.min.shared.ui.release.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.sys.min.u32
  %tmp339 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.acq.rel.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.global.min.u32
  %tmp340 = tail call i32 @llvm.nvvm.atomic.min.global.ui.acq.rel.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.sys.shared.min.u32
  %tmp341 = tail call i32 @llvm.nvvm.atomic.min.shared.ui.acq.rel.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.cta.min.u32
  %tmp342 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.acquire.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.global.min.u32
  %tmp343 = tail call i32 @llvm.nvvm.atomic.min.global.ui.acquire.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.cta.shared.min.u32
  %tmp344 = tail call i32 @llvm.nvvm.atomic.min.shared.ui.acquire.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.cta.min.u32
  %tmp345 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.release.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.global.min.u32
  %tmp346 = tail call i32 @llvm.nvvm.atomic.min.global.ui.release.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.cta.shared.min.u32
  %tmp347 = tail call i32 @llvm.nvvm.atomic.min.shared.ui.release.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.cta.min.u32
  %tmp348 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.acq.rel.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.global.min.u32
  %tmp349 = tail call i32 @llvm.nvvm.atomic.min.global.ui.acq.rel.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.cta.shared.min.u32
  %tmp350 = tail call i32 @llvm.nvvm.atomic.min.shared.ui.acq.rel.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.min.u64
  %tmp351 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.global.min.u64
  %tmp352 = tail call i64 @llvm.nvvm.atomic.min.global.ui.acquire.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.shared.min.u64
  %tmp353 = tail call i64 @llvm.nvvm.atomic.min.shared.ui.acquire.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.min.u64
  %tmp354 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.global.min.u64
  %tmp355 = tail call i64 @llvm.nvvm.atomic.min.global.ui.release.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.shared.min.u64
  %tmp356 = tail call i64 @llvm.nvvm.atomic.min.shared.ui.release.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.min.u64
  %tmp357 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.global.min.u64
  %tmp358 = tail call i64 @llvm.nvvm.atomic.min.global.ui.acq.rel.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.shared.min.u64
  %tmp359 = tail call i64 @llvm.nvvm.atomic.min.shared.ui.acq.rel.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.sys.min.u64
  %tmp360 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.acquire.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.global.min.u64
  %tmp361 = tail call i64 @llvm.nvvm.atomic.min.global.ui.acquire.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.sys.shared.min.u64
  %tmp362 = tail call i64 @llvm.nvvm.atomic.min.shared.ui.acquire.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.sys.min.u64
  %tmp363 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.release.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.global.min.u64
  %tmp364 = tail call i64 @llvm.nvvm.atomic.min.global.ui.release.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.sys.shared.min.u64
  %tmp365 = tail call i64 @llvm.nvvm.atomic.min.shared.ui.release.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.sys.min.u64
  %tmp366 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.acq.rel.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.global.min.u64
  %tmp367 = tail call i64 @llvm.nvvm.atomic.min.global.ui.acq.rel.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.sys.shared.min.u64
  %tmp368 = tail call i64 @llvm.nvvm.atomic.min.shared.ui.acq.rel.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.cta.min.u64
  %tmp369 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.acquire.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.global.min.u64
  %tmp370 = tail call i64 @llvm.nvvm.atomic.min.global.ui.acquire.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.cta.shared.min.u64
  %tmp371 = tail call i64 @llvm.nvvm.atomic.min.shared.ui.acquire.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.cta.min.u64
  %tmp372 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.release.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.global.min.u64
  %tmp373 = tail call i64 @llvm.nvvm.atomic.min.global.ui.release.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.cta.shared.min.u64
  %tmp374 = tail call i64 @llvm.nvvm.atomic.min.shared.ui.release.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.cta.min.u64
  %tmp375 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.acq.rel.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.global.min.u64
  %tmp376 = tail call i64 @llvm.nvvm.atomic.min.global.ui.acq.rel.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.cta.shared.min.u64
  %tmp377 = tail call i64 @llvm.nvvm.atomic.min.shared.ui.acq.rel.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.inc.u32
  %tmp378 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.global.inc.u32
  %tmp379 = tail call i32 @llvm.nvvm.atomic.inc.global.i.acquire.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.shared.inc.u32
  %tmp380 = tail call i32 @llvm.nvvm.atomic.inc.shared.i.acquire.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.inc.u32
  %tmp381 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.global.inc.u32
  %tmp382 = tail call i32 @llvm.nvvm.atomic.inc.global.i.release.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.shared.inc.u32
  %tmp383 = tail call i32 @llvm.nvvm.atomic.inc.shared.i.release.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.inc.u32
  %tmp384 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.global.inc.u32
  %tmp385 = tail call i32 @llvm.nvvm.atomic.inc.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.shared.inc.u32
  %tmp386 = tail call i32 @llvm.nvvm.atomic.inc.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.sys.inc.u32
  %tmp387 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.acquire.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.global.inc.u32
  %tmp388 = tail call i32 @llvm.nvvm.atomic.inc.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.sys.shared.inc.u32
  %tmp389 = tail call i32 @llvm.nvvm.atomic.inc.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.sys.inc.u32
  %tmp390 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.release.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.global.inc.u32
  %tmp391 = tail call i32 @llvm.nvvm.atomic.inc.global.i.release.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.sys.shared.inc.u32
  %tmp392 = tail call i32 @llvm.nvvm.atomic.inc.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.sys.inc.u32
  %tmp393 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.acq.rel.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.global.inc.u32
  %tmp394 = tail call i32 @llvm.nvvm.atomic.inc.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.sys.shared.inc.u32
  %tmp395 = tail call i32 @llvm.nvvm.atomic.inc.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.cta.inc.u32
  %tmp396 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.acquire.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.global.inc.u32
  %tmp397 = tail call i32 @llvm.nvvm.atomic.inc.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.cta.shared.inc.u32
  %tmp398 = tail call i32 @llvm.nvvm.atomic.inc.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.cta.inc.u32
  %tmp399 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.release.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.global.inc.u32
  %tmp400 = tail call i32 @llvm.nvvm.atomic.inc.global.i.release.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.cta.shared.inc.u32
  %tmp401 = tail call i32 @llvm.nvvm.atomic.inc.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.cta.inc.u32
  %tmp402 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.acq.rel.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.global.inc.u32
  %tmp403 = tail call i32 @llvm.nvvm.atomic.inc.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.cta.shared.inc.u32
  %tmp404 = tail call i32 @llvm.nvvm.atomic.inc.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.inc.u64
  %tmp405 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.global.inc.u64
  %tmp406 = tail call i64 @llvm.nvvm.atomic.inc.global.i.acquire.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.shared.inc.u64
  %tmp407 = tail call i64 @llvm.nvvm.atomic.inc.shared.i.acquire.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.inc.u64
  %tmp408 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.global.inc.u64
  %tmp409 = tail call i64 @llvm.nvvm.atomic.inc.global.i.release.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.shared.inc.u64
  %tmp410 = tail call i64 @llvm.nvvm.atomic.inc.shared.i.release.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.inc.u64
  %tmp411 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.global.inc.u64
  %tmp412 = tail call i64 @llvm.nvvm.atomic.inc.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.shared.inc.u64
  %tmp413 = tail call i64 @llvm.nvvm.atomic.inc.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.sys.inc.u64
  %tmp414 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.acquire.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.global.inc.u64
  %tmp415 = tail call i64 @llvm.nvvm.atomic.inc.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.sys.shared.inc.u64
  %tmp416 = tail call i64 @llvm.nvvm.atomic.inc.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.sys.inc.u64
  %tmp417 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.release.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.global.inc.u64
  %tmp418 = tail call i64 @llvm.nvvm.atomic.inc.global.i.release.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.sys.shared.inc.u64
  %tmp419 = tail call i64 @llvm.nvvm.atomic.inc.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.sys.inc.u64
  %tmp420 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.acq.rel.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.global.inc.u64
  %tmp421 = tail call i64 @llvm.nvvm.atomic.inc.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.sys.shared.inc.u64
  %tmp422 = tail call i64 @llvm.nvvm.atomic.inc.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.cta.inc.u64
  %tmp423 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.acquire.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.global.inc.u64
  %tmp424 = tail call i64 @llvm.nvvm.atomic.inc.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.cta.shared.inc.u64
  %tmp425 = tail call i64 @llvm.nvvm.atomic.inc.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.cta.inc.u64
  %tmp426 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.release.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.global.inc.u64
  %tmp427 = tail call i64 @llvm.nvvm.atomic.inc.global.i.release.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.cta.shared.inc.u64
  %tmp428 = tail call i64 @llvm.nvvm.atomic.inc.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.cta.inc.u64
  %tmp429 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.acq.rel.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.global.inc.u64
  %tmp430 = tail call i64 @llvm.nvvm.atomic.inc.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.cta.shared.inc.u64
  %tmp431 = tail call i64 @llvm.nvvm.atomic.inc.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.dec.u32
  %tmp432 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.global.dec.u32
  %tmp433 = tail call i32 @llvm.nvvm.atomic.dec.global.i.acquire.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.shared.dec.u32
  %tmp434 = tail call i32 @llvm.nvvm.atomic.dec.shared.i.acquire.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.dec.u32
  %tmp435 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.global.dec.u32
  %tmp436 = tail call i32 @llvm.nvvm.atomic.dec.global.i.release.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.shared.dec.u32
  %tmp437 = tail call i32 @llvm.nvvm.atomic.dec.shared.i.release.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.dec.u32
  %tmp438 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.global.dec.u32
  %tmp439 = tail call i32 @llvm.nvvm.atomic.dec.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.shared.dec.u32
  %tmp440 = tail call i32 @llvm.nvvm.atomic.dec.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.sys.dec.u32
  %tmp441 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.acquire.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.global.dec.u32
  %tmp442 = tail call i32 @llvm.nvvm.atomic.dec.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.sys.shared.dec.u32
  %tmp443 = tail call i32 @llvm.nvvm.atomic.dec.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.sys.dec.u32
  %tmp444 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.release.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.global.dec.u32
  %tmp445 = tail call i32 @llvm.nvvm.atomic.dec.global.i.release.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.sys.shared.dec.u32
  %tmp446 = tail call i32 @llvm.nvvm.atomic.dec.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.sys.dec.u32
  %tmp447 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.acq.rel.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.global.dec.u32
  %tmp448 = tail call i32 @llvm.nvvm.atomic.dec.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.sys.shared.dec.u32
  %tmp449 = tail call i32 @llvm.nvvm.atomic.dec.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.cta.dec.u32
  %tmp450 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.acquire.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.global.dec.u32
  %tmp451 = tail call i32 @llvm.nvvm.atomic.dec.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.cta.shared.dec.u32
  %tmp452 = tail call i32 @llvm.nvvm.atomic.dec.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.cta.dec.u32
  %tmp453 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.release.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.global.dec.u32
  %tmp454 = tail call i32 @llvm.nvvm.atomic.dec.global.i.release.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.cta.shared.dec.u32
  %tmp455 = tail call i32 @llvm.nvvm.atomic.dec.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.cta.dec.u32
  %tmp456 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.acq.rel.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.global.dec.u32
  %tmp457 = tail call i32 @llvm.nvvm.atomic.dec.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.cta.shared.dec.u32
  %tmp458 = tail call i32 @llvm.nvvm.atomic.dec.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.dec.u64
  %tmp459 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.global.dec.u64
  %tmp460 = tail call i64 @llvm.nvvm.atomic.dec.global.i.acquire.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.shared.dec.u64
  %tmp461 = tail call i64 @llvm.nvvm.atomic.dec.shared.i.acquire.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.dec.u64
  %tmp462 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.global.dec.u64
  %tmp463 = tail call i64 @llvm.nvvm.atomic.dec.global.i.release.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.shared.dec.u64
  %tmp464 = tail call i64 @llvm.nvvm.atomic.dec.shared.i.release.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.dec.u64
  %tmp465 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.global.dec.u64
  %tmp466 = tail call i64 @llvm.nvvm.atomic.dec.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.shared.dec.u64
  %tmp467 = tail call i64 @llvm.nvvm.atomic.dec.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.sys.dec.u64
  %tmp468 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.acquire.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.global.dec.u64
  %tmp469 = tail call i64 @llvm.nvvm.atomic.dec.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.sys.shared.dec.u64
  %tmp470 = tail call i64 @llvm.nvvm.atomic.dec.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.sys.dec.u64
  %tmp471 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.release.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.global.dec.u64
  %tmp472 = tail call i64 @llvm.nvvm.atomic.dec.global.i.release.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.sys.shared.dec.u64
  %tmp473 = tail call i64 @llvm.nvvm.atomic.dec.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.sys.dec.u64
  %tmp474 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.acq.rel.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.global.dec.u64
  %tmp475 = tail call i64 @llvm.nvvm.atomic.dec.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.sys.shared.dec.u64
  %tmp476 = tail call i64 @llvm.nvvm.atomic.dec.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.cta.dec.u64
  %tmp477 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.acquire.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.global.dec.u64
  %tmp478 = tail call i64 @llvm.nvvm.atomic.dec.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.cta.shared.dec.u64
  %tmp479 = tail call i64 @llvm.nvvm.atomic.dec.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.cta.dec.u64
  %tmp480 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.release.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.global.dec.u64
  %tmp481 = tail call i64 @llvm.nvvm.atomic.dec.global.i.release.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.cta.shared.dec.u64
  %tmp482 = tail call i64 @llvm.nvvm.atomic.dec.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.cta.dec.u64
  %tmp483 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.acq.rel.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.global.dec.u64
  %tmp484 = tail call i64 @llvm.nvvm.atomic.dec.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.cta.shared.dec.u64
  %tmp485 = tail call i64 @llvm.nvvm.atomic.dec.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.and.b32
  %tmp486 = tail call i32 @llvm.nvvm.atomic.and.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.global.and.b32
  %tmp487 = tail call i32 @llvm.nvvm.atomic.and.global.i.acquire.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.shared.and.b32
  %tmp488 = tail call i32 @llvm.nvvm.atomic.and.shared.i.acquire.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.and.b32
  %tmp489 = tail call i32 @llvm.nvvm.atomic.and.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.global.and.b32
  %tmp490 = tail call i32 @llvm.nvvm.atomic.and.global.i.release.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.shared.and.b32
  %tmp491 = tail call i32 @llvm.nvvm.atomic.and.shared.i.release.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.and.b32
  %tmp492 = tail call i32 @llvm.nvvm.atomic.and.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.global.and.b32
  %tmp493 = tail call i32 @llvm.nvvm.atomic.and.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.shared.and.b32
  %tmp494 = tail call i32 @llvm.nvvm.atomic.and.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.sys.and.b32
  %tmp495 = tail call i32 @llvm.nvvm.atomic.and.gen.i.acquire.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.global.and.b32
  %tmp496 = tail call i32 @llvm.nvvm.atomic.and.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.sys.shared.and.b32
  %tmp497 = tail call i32 @llvm.nvvm.atomic.and.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.sys.and.b32
  %tmp498 = tail call i32 @llvm.nvvm.atomic.and.gen.i.release.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.global.and.b32
  %tmp499 = tail call i32 @llvm.nvvm.atomic.and.global.i.release.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.sys.shared.and.b32
  %tmp500 = tail call i32 @llvm.nvvm.atomic.and.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.sys.and.b32
  %tmp501 = tail call i32 @llvm.nvvm.atomic.and.gen.i.acq.rel.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.global.and.b32
  %tmp502 = tail call i32 @llvm.nvvm.atomic.and.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.sys.shared.and.b32
  %tmp503 = tail call i32 @llvm.nvvm.atomic.and.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.cta.and.b32
  %tmp504 = tail call i32 @llvm.nvvm.atomic.and.gen.i.acquire.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.global.and.b32
  %tmp505 = tail call i32 @llvm.nvvm.atomic.and.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.cta.shared.and.b32
  %tmp506 = tail call i32 @llvm.nvvm.atomic.and.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.cta.and.b32
  %tmp507 = tail call i32 @llvm.nvvm.atomic.and.gen.i.release.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.global.and.b32
  %tmp508 = tail call i32 @llvm.nvvm.atomic.and.global.i.release.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.cta.shared.and.b32
  %tmp509 = tail call i32 @llvm.nvvm.atomic.and.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.cta.and.b32
  %tmp510 = tail call i32 @llvm.nvvm.atomic.and.gen.i.acq.rel.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.global.and.b32
  %tmp511 = tail call i32 @llvm.nvvm.atomic.and.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.cta.shared.and.b32
  %tmp512 = tail call i32 @llvm.nvvm.atomic.and.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.and.b64
  %tmp513 = tail call i64 @llvm.nvvm.atomic.and.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.global.and.b64
  %tmp514 = tail call i64 @llvm.nvvm.atomic.and.global.i.acquire.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.shared.and.b64
  %tmp515 = tail call i64 @llvm.nvvm.atomic.and.shared.i.acquire.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.and.b64
  %tmp516 = tail call i64 @llvm.nvvm.atomic.and.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.global.and.b64
  %tmp517 = tail call i64 @llvm.nvvm.atomic.and.global.i.release.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.shared.and.b64
  %tmp518 = tail call i64 @llvm.nvvm.atomic.and.shared.i.release.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.and.b64
  %tmp519 = tail call i64 @llvm.nvvm.atomic.and.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.global.and.b64
  %tmp520 = tail call i64 @llvm.nvvm.atomic.and.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.shared.and.b64
  %tmp521 = tail call i64 @llvm.nvvm.atomic.and.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.sys.and.b64
  %tmp522 = tail call i64 @llvm.nvvm.atomic.and.gen.i.acquire.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.global.and.b64
  %tmp523 = tail call i64 @llvm.nvvm.atomic.and.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.sys.shared.and.b64
  %tmp524 = tail call i64 @llvm.nvvm.atomic.and.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.sys.and.b64
  %tmp525 = tail call i64 @llvm.nvvm.atomic.and.gen.i.release.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.global.and.b64
  %tmp526 = tail call i64 @llvm.nvvm.atomic.and.global.i.release.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.sys.shared.and.b64
  %tmp527 = tail call i64 @llvm.nvvm.atomic.and.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.sys.and.b64
  %tmp528 = tail call i64 @llvm.nvvm.atomic.and.gen.i.acq.rel.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.global.and.b64
  %tmp529 = tail call i64 @llvm.nvvm.atomic.and.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.sys.shared.and.b64
  %tmp530 = tail call i64 @llvm.nvvm.atomic.and.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.cta.and.b64
  %tmp531 = tail call i64 @llvm.nvvm.atomic.and.gen.i.acquire.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.global.and.b64
  %tmp532 = tail call i64 @llvm.nvvm.atomic.and.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.cta.shared.and.b64
  %tmp533 = tail call i64 @llvm.nvvm.atomic.and.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.cta.and.b64
  %tmp534 = tail call i64 @llvm.nvvm.atomic.and.gen.i.release.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.global.and.b64
  %tmp535 = tail call i64 @llvm.nvvm.atomic.and.global.i.release.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.cta.shared.and.b64
  %tmp536 = tail call i64 @llvm.nvvm.atomic.and.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.cta.and.b64
  %tmp537 = tail call i64 @llvm.nvvm.atomic.and.gen.i.acq.rel.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.global.and.b64
  %tmp538 = tail call i64 @llvm.nvvm.atomic.and.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.cta.shared.and.b64
  %tmp539 = tail call i64 @llvm.nvvm.atomic.and.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.or.b32
  %tmp540 = tail call i32 @llvm.nvvm.atomic.or.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.global.or.b32
  %tmp541 = tail call i32 @llvm.nvvm.atomic.or.global.i.acquire.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.shared.or.b32
  %tmp542 = tail call i32 @llvm.nvvm.atomic.or.shared.i.acquire.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.or.b32
  %tmp543 = tail call i32 @llvm.nvvm.atomic.or.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.global.or.b32
  %tmp544 = tail call i32 @llvm.nvvm.atomic.or.global.i.release.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.shared.or.b32
  %tmp545 = tail call i32 @llvm.nvvm.atomic.or.shared.i.release.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.or.b32
  %tmp546 = tail call i32 @llvm.nvvm.atomic.or.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.global.or.b32
  %tmp547 = tail call i32 @llvm.nvvm.atomic.or.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.shared.or.b32
  %tmp548 = tail call i32 @llvm.nvvm.atomic.or.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.sys.or.b32
  %tmp549 = tail call i32 @llvm.nvvm.atomic.or.gen.i.acquire.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.global.or.b32
  %tmp550 = tail call i32 @llvm.nvvm.atomic.or.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.sys.shared.or.b32
  %tmp551 = tail call i32 @llvm.nvvm.atomic.or.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.sys.or.b32
  %tmp552 = tail call i32 @llvm.nvvm.atomic.or.gen.i.release.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.global.or.b32
  %tmp553 = tail call i32 @llvm.nvvm.atomic.or.global.i.release.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.sys.shared.or.b32
  %tmp554 = tail call i32 @llvm.nvvm.atomic.or.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.sys.or.b32
  %tmp555 = tail call i32 @llvm.nvvm.atomic.or.gen.i.acq.rel.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.global.or.b32
  %tmp556 = tail call i32 @llvm.nvvm.atomic.or.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.sys.shared.or.b32
  %tmp557 = tail call i32 @llvm.nvvm.atomic.or.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.cta.or.b32
  %tmp558 = tail call i32 @llvm.nvvm.atomic.or.gen.i.acquire.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.global.or.b32
  %tmp559 = tail call i32 @llvm.nvvm.atomic.or.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.cta.shared.or.b32
  %tmp560 = tail call i32 @llvm.nvvm.atomic.or.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.cta.or.b32
  %tmp561 = tail call i32 @llvm.nvvm.atomic.or.gen.i.release.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.global.or.b32
  %tmp562 = tail call i32 @llvm.nvvm.atomic.or.global.i.release.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.cta.shared.or.b32
  %tmp563 = tail call i32 @llvm.nvvm.atomic.or.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.cta.or.b32
  %tmp564 = tail call i32 @llvm.nvvm.atomic.or.gen.i.acq.rel.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.global.or.b32
  %tmp565 = tail call i32 @llvm.nvvm.atomic.or.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.cta.shared.or.b32
  %tmp566 = tail call i32 @llvm.nvvm.atomic.or.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.or.b64
  %tmp567 = tail call i64 @llvm.nvvm.atomic.or.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.global.or.b64
  %tmp568 = tail call i64 @llvm.nvvm.atomic.or.global.i.acquire.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.shared.or.b64
  %tmp569 = tail call i64 @llvm.nvvm.atomic.or.shared.i.acquire.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.or.b64
  %tmp570 = tail call i64 @llvm.nvvm.atomic.or.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.global.or.b64
  %tmp571 = tail call i64 @llvm.nvvm.atomic.or.global.i.release.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.shared.or.b64
  %tmp572 = tail call i64 @llvm.nvvm.atomic.or.shared.i.release.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.or.b64
  %tmp573 = tail call i64 @llvm.nvvm.atomic.or.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.global.or.b64
  %tmp574 = tail call i64 @llvm.nvvm.atomic.or.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.shared.or.b64
  %tmp575 = tail call i64 @llvm.nvvm.atomic.or.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.sys.or.b64
  %tmp576 = tail call i64 @llvm.nvvm.atomic.or.gen.i.acquire.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.global.or.b64
  %tmp577 = tail call i64 @llvm.nvvm.atomic.or.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.sys.shared.or.b64
  %tmp578 = tail call i64 @llvm.nvvm.atomic.or.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.sys.or.b64
  %tmp579 = tail call i64 @llvm.nvvm.atomic.or.gen.i.release.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.global.or.b64
  %tmp580 = tail call i64 @llvm.nvvm.atomic.or.global.i.release.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.sys.shared.or.b64
  %tmp581 = tail call i64 @llvm.nvvm.atomic.or.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.sys.or.b64
  %tmp582 = tail call i64 @llvm.nvvm.atomic.or.gen.i.acq.rel.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.global.or.b64
  %tmp583 = tail call i64 @llvm.nvvm.atomic.or.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.sys.shared.or.b64
  %tmp584 = tail call i64 @llvm.nvvm.atomic.or.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.cta.or.b64
  %tmp585 = tail call i64 @llvm.nvvm.atomic.or.gen.i.acquire.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.global.or.b64
  %tmp586 = tail call i64 @llvm.nvvm.atomic.or.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.cta.shared.or.b64
  %tmp587 = tail call i64 @llvm.nvvm.atomic.or.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.cta.or.b64
  %tmp588 = tail call i64 @llvm.nvvm.atomic.or.gen.i.release.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.global.or.b64
  %tmp589 = tail call i64 @llvm.nvvm.atomic.or.global.i.release.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.cta.shared.or.b64
  %tmp590 = tail call i64 @llvm.nvvm.atomic.or.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.cta.or.b64
  %tmp591 = tail call i64 @llvm.nvvm.atomic.or.gen.i.acq.rel.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.global.or.b64
  %tmp592 = tail call i64 @llvm.nvvm.atomic.or.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.cta.shared.or.b64
  %tmp593 = tail call i64 @llvm.nvvm.atomic.or.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.xor.b32
  %tmp594 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.global.xor.b32
  %tmp595 = tail call i32 @llvm.nvvm.atomic.xor.global.i.acquire.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.shared.xor.b32
  %tmp596 = tail call i32 @llvm.nvvm.atomic.xor.shared.i.acquire.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.xor.b32
  %tmp597 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.global.xor.b32
  %tmp598 = tail call i32 @llvm.nvvm.atomic.xor.global.i.release.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.shared.xor.b32
  %tmp599 = tail call i32 @llvm.nvvm.atomic.xor.shared.i.release.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.xor.b32
  %tmp600 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.global.xor.b32
  %tmp601 = tail call i32 @llvm.nvvm.atomic.xor.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.shared.xor.b32
  %tmp602 = tail call i32 @llvm.nvvm.atomic.xor.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.sys.xor.b32
  %tmp603 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.acquire.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.global.xor.b32
  %tmp604 = tail call i32 @llvm.nvvm.atomic.xor.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.sys.shared.xor.b32
  %tmp605 = tail call i32 @llvm.nvvm.atomic.xor.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.sys.xor.b32
  %tmp606 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.release.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.global.xor.b32
  %tmp607 = tail call i32 @llvm.nvvm.atomic.xor.global.i.release.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.sys.shared.xor.b32
  %tmp608 = tail call i32 @llvm.nvvm.atomic.xor.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.sys.xor.b32
  %tmp609 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.acq.rel.sys.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.global.xor.b32
  %tmp610 = tail call i32 @llvm.nvvm.atomic.xor.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.sys.shared.xor.b32
  %tmp611 = tail call i32 @llvm.nvvm.atomic.xor.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.cta.xor.b32
  %tmp612 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.acquire.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.global.xor.b32
  %tmp613 = tail call i32 @llvm.nvvm.atomic.xor.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acquire.cta.shared.xor.b32
  %tmp614 = tail call i32 @llvm.nvvm.atomic.xor.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.release.cta.xor.b32
  %tmp615 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.release.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.global.xor.b32
  %tmp616 = tail call i32 @llvm.nvvm.atomic.xor.global.i.release.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.release.cta.shared.xor.b32
  %tmp617 = tail call i32 @llvm.nvvm.atomic.xor.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acq_rel.cta.xor.b32
  %tmp618 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.acq.rel.cta.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.global.xor.b32
  %tmp619 = tail call i32 @llvm.nvvm.atomic.xor.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: atom.acq_rel.cta.shared.xor.b32
  %tmp620 = tail call i32 @llvm.nvvm.atomic.xor.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: atom.acquire.xor.b64
  %tmp621 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.global.xor.b64
  %tmp622 = tail call i64 @llvm.nvvm.atomic.xor.global.i.acquire.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.shared.xor.b64
  %tmp623 = tail call i64 @llvm.nvvm.atomic.xor.shared.i.acquire.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.xor.b64
  %tmp624 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.global.xor.b64
  %tmp625 = tail call i64 @llvm.nvvm.atomic.xor.global.i.release.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.shared.xor.b64
  %tmp626 = tail call i64 @llvm.nvvm.atomic.xor.shared.i.release.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.xor.b64
  %tmp627 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.global.xor.b64
  %tmp628 = tail call i64 @llvm.nvvm.atomic.xor.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.shared.xor.b64
  %tmp629 = tail call i64 @llvm.nvvm.atomic.xor.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.sys.xor.b64
  %tmp630 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.acquire.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.global.xor.b64
  %tmp631 = tail call i64 @llvm.nvvm.atomic.xor.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.sys.shared.xor.b64
  %tmp632 = tail call i64 @llvm.nvvm.atomic.xor.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.sys.xor.b64
  %tmp633 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.release.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.global.xor.b64
  %tmp634 = tail call i64 @llvm.nvvm.atomic.xor.global.i.release.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.sys.shared.xor.b64
  %tmp635 = tail call i64 @llvm.nvvm.atomic.xor.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.sys.xor.b64
  %tmp636 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.acq.rel.sys.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.global.xor.b64
  %tmp637 = tail call i64 @llvm.nvvm.atomic.xor.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.sys.shared.xor.b64
  %tmp638 = tail call i64 @llvm.nvvm.atomic.xor.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.cta.xor.b64
  %tmp639 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.acquire.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.global.xor.b64
  %tmp640 = tail call i64 @llvm.nvvm.atomic.xor.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acquire.cta.shared.xor.b64
  %tmp641 = tail call i64 @llvm.nvvm.atomic.xor.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.release.cta.xor.b64
  %tmp642 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.release.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.global.xor.b64
  %tmp643 = tail call i64 @llvm.nvvm.atomic.xor.global.i.release.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.release.cta.shared.xor.b64
  %tmp644 = tail call i64 @llvm.nvvm.atomic.xor.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acq_rel.cta.xor.b64
  %tmp645 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.acq.rel.cta.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.global.xor.b64
  %tmp646 = tail call i64 @llvm.nvvm.atomic.xor.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: atom.acq_rel.cta.shared.xor.b64
  %tmp647 = tail call i64 @llvm.nvvm.atomic.xor.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: atom.acquire.cas.b32
  %tmp648 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.acquire.global.cas.b32
  %tmp649 = tail call i32 @llvm.nvvm.atomic.cas.global.i.acquire.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i, i32 %i);

  ; CHECK: atom.acquire.shared.cas.b32
  %tmp650 = tail call i32 @llvm.nvvm.atomic.cas.shared.i.acquire.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i, i32 %i);

  ; CHECK: atom.release.cas.b32
  %tmp651 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.release.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.release.global.cas.b32
  %tmp652 = tail call i32 @llvm.nvvm.atomic.cas.global.i.release.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i, i32 %i);

  ; CHECK: atom.release.shared.cas.b32
  %tmp653 = tail call i32 @llvm.nvvm.atomic.cas.shared.i.release.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i, i32 %i);

  ; CHECK: atom.acq_rel.cas.b32
  %tmp654 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.acq_rel.global.cas.b32
  %tmp655 = tail call i32 @llvm.nvvm.atomic.cas.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i, i32 %i);

  ; CHECK: atom.acq_rel.shared.cas.b32
  %tmp656 = tail call i32 @llvm.nvvm.atomic.cas.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i, i32 %i);

  ; CHECK: atom.acquire.sys.cas.b32
  %tmp657 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.acquire.sys.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.acquire.sys.global.cas.b32
  %tmp658 = tail call i32 @llvm.nvvm.atomic.cas.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i, i32 %i);

  ; CHECK: atom.acquire.sys.shared.cas.b32
  %tmp659 = tail call i32 @llvm.nvvm.atomic.cas.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i, i32 %i);

  ; CHECK: atom.release.sys.cas.b32
  %tmp660 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.release.sys.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.release.sys.global.cas.b32
  %tmp661 = tail call i32 @llvm.nvvm.atomic.cas.global.i.release.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i, i32 %i);

  ; CHECK: atom.release.sys.shared.cas.b32
  %tmp662 = tail call i32 @llvm.nvvm.atomic.cas.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i, i32 %i);

  ; CHECK: atom.acq_rel.sys.cas.b32
  %tmp663 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.acq.rel.sys.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.acq_rel.sys.global.cas.b32
  %tmp664 = tail call i32 @llvm.nvvm.atomic.cas.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i, i32 %i);

  ; CHECK: atom.acq_rel.sys.shared.cas.b32
  %tmp665 = tail call i32 @llvm.nvvm.atomic.cas.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i, i32 %i);

  ; CHECK: atom.acquire.cta.cas.b32
  %tmp666 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.acquire.cta.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.acquire.cta.global.cas.b32
  %tmp667 = tail call i32 @llvm.nvvm.atomic.cas.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i, i32 %i);

  ; CHECK: atom.acquire.cta.shared.cas.b32
  %tmp668 = tail call i32 @llvm.nvvm.atomic.cas.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i, i32 %i);

  ; CHECK: atom.release.cta.cas.b32
  %tmp669 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.release.cta.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.release.cta.global.cas.b32
  %tmp670 = tail call i32 @llvm.nvvm.atomic.cas.global.i.release.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i, i32 %i);

  ; CHECK: atom.release.cta.shared.cas.b32
  %tmp671 = tail call i32 @llvm.nvvm.atomic.cas.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i, i32 %i);

  ; CHECK: atom.acq_rel.cta.cas.b32
  %tmp672 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.acq.rel.cta.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.acq_rel.cta.global.cas.b32
  %tmp673 = tail call i32 @llvm.nvvm.atomic.cas.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* %ip1, i32 %i, i32 %i);

  ; CHECK: atom.acq_rel.cta.shared.cas.b32
  %tmp674 = tail call i32 @llvm.nvvm.atomic.cas.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* %ip3, i32 %i, i32 %i);

  ; CHECK: atom.acquire.cas.b64
  %tmp675 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);

  ; CHECK: atom.acquire.global.cas.b64
  %tmp676 = tail call i64 @llvm.nvvm.atomic.cas.global.i.acquire.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll, i64 %ll);

  ; CHECK: atom.acquire.shared.cas.b64
  %tmp677 = tail call i64 @llvm.nvvm.atomic.cas.shared.i.acquire.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll, i64 %ll);

  ; CHECK: atom.release.cas.b64
  %tmp678 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.release.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);

  ; CHECK: atom.release.global.cas.b64
  %tmp679 = tail call i64 @llvm.nvvm.atomic.cas.global.i.release.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll, i64 %ll);

  ; CHECK: atom.release.shared.cas.b64
  %tmp680 = tail call i64 @llvm.nvvm.atomic.cas.shared.i.release.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll, i64 %ll);

  ; CHECK: atom.acq_rel.cas.b64
  %tmp681 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);

  ; CHECK: atom.acq_rel.global.cas.b64
  %tmp682 = tail call i64 @llvm.nvvm.atomic.cas.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll, i64 %ll);

  ; CHECK: atom.acq_rel.shared.cas.b64
  %tmp683 = tail call i64 @llvm.nvvm.atomic.cas.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll, i64 %ll);

  ; CHECK: atom.acquire.sys.cas.b64
  %tmp684 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.acquire.sys.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);

  ; CHECK: atom.acquire.sys.global.cas.b64
  %tmp685 = tail call i64 @llvm.nvvm.atomic.cas.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll, i64 %ll);

  ; CHECK: atom.acquire.sys.shared.cas.b64
  %tmp686 = tail call i64 @llvm.nvvm.atomic.cas.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll, i64 %ll);

  ; CHECK: atom.release.sys.cas.b64
  %tmp687 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.release.sys.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);

  ; CHECK: atom.release.sys.global.cas.b64
  %tmp688 = tail call i64 @llvm.nvvm.atomic.cas.global.i.release.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll, i64 %ll);

  ; CHECK: atom.release.sys.shared.cas.b64
  %tmp689 = tail call i64 @llvm.nvvm.atomic.cas.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll, i64 %ll);

  ; CHECK: atom.acq_rel.sys.cas.b64
  %tmp690 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.acq.rel.sys.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);

  ; CHECK: atom.acq_rel.sys.global.cas.b64
  %tmp691 = tail call i64 @llvm.nvvm.atomic.cas.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll, i64 %ll);

  ; CHECK: atom.acq_rel.sys.shared.cas.b64
  %tmp692 = tail call i64 @llvm.nvvm.atomic.cas.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll, i64 %ll);

  ; CHECK: atom.acquire.cta.cas.b64
  %tmp693 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.acquire.cta.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);

  ; CHECK: atom.acquire.cta.global.cas.b64
  %tmp694 = tail call i64 @llvm.nvvm.atomic.cas.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll, i64 %ll);

  ; CHECK: atom.acquire.cta.shared.cas.b64
  %tmp695 = tail call i64 @llvm.nvvm.atomic.cas.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll, i64 %ll);

  ; CHECK: atom.release.cta.cas.b64
  %tmp696 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.release.cta.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);

  ; CHECK: atom.release.cta.global.cas.b64
  %tmp697 = tail call i64 @llvm.nvvm.atomic.cas.global.i.release.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll, i64 %ll);

  ; CHECK: atom.release.cta.shared.cas.b64
  %tmp698 = tail call i64 @llvm.nvvm.atomic.cas.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll, i64 %ll);

  ; CHECK: atom.acq_rel.cta.cas.b64
  %tmp699 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.acq.rel.cta.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);

  ; CHECK: atom.acq_rel.cta.global.cas.b64
  %tmp700 = tail call i64 @llvm.nvvm.atomic.cas.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* %llp1, i64 %ll, i64 %ll);

  ; CHECK: atom.acq_rel.cta.shared.cas.b64
  %tmp701 = tail call i64 @llvm.nvvm.atomic.cas.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* %llp3, i64 %ll, i64 %ll);

  ; CHECK: ld.relaxed.gpu.s32
  %tmpldst0 = tail call i32 @llvm.nvvm.ld.gen.i.i32.p0(i32* %ip);

  ; CHECK: ld.relaxed.gpu.global.s32
  %tmpldst1 = tail call i32 @llvm.nvvm.ld.global.i.i32.p1(i32 addrspace(1)* %ip1);

  ; CHECK: ld.relaxed.gpu.shared.s32
  %tmpldst2 = tail call i32 @llvm.nvvm.ld.shared.i.i32.p3(i32 addrspace(3)* %ip3);

  ; CHECK: ld.acquire.gpu.s32
  %tmpldst3 = tail call i32 @llvm.nvvm.ld.gen.i.acquire.i32.p0(i32* %ip);

  ; CHECK: ld.acquire.gpu.global.s32
  %tmpldst4 = tail call i32 @llvm.nvvm.ld.global.i.acquire.i32.p1(i32 addrspace(1)* %ip1);

  ; CHECK: ld.acquire.gpu.shared.s32
  %tmpldst5 = tail call i32 @llvm.nvvm.ld.shared.i.acquire.i32.p3(i32 addrspace(3)* %ip3);

  ; CHECK: ld.relaxed.sys.s32
  %tmpldst6 = tail call i32 @llvm.nvvm.ld.gen.i.sys.i32.p0(i32* %ip);

  ; CHECK: ld.relaxed.sys.global.s32
  %tmpldst7 = tail call i32 @llvm.nvvm.ld.global.i.sys.i32.p1(i32 addrspace(1)* %ip1);

  ; CHECK: ld.relaxed.sys.shared.s32
  %tmpldst8 = tail call i32 @llvm.nvvm.ld.shared.i.sys.i32.p3(i32 addrspace(3)* %ip3);

  ; CHECK: ld.acquire.sys.s32
  %tmpldst9 = tail call i32 @llvm.nvvm.ld.gen.i.acquire.sys.i32.p0(i32* %ip);

  ; CHECK: ld.acquire.sys.global.s32
  %tmpldst10 = tail call i32 @llvm.nvvm.ld.global.i.acquire.sys.i32.p1(i32 addrspace(1)* %ip1);

  ; CHECK: ld.acquire.sys.shared.s32
  %tmpldst11 = tail call i32 @llvm.nvvm.ld.shared.i.acquire.sys.i32.p3(i32 addrspace(3)* %ip3);

  ; CHECK: ld.relaxed.cta.s32
  %tmpldst12 = tail call i32 @llvm.nvvm.ld.gen.i.cta.i32.p0(i32* %ip);

  ; CHECK: ld.relaxed.cta.global.s32
  %tmpldst13 = tail call i32 @llvm.nvvm.ld.global.i.cta.i32.p1(i32 addrspace(1)* %ip1);

  ; CHECK: ld.relaxed.cta.shared.s32
  %tmpldst14 = tail call i32 @llvm.nvvm.ld.shared.i.cta.i32.p3(i32 addrspace(3)* %ip3);

  ; CHECK: ld.acquire.cta.s32
  %tmpldst15 = tail call i32 @llvm.nvvm.ld.gen.i.acquire.cta.i32.p0(i32* %ip);

  ; CHECK: ld.acquire.cta.global.s32
  %tmpldst16 = tail call i32 @llvm.nvvm.ld.global.i.acquire.cta.i32.p1(i32 addrspace(1)* %ip1);

  ; CHECK: ld.acquire.cta.shared.s32
  %tmpldst17 = tail call i32 @llvm.nvvm.ld.shared.i.acquire.cta.i32.p3(i32 addrspace(3)* %ip3);

  ; CHECK: ld.relaxed.gpu.s64
  %tmpldst18 = tail call i64 @llvm.nvvm.ld.gen.i.i64.p0(i64* %llp);

  ; CHECK: ld.relaxed.gpu.global.s64
  %tmpldst19 = tail call i64 @llvm.nvvm.ld.global.i.i64.p1(i64 addrspace(1)* %llp1);

  ; CHECK: ld.relaxed.gpu.shared.s64
  %tmpldst20 = tail call i64 @llvm.nvvm.ld.shared.i.i64.p3(i64 addrspace(3)* %llp3);

  ; CHECK: ld.acquire.gpu.s64
  %tmpldst21 = tail call i64 @llvm.nvvm.ld.gen.i.acquire.i64.p0(i64* %llp);

  ; CHECK: ld.acquire.gpu.global.s64
  %tmpldst22 = tail call i64 @llvm.nvvm.ld.global.i.acquire.i64.p1(i64 addrspace(1)* %llp1);

  ; CHECK: ld.acquire.gpu.shared.s64
  %tmpldst23 = tail call i64 @llvm.nvvm.ld.shared.i.acquire.i64.p3(i64 addrspace(3)* %llp3);

  ; CHECK: ld.relaxed.sys.s64
  %tmpldst24 = tail call i64 @llvm.nvvm.ld.gen.i.sys.i64.p0(i64* %llp);

  ; CHECK: ld.relaxed.sys.global.s64
  %tmpldst25 = tail call i64 @llvm.nvvm.ld.global.i.sys.i64.p1(i64 addrspace(1)* %llp1);

  ; CHECK: ld.relaxed.sys.shared.s64
  %tmpldst26 = tail call i64 @llvm.nvvm.ld.shared.i.sys.i64.p3(i64 addrspace(3)* %llp3);

  ; CHECK: ld.acquire.sys.s64
  %tmpldst27 = tail call i64 @llvm.nvvm.ld.gen.i.acquire.sys.i64.p0(i64* %llp);

  ; CHECK: ld.acquire.sys.global.s64
  %tmpldst28 = tail call i64 @llvm.nvvm.ld.global.i.acquire.sys.i64.p1(i64 addrspace(1)* %llp1);

  ; CHECK: ld.acquire.sys.shared.s64
  %tmpldst29 = tail call i64 @llvm.nvvm.ld.shared.i.acquire.sys.i64.p3(i64 addrspace(3)* %llp3);

  ; CHECK: ld.relaxed.cta.s64
  %tmpldst30 = tail call i64 @llvm.nvvm.ld.gen.i.cta.i64.p0(i64* %llp);

  ; CHECK: ld.relaxed.cta.global.s64
  %tmpldst31 = tail call i64 @llvm.nvvm.ld.global.i.cta.i64.p1(i64 addrspace(1)* %llp1);

  ; CHECK: ld.relaxed.cta.shared.s64
  %tmpldst32 = tail call i64 @llvm.nvvm.ld.shared.i.cta.i64.p3(i64 addrspace(3)* %llp3);

  ; CHECK: ld.acquire.cta.s64
  %tmpldst33 = tail call i64 @llvm.nvvm.ld.gen.i.acquire.cta.i64.p0(i64* %llp);

  ; CHECK: ld.acquire.cta.global.s64
  %tmpldst34 = tail call i64 @llvm.nvvm.ld.global.i.acquire.cta.i64.p1(i64 addrspace(1)* %llp1);

  ; CHECK: ld.acquire.cta.shared.s64
  %tmpldst35 = tail call i64 @llvm.nvvm.ld.shared.i.acquire.cta.i64.p3(i64 addrspace(3)* %llp3);

  ; CHECK: ld.relaxed.gpu.f32
  %tmpldst36 = tail call float @llvm.nvvm.ld.gen.f.f32.p0(float* %fp);

  ; CHECK: ld.relaxed.gpu.global.f32
  %tmpldst37 = tail call float @llvm.nvvm.ld.global.f.f32.p1(float addrspace(1)* %fp1);

  ; CHECK: ld.relaxed.gpu.shared.f32
  %tmpldst38 = tail call float @llvm.nvvm.ld.shared.f.f32.p3(float addrspace(3)* %fp3);

  ; CHECK: ld.acquire.gpu.f32
  %tmpldst39 = tail call float @llvm.nvvm.ld.gen.f.acquire.f32.p0(float* %fp);

  ; CHECK: ld.acquire.gpu.global.f32
  %tmpldst40 = tail call float @llvm.nvvm.ld.global.f.acquire.f32.p1(float addrspace(1)* %fp1);

  ; CHECK: ld.acquire.gpu.shared.f32
  %tmpldst41 = tail call float @llvm.nvvm.ld.shared.f.acquire.f32.p3(float addrspace(3)* %fp3);

  ; CHECK: ld.relaxed.sys.f32
  %tmpldst42 = tail call float @llvm.nvvm.ld.gen.f.sys.f32.p0(float* %fp);

  ; CHECK: ld.relaxed.sys.global.f32
  %tmpldst43 = tail call float @llvm.nvvm.ld.global.f.sys.f32.p1(float addrspace(1)* %fp1);

  ; CHECK: ld.relaxed.sys.shared.f32
  %tmpldst44 = tail call float @llvm.nvvm.ld.shared.f.sys.f32.p3(float addrspace(3)* %fp3);

  ; CHECK: ld.acquire.sys.f32
  %tmpldst45 = tail call float @llvm.nvvm.ld.gen.f.acquire.sys.f32.p0(float* %fp);

  ; CHECK: ld.acquire.sys.global.f32
  %tmpldst46 = tail call float @llvm.nvvm.ld.global.f.acquire.sys.f32.p1(float addrspace(1)* %fp1);

  ; CHECK: ld.acquire.sys.shared.f32
  %tmpldst47 = tail call float @llvm.nvvm.ld.shared.f.acquire.sys.f32.p3(float addrspace(3)* %fp3);

  ; CHECK: ld.relaxed.cta.f32
  %tmpldst48 = tail call float @llvm.nvvm.ld.gen.f.cta.f32.p0(float* %fp);

  ; CHECK: ld.relaxed.cta.global.f32
  %tmpldst49 = tail call float @llvm.nvvm.ld.global.f.cta.f32.p1(float addrspace(1)* %fp1);

  ; CHECK: ld.relaxed.cta.shared.f32
  %tmpldst50 = tail call float @llvm.nvvm.ld.shared.f.cta.f32.p3(float addrspace(3)* %fp3);

  ; CHECK: ld.acquire.cta.f32
  %tmpldst51 = tail call float @llvm.nvvm.ld.gen.f.acquire.cta.f32.p0(float* %fp);

  ; CHECK: ld.acquire.cta.global.f32
  %tmpldst52 = tail call float @llvm.nvvm.ld.global.f.acquire.cta.f32.p1(float addrspace(1)* %fp1);

  ; CHECK: ld.acquire.cta.shared.f32
  %tmpldst53 = tail call float @llvm.nvvm.ld.shared.f.acquire.cta.f32.p3(float addrspace(3)* %fp3);

  ; CHECK: ld.relaxed.gpu.f64
  %tmpldst54 = tail call double @llvm.nvvm.ld.gen.f.f64.p0(double* %dfp);

  ; CHECK: ld.relaxed.gpu.global.f64
  %tmpldst55 = tail call double @llvm.nvvm.ld.global.f.f64.p1(double addrspace(1)* %dfp1);

  ; CHECK: ld.relaxed.gpu.shared.f64
  %tmpldst56 = tail call double @llvm.nvvm.ld.shared.f.f64.p3(double addrspace(3)* %dfp3);

  ; CHECK: ld.acquire.gpu.f64
  %tmpldst57 = tail call double @llvm.nvvm.ld.gen.f.acquire.f64.p0(double* %dfp);

  ; CHECK: ld.acquire.gpu.global.f64
  %tmpldst58 = tail call double @llvm.nvvm.ld.global.f.acquire.f64.p1(double addrspace(1)* %dfp1);

  ; CHECK: ld.acquire.gpu.shared.f64
  %tmpldst59 = tail call double @llvm.nvvm.ld.shared.f.acquire.f64.p3(double addrspace(3)* %dfp3);

  ; CHECK: ld.relaxed.sys.f64
  %tmpldst60 = tail call double @llvm.nvvm.ld.gen.f.sys.f64.p0(double* %dfp);

  ; CHECK: ld.relaxed.sys.global.f64
  %tmpldst61 = tail call double @llvm.nvvm.ld.global.f.sys.f64.p1(double addrspace(1)* %dfp1);

  ; CHECK: ld.relaxed.sys.shared.f64
  %tmpldst62 = tail call double @llvm.nvvm.ld.shared.f.sys.f64.p3(double addrspace(3)* %dfp3);

  ; CHECK: ld.acquire.sys.f64
  %tmpldst63 = tail call double @llvm.nvvm.ld.gen.f.acquire.sys.f64.p0(double* %dfp);

  ; CHECK: ld.acquire.sys.global.f64
  %tmpldst64 = tail call double @llvm.nvvm.ld.global.f.acquire.sys.f64.p1(double addrspace(1)* %dfp1);

  ; CHECK: ld.acquire.sys.shared.f64
  %tmpldst65 = tail call double @llvm.nvvm.ld.shared.f.acquire.sys.f64.p3(double addrspace(3)* %dfp3);

  ; CHECK: ld.relaxed.cta.f64
  %tmpldst66 = tail call double @llvm.nvvm.ld.gen.f.cta.f64.p0(double* %dfp);

  ; CHECK: ld.relaxed.cta.global.f64
  %tmpldst67 = tail call double @llvm.nvvm.ld.global.f.cta.f64.p1(double addrspace(1)* %dfp1);

  ; CHECK: ld.relaxed.cta.shared.f64
  %tmpldst68 = tail call double @llvm.nvvm.ld.shared.f.cta.f64.p3(double addrspace(3)* %dfp3);

  ; CHECK: ld.acquire.cta.f64
  %tmpldst69 = tail call double @llvm.nvvm.ld.gen.f.acquire.cta.f64.p0(double* %dfp);

  ; CHECK: ld.acquire.cta.global.f64
  %tmpldst70 = tail call double @llvm.nvvm.ld.global.f.acquire.cta.f64.p1(double addrspace(1)* %dfp1);

  ; CHECK: ld.acquire.cta.shared.f64
  %tmpldst71 = tail call double @llvm.nvvm.ld.shared.f.acquire.cta.f64.p3(double addrspace(3)* %dfp3);

  ; CHECK: st.relaxed.gpu.s32
  tail call void @llvm.nvvm.st.gen.i.p0i32.i32(i32* %ip, i32 %i);

  ; CHECK: st.relaxed.gpu.global.s32
  tail call void @llvm.nvvm.st.global.i.p1i32.i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: st.relaxed.gpu.shared.s32
  tail call void @llvm.nvvm.st.shared.i.p3i32.i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: st.release.gpu.s32
  tail call void @llvm.nvvm.st.gen.i.release.p0i32.i32(i32* %ip, i32 %i);

  ; CHECK: st.release.gpu.global.s32
  tail call void @llvm.nvvm.st.global.i.release.p1i32.i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: st.release.gpu.shared.s32
  tail call void @llvm.nvvm.st.shared.i.release.p3i32.i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: st.relaxed.sys.s32
  tail call void @llvm.nvvm.st.gen.i.sys.p0i32.i32(i32* %ip, i32 %i);

  ; CHECK: st.relaxed.sys.global.s32
  tail call void @llvm.nvvm.st.global.i.sys.p1i32.i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: st.relaxed.sys.shared.s32
  tail call void @llvm.nvvm.st.shared.i.sys.p3i32.i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: st.release.sys.s32
  tail call void @llvm.nvvm.st.gen.i.release.sys.p0i32.i32(i32* %ip, i32 %i);

  ; CHECK: st.release.sys.global.s32
  tail call void @llvm.nvvm.st.global.i.release.sys.p1i32.i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: st.release.sys.shared.s32
  tail call void @llvm.nvvm.st.shared.i.release.sys.p3i32.i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: st.relaxed.cta.s32
  tail call void @llvm.nvvm.st.gen.i.cta.p0i32.i32(i32* %ip, i32 %i);

  ; CHECK: st.relaxed.cta.global.s32
  tail call void @llvm.nvvm.st.global.i.cta.p1i32.i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: st.relaxed.cta.shared.s32
  tail call void @llvm.nvvm.st.shared.i.cta.p3i32.i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: st.release.cta.s32
  tail call void @llvm.nvvm.st.gen.i.release.cta.p0i32.i32(i32* %ip, i32 %i);

  ; CHECK: st.release.cta.global.s32
  tail call void @llvm.nvvm.st.global.i.release.cta.p1i32.i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: st.release.cta.shared.s32
  tail call void @llvm.nvvm.st.shared.i.release.cta.p3i32.i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: st.relaxed.gpu.s64
  tail call void @llvm.nvvm.st.gen.i.p0i64.i64(i64* %llp, i64 %ll);

  ; CHECK: st.relaxed.gpu.global.s64
  tail call void @llvm.nvvm.st.global.i.p1i64.i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: st.relaxed.gpu.shared.s64
  tail call void @llvm.nvvm.st.shared.i.p3i64.i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: st.release.gpu.s64
  tail call void @llvm.nvvm.st.gen.i.release.p0i64.i64(i64* %llp, i64 %ll);

  ; CHECK: st.release.gpu.global.s64
  tail call void @llvm.nvvm.st.global.i.release.p1i64.i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: st.release.gpu.shared.s64
  tail call void @llvm.nvvm.st.shared.i.release.p3i64.i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: st.relaxed.sys.s64
  tail call void @llvm.nvvm.st.gen.i.sys.p0i64.i64(i64* %llp, i64 %ll);

  ; CHECK: st.relaxed.sys.global.s64
  tail call void @llvm.nvvm.st.global.i.sys.p1i64.i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: st.relaxed.sys.shared.s64
  tail call void @llvm.nvvm.st.shared.i.sys.p3i64.i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: st.release.sys.s64
  tail call void @llvm.nvvm.st.gen.i.release.sys.p0i64.i64(i64* %llp, i64 %ll);

  ; CHECK: st.release.sys.global.s64
  tail call void @llvm.nvvm.st.global.i.release.sys.p1i64.i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: st.release.sys.shared.s64
  tail call void @llvm.nvvm.st.shared.i.release.sys.p3i64.i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: st.relaxed.cta.s64
  tail call void @llvm.nvvm.st.gen.i.cta.p0i64.i64(i64* %llp, i64 %ll);

  ; CHECK: st.relaxed.cta.global.s64
  tail call void @llvm.nvvm.st.global.i.cta.p1i64.i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: st.relaxed.cta.shared.s64
  tail call void @llvm.nvvm.st.shared.i.cta.p3i64.i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: st.release.cta.s64
  tail call void @llvm.nvvm.st.gen.i.release.cta.p0i64.i64(i64* %llp, i64 %ll);

  ; CHECK: st.release.cta.global.s64
  tail call void @llvm.nvvm.st.global.i.release.cta.p1i64.i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: st.release.cta.shared.s64
  tail call void @llvm.nvvm.st.shared.i.release.cta.p3i64.i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: st.relaxed.gpu.f32
  tail call void @llvm.nvvm.st.gen.f.p0f32.f32(float* %fp, float %f);

  ; CHECK: st.relaxed.gpu.global.f32
  tail call void @llvm.nvvm.st.global.f.p1f32.f32(float addrspace(1)* %fp1, float %f);

  ; CHECK: st.relaxed.gpu.shared.f32
  tail call void @llvm.nvvm.st.shared.f.p3f32.f32(float addrspace(3)* %fp3, float %f);

  ; CHECK: st.release.gpu.f32
  tail call void @llvm.nvvm.st.gen.f.release.p0f32.f32(float* %fp, float %f);

  ; CHECK: st.release.gpu.global.f32
  tail call void @llvm.nvvm.st.global.f.release.p1f32.f32(float addrspace(1)* %fp1, float %f);

  ; CHECK: st.release.gpu.shared.f32
  tail call void @llvm.nvvm.st.shared.f.release.p3f32.f32(float addrspace(3)* %fp3, float %f);

  ; CHECK: st.relaxed.sys.f32
  tail call void @llvm.nvvm.st.gen.f.sys.p0f32.f32(float* %fp, float %f);

  ; CHECK: st.relaxed.sys.global.f32
  tail call void @llvm.nvvm.st.global.f.sys.p1f32.f32(float addrspace(1)* %fp1, float %f);

  ; CHECK: st.relaxed.sys.shared.f32
  tail call void @llvm.nvvm.st.shared.f.sys.p3f32.f32(float addrspace(3)* %fp3, float %f);

  ; CHECK: st.release.sys.f32
  tail call void @llvm.nvvm.st.gen.f.release.sys.p0f32.f32(float* %fp, float %f);

  ; CHECK: st.release.sys.global.f32
  tail call void @llvm.nvvm.st.global.f.release.sys.p1f32.f32(float addrspace(1)* %fp1, float %f);

  ; CHECK: st.release.sys.shared.f32
  tail call void @llvm.nvvm.st.shared.f.release.sys.p3f32.f32(float addrspace(3)* %fp3, float %f);

  ; CHECK: st.relaxed.cta.f32
  tail call void @llvm.nvvm.st.gen.f.cta.p0f32.f32(float* %fp, float %f);

  ; CHECK: st.relaxed.cta.global.f32
  tail call void @llvm.nvvm.st.global.f.cta.p1f32.f32(float addrspace(1)* %fp1, float %f);

  ; CHECK: st.relaxed.cta.shared.f32
  tail call void @llvm.nvvm.st.shared.f.cta.p3f32.f32(float addrspace(3)* %fp3, float %f);

  ; CHECK: st.release.cta.f32
  tail call void @llvm.nvvm.st.gen.f.release.cta.p0f32.f32(float* %fp, float %f);

  ; CHECK: st.release.cta.global.f32
  tail call void @llvm.nvvm.st.global.f.release.cta.p1f32.f32(float addrspace(1)* %fp1, float %f);

  ; CHECK: st.release.cta.shared.f32
  tail call void @llvm.nvvm.st.shared.f.release.cta.p3f32.f32(float addrspace(3)* %fp3, float %f);

  ; CHECK: st.relaxed.gpu.f64
  tail call void @llvm.nvvm.st.gen.f.p0f64.f64(double* %dfp, double %df);

  ; CHECK: st.relaxed.gpu.global.f64
  tail call void @llvm.nvvm.st.global.f.p1f64.f64(double addrspace(1)* %dfp1, double %df);

  ; CHECK: st.relaxed.gpu.shared.f64
  tail call void @llvm.nvvm.st.shared.f.p3f64.f64(double addrspace(3)* %dfp3, double %df);

  ; CHECK: st.release.gpu.f64
  tail call void @llvm.nvvm.st.gen.f.release.p0f64.f64(double* %dfp, double %df);

  ; CHECK: st.release.gpu.global.f64
  tail call void @llvm.nvvm.st.global.f.release.p1f64.f64(double addrspace(1)* %dfp1, double %df);

  ; CHECK: st.release.gpu.shared.f64
  tail call void @llvm.nvvm.st.shared.f.release.p3f64.f64(double addrspace(3)* %dfp3, double %df);

  ; CHECK: st.relaxed.sys.f64
  tail call void @llvm.nvvm.st.gen.f.sys.p0f64.f64(double* %dfp, double %df);

  ; CHECK: st.relaxed.sys.global.f64
  tail call void @llvm.nvvm.st.global.f.sys.p1f64.f64(double addrspace(1)* %dfp1, double %df);

  ; CHECK: st.relaxed.sys.shared.f64
  tail call void @llvm.nvvm.st.shared.f.sys.p3f64.f64(double addrspace(3)* %dfp3, double %df);

  ; CHECK: st.release.sys.f64
  tail call void @llvm.nvvm.st.gen.f.release.sys.p0f64.f64(double* %dfp, double %df);

  ; CHECK: st.release.sys.global.f64
  tail call void @llvm.nvvm.st.global.f.release.sys.p1f64.f64(double addrspace(1)* %dfp1, double %df);

  ; CHECK: st.release.sys.shared.f64
  tail call void @llvm.nvvm.st.shared.f.release.sys.p3f64.f64(double addrspace(3)* %dfp3, double %df);

  ; CHECK: st.relaxed.cta.f64
  tail call void @llvm.nvvm.st.gen.f.cta.p0f64.f64(double* %dfp, double %df);

  ; CHECK: st.relaxed.cta.global.f64
  tail call void @llvm.nvvm.st.global.f.cta.p1f64.f64(double addrspace(1)* %dfp1, double %df);

  ; CHECK: st.relaxed.cta.shared.f64
  tail call void @llvm.nvvm.st.shared.f.cta.p3f64.f64(double addrspace(3)* %dfp3, double %df);

  ; CHECK: st.release.cta.f64
  tail call void @llvm.nvvm.st.gen.f.release.cta.p0f64.f64(double* %dfp, double %df);

  ; CHECK: st.release.cta.global.f64
  tail call void @llvm.nvvm.st.global.f.release.cta.p1f64.f64(double addrspace(1)* %dfp1, double %df);

  ; CHECK: st.release.cta.shared.f64
  tail call void @llvm.nvvm.st.shared.f.release.cta.p3f64.f64(double addrspace(3)* %dfp3, double %df);

  ; CHECK: ld.volatile.s32
  %tmpldst144 = tail call i32 @llvm.nvvm.ld.gen.i.volatile.i32.p0(i32* %ip);

  ; CHECK: ld.volatile.global.s32
  %tmpldst145 = tail call i32 @llvm.nvvm.ld.global.i.volatile.i32.p1(i32 addrspace(1)* %ip1);

  ; CHECK: ld.volatile.shared.s32
  %tmpldst146 = tail call i32 @llvm.nvvm.ld.shared.i.volatile.i32.p3(i32 addrspace(3)* %ip3);

  ; CHECK: ld.volatile.s64
  %tmpldst147 = tail call i64 @llvm.nvvm.ld.gen.i.volatile.i64.p0(i64* %llp);

  ; CHECK: ld.volatile.global.s64
  %tmpldst148 = tail call i64 @llvm.nvvm.ld.global.i.volatile.i64.p1(i64 addrspace(1)* %llp1);

  ; CHECK: ld.volatile.shared.s64
  %tmpldst149 = tail call i64 @llvm.nvvm.ld.shared.i.volatile.i64.p3(i64 addrspace(3)* %llp3);

  ; CHECK: ld.volatile.f32
  %tmpldst150 = tail call float @llvm.nvvm.ld.gen.f.volatile.f32.p0(float* %fp);

  ; CHECK: ld.volatile.global.f32
  %tmpldst151 = tail call float @llvm.nvvm.ld.global.f.volatile.f32.p1(float addrspace(1)* %fp1);

  ; CHECK: ld.volatile.shared.f32
  %tmpldst152 = tail call float @llvm.nvvm.ld.shared.f.volatile.f32.p3(float addrspace(3)* %fp3);

  ; CHECK: ld.volatile.f64
  %tmpldst153 = tail call double @llvm.nvvm.ld.gen.f.volatile.f64.p0(double* %dfp);

  ; CHECK: ld.volatile.global.f64
  %tmpldst154 = tail call double @llvm.nvvm.ld.global.f.volatile.f64.p1(double addrspace(1)* %dfp1);

  ; CHECK: ld.volatile.shared.f64
  %tmpldst155 = tail call double @llvm.nvvm.ld.shared.f.volatile.f64.p3(double addrspace(3)* %dfp3);

  ; CHECK: st.volatile.s32
  tail call void @llvm.nvvm.st.gen.i.volatile.p0i32.i32(i32* %ip, i32 %i);

  ; CHECK: st.volatile.global.s32
  tail call void @llvm.nvvm.st.global.i.volatile.p1i32.i32(i32 addrspace(1)* %ip1, i32 %i);

  ; CHECK: st.volatile.shared.s32
  tail call void @llvm.nvvm.st.shared.i.volatile.p3i32.i32(i32 addrspace(3)* %ip3, i32 %i);

  ; CHECK: st.volatile.s64
  tail call void @llvm.nvvm.st.gen.i.volatile.p0i64.i64(i64* %llp, i64 %ll);

  ; CHECK: st.volatile.global.s64
  tail call void @llvm.nvvm.st.global.i.volatile.p1i64.i64(i64 addrspace(1)* %llp1, i64 %ll);

  ; CHECK: st.volatile.shared.s64
  tail call void @llvm.nvvm.st.shared.i.volatile.p3i64.i64(i64 addrspace(3)* %llp3, i64 %ll);

  ; CHECK: st.volatile.f32
  tail call void @llvm.nvvm.st.gen.f.volatile.p0f32.f32(float* %fp, float %f);

  ; CHECK: st.volatile.global.f32
  tail call void @llvm.nvvm.st.global.f.volatile.p1f32.f32(float addrspace(1)* %fp1, float %f);

  ; CHECK: st.volatile.shared.f32
  tail call void @llvm.nvvm.st.shared.f.volatile.p3f32.f32(float addrspace(3)* %fp3, float %f);

  ; CHECK: st.volatile.f64
  tail call void @llvm.nvvm.st.gen.f.volatile.p0f64.f64(double* %dfp, double %df);

  ; CHECK: st.volatile.global.f64
  tail call void @llvm.nvvm.st.global.f.volatile.p1f64.f64(double addrspace(1)* %dfp1, double %df);

  ; CHECK: st.volatile.shared.f64
  tail call void @llvm.nvvm.st.shared.f.volatile.p3f64.f64(double addrspace(3)* %dfp3, double %df);

  ; CHECK: ret
  ret void
}

; Make sure we use constants as operands to our scoped atomic calls, where appropriate.
; CHECK-LABEL: .func test_atomics_scope_imm(
define void @test_atomics_scope_imm(float* %fp, float %f,
                                    double* %dfp, double %df,
                                    i32* %ip, i32 %i,
                                    i32* %uip, i32 %ui,
                                    i64* %llp, i64 %ll) #0 {



  ; CHECK: ret
  ret void
}

declare i32 @llvm.nvvm.ld.gen.i.i32.p0(i32* nocapture) #1
declare i32 @llvm.nvvm.ld.global.i.i32.p1(i32 addrspace(1)* nocapture) #1
declare i32 @llvm.nvvm.ld.shared.i.i32.p3(i32 addrspace(3)* nocapture) #1
declare i32 @llvm.nvvm.ld.gen.i.acquire.i32.p0(i32* nocapture) #1
declare i32 @llvm.nvvm.ld.global.i.acquire.i32.p1(i32 addrspace(1)* nocapture) #1
declare i32 @llvm.nvvm.ld.shared.i.acquire.i32.p3(i32 addrspace(3)* nocapture) #1
declare i32 @llvm.nvvm.ld.gen.i.sys.i32.p0(i32* nocapture) #1
declare i32 @llvm.nvvm.ld.global.i.sys.i32.p1(i32 addrspace(1)* nocapture) #1
declare i32 @llvm.nvvm.ld.shared.i.sys.i32.p3(i32 addrspace(3)* nocapture) #1
declare i32 @llvm.nvvm.ld.gen.i.acquire.sys.i32.p0(i32* nocapture) #1
declare i32 @llvm.nvvm.ld.global.i.acquire.sys.i32.p1(i32 addrspace(1)* nocapture) #1
declare i32 @llvm.nvvm.ld.shared.i.acquire.sys.i32.p3(i32 addrspace(3)* nocapture) #1
declare i32 @llvm.nvvm.ld.gen.i.cta.i32.p0(i32* nocapture) #1
declare i32 @llvm.nvvm.ld.global.i.cta.i32.p1(i32 addrspace(1)* nocapture) #1
declare i32 @llvm.nvvm.ld.shared.i.cta.i32.p3(i32 addrspace(3)* nocapture) #1
declare i32 @llvm.nvvm.ld.gen.i.acquire.cta.i32.p0(i32* nocapture) #1
declare i32 @llvm.nvvm.ld.global.i.acquire.cta.i32.p1(i32 addrspace(1)* nocapture) #1
declare i32 @llvm.nvvm.ld.shared.i.acquire.cta.i32.p3(i32 addrspace(3)* nocapture) #1
declare i64 @llvm.nvvm.ld.gen.i.i64.p0(i64* nocapture) #1
declare i64 @llvm.nvvm.ld.global.i.i64.p1(i64 addrspace(1)* nocapture) #1
declare i64 @llvm.nvvm.ld.shared.i.i64.p3(i64 addrspace(3)* nocapture) #1
declare i64 @llvm.nvvm.ld.gen.i.acquire.i64.p0(i64* nocapture) #1
declare i64 @llvm.nvvm.ld.global.i.acquire.i64.p1(i64 addrspace(1)* nocapture) #1
declare i64 @llvm.nvvm.ld.shared.i.acquire.i64.p3(i64 addrspace(3)* nocapture) #1
declare i64 @llvm.nvvm.ld.gen.i.sys.i64.p0(i64* nocapture) #1
declare i64 @llvm.nvvm.ld.global.i.sys.i64.p1(i64 addrspace(1)* nocapture) #1
declare i64 @llvm.nvvm.ld.shared.i.sys.i64.p3(i64 addrspace(3)* nocapture) #1
declare i64 @llvm.nvvm.ld.gen.i.acquire.sys.i64.p0(i64* nocapture) #1
declare i64 @llvm.nvvm.ld.global.i.acquire.sys.i64.p1(i64 addrspace(1)* nocapture) #1
declare i64 @llvm.nvvm.ld.shared.i.acquire.sys.i64.p3(i64 addrspace(3)* nocapture) #1
declare i64 @llvm.nvvm.ld.gen.i.cta.i64.p0(i64* nocapture) #1
declare i64 @llvm.nvvm.ld.global.i.cta.i64.p1(i64 addrspace(1)* nocapture) #1
declare i64 @llvm.nvvm.ld.shared.i.cta.i64.p3(i64 addrspace(3)* nocapture) #1
declare i64 @llvm.nvvm.ld.gen.i.acquire.cta.i64.p0(i64* nocapture) #1
declare i64 @llvm.nvvm.ld.global.i.acquire.cta.i64.p1(i64 addrspace(1)* nocapture) #1
declare i64 @llvm.nvvm.ld.shared.i.acquire.cta.i64.p3(i64 addrspace(3)* nocapture) #1
declare float @llvm.nvvm.ld.gen.f.f32.p0(float* nocapture) #1
declare float @llvm.nvvm.ld.global.f.f32.p1(float addrspace(1)* nocapture) #1
declare float @llvm.nvvm.ld.shared.f.f32.p3(float addrspace(3)* nocapture) #1
declare float @llvm.nvvm.ld.gen.f.acquire.f32.p0(float* nocapture) #1
declare float @llvm.nvvm.ld.global.f.acquire.f32.p1(float addrspace(1)* nocapture) #1
declare float @llvm.nvvm.ld.shared.f.acquire.f32.p3(float addrspace(3)* nocapture) #1
declare float @llvm.nvvm.ld.gen.f.sys.f32.p0(float* nocapture) #1
declare float @llvm.nvvm.ld.global.f.sys.f32.p1(float addrspace(1)* nocapture) #1
declare float @llvm.nvvm.ld.shared.f.sys.f32.p3(float addrspace(3)* nocapture) #1
declare float @llvm.nvvm.ld.gen.f.acquire.sys.f32.p0(float* nocapture) #1
declare float @llvm.nvvm.ld.global.f.acquire.sys.f32.p1(float addrspace(1)* nocapture) #1
declare float @llvm.nvvm.ld.shared.f.acquire.sys.f32.p3(float addrspace(3)* nocapture) #1
declare float @llvm.nvvm.ld.gen.f.cta.f32.p0(float* nocapture) #1
declare float @llvm.nvvm.ld.global.f.cta.f32.p1(float addrspace(1)* nocapture) #1
declare float @llvm.nvvm.ld.shared.f.cta.f32.p3(float addrspace(3)* nocapture) #1
declare float @llvm.nvvm.ld.gen.f.acquire.cta.f32.p0(float* nocapture) #1
declare float @llvm.nvvm.ld.global.f.acquire.cta.f32.p1(float addrspace(1)* nocapture) #1
declare float @llvm.nvvm.ld.shared.f.acquire.cta.f32.p3(float addrspace(3)* nocapture) #1
declare double @llvm.nvvm.ld.gen.f.f64.p0(double* nocapture) #1
declare double @llvm.nvvm.ld.global.f.f64.p1(double addrspace(1)* nocapture) #1
declare double @llvm.nvvm.ld.shared.f.f64.p3(double addrspace(3)* nocapture) #1
declare double @llvm.nvvm.ld.gen.f.acquire.f64.p0(double* nocapture) #1
declare double @llvm.nvvm.ld.global.f.acquire.f64.p1(double addrspace(1)* nocapture) #1
declare double @llvm.nvvm.ld.shared.f.acquire.f64.p3(double addrspace(3)* nocapture) #1
declare double @llvm.nvvm.ld.gen.f.sys.f64.p0(double* nocapture) #1
declare double @llvm.nvvm.ld.global.f.sys.f64.p1(double addrspace(1)* nocapture) #1
declare double @llvm.nvvm.ld.shared.f.sys.f64.p3(double addrspace(3)* nocapture) #1
declare double @llvm.nvvm.ld.gen.f.acquire.sys.f64.p0(double* nocapture) #1
declare double @llvm.nvvm.ld.global.f.acquire.sys.f64.p1(double addrspace(1)* nocapture) #1
declare double @llvm.nvvm.ld.shared.f.acquire.sys.f64.p3(double addrspace(3)* nocapture) #1
declare double @llvm.nvvm.ld.gen.f.cta.f64.p0(double* nocapture) #1
declare double @llvm.nvvm.ld.global.f.cta.f64.p1(double addrspace(1)* nocapture) #1
declare double @llvm.nvvm.ld.shared.f.cta.f64.p3(double addrspace(3)* nocapture) #1
declare double @llvm.nvvm.ld.gen.f.acquire.cta.f64.p0(double* nocapture) #1
declare double @llvm.nvvm.ld.global.f.acquire.cta.f64.p1(double addrspace(1)* nocapture) #1
declare double @llvm.nvvm.ld.shared.f.acquire.cta.f64.p3(double addrspace(3)* nocapture) #1
declare void @llvm.nvvm.st.gen.i.p0i32.i32(i32* nocapture, i32) #1
declare void @llvm.nvvm.st.global.i.p1i32.i32(i32 addrspace(1)* nocapture, i32) #1
declare void @llvm.nvvm.st.shared.i.p3i32.i32(i32 addrspace(3)* nocapture, i32) #1
declare void @llvm.nvvm.st.gen.i.release.p0i32.i32(i32* nocapture, i32) #1
declare void @llvm.nvvm.st.global.i.release.p1i32.i32(i32 addrspace(1)* nocapture, i32) #1
declare void @llvm.nvvm.st.shared.i.release.p3i32.i32(i32 addrspace(3)* nocapture, i32) #1
declare void @llvm.nvvm.st.gen.i.sys.p0i32.i32(i32* nocapture, i32) #1
declare void @llvm.nvvm.st.global.i.sys.p1i32.i32(i32 addrspace(1)* nocapture, i32) #1
declare void @llvm.nvvm.st.shared.i.sys.p3i32.i32(i32 addrspace(3)* nocapture, i32) #1
declare void @llvm.nvvm.st.gen.i.release.sys.p0i32.i32(i32* nocapture, i32) #1
declare void @llvm.nvvm.st.global.i.release.sys.p1i32.i32(i32 addrspace(1)* nocapture, i32) #1
declare void @llvm.nvvm.st.shared.i.release.sys.p3i32.i32(i32 addrspace(3)* nocapture, i32) #1
declare void @llvm.nvvm.st.gen.i.cta.p0i32.i32(i32* nocapture, i32) #1
declare void @llvm.nvvm.st.global.i.cta.p1i32.i32(i32 addrspace(1)* nocapture, i32) #1
declare void @llvm.nvvm.st.shared.i.cta.p3i32.i32(i32 addrspace(3)* nocapture, i32) #1
declare void @llvm.nvvm.st.gen.i.release.cta.p0i32.i32(i32* nocapture, i32) #1
declare void @llvm.nvvm.st.global.i.release.cta.p1i32.i32(i32 addrspace(1)* nocapture, i32) #1
declare void @llvm.nvvm.st.shared.i.release.cta.p3i32.i32(i32 addrspace(3)* nocapture, i32) #1
declare void @llvm.nvvm.st.gen.i.p0i64.i64(i64* nocapture, i64) #1
declare void @llvm.nvvm.st.global.i.p1i64.i64(i64 addrspace(1)* nocapture, i64) #1
declare void @llvm.nvvm.st.shared.i.p3i64.i64(i64 addrspace(3)* nocapture, i64) #1
declare void @llvm.nvvm.st.gen.i.release.p0i64.i64(i64* nocapture, i64) #1
declare void @llvm.nvvm.st.global.i.release.p1i64.i64(i64 addrspace(1)* nocapture, i64) #1
declare void @llvm.nvvm.st.shared.i.release.p3i64.i64(i64 addrspace(3)* nocapture, i64) #1
declare void @llvm.nvvm.st.gen.i.sys.p0i64.i64(i64* nocapture, i64) #1
declare void @llvm.nvvm.st.global.i.sys.p1i64.i64(i64 addrspace(1)* nocapture, i64) #1
declare void @llvm.nvvm.st.shared.i.sys.p3i64.i64(i64 addrspace(3)* nocapture, i64) #1
declare void @llvm.nvvm.st.gen.i.release.sys.p0i64.i64(i64* nocapture, i64) #1
declare void @llvm.nvvm.st.global.i.release.sys.p1i64.i64(i64 addrspace(1)* nocapture, i64) #1
declare void @llvm.nvvm.st.shared.i.release.sys.p3i64.i64(i64 addrspace(3)* nocapture, i64) #1
declare void @llvm.nvvm.st.gen.i.cta.p0i64.i64(i64* nocapture, i64) #1
declare void @llvm.nvvm.st.global.i.cta.p1i64.i64(i64 addrspace(1)* nocapture, i64) #1
declare void @llvm.nvvm.st.shared.i.cta.p3i64.i64(i64 addrspace(3)* nocapture, i64) #1
declare void @llvm.nvvm.st.gen.i.release.cta.p0i64.i64(i64* nocapture, i64) #1
declare void @llvm.nvvm.st.global.i.release.cta.p1i64.i64(i64 addrspace(1)* nocapture, i64) #1
declare void @llvm.nvvm.st.shared.i.release.cta.p3i64.i64(i64 addrspace(3)* nocapture, i64) #1
declare void @llvm.nvvm.st.gen.f.p0f32.f32(float* nocapture, float) #1
declare void @llvm.nvvm.st.global.f.p1f32.f32(float addrspace(1)* nocapture, float) #1
declare void @llvm.nvvm.st.shared.f.p3f32.f32(float addrspace(3)* nocapture, float) #1
declare void @llvm.nvvm.st.gen.f.release.p0f32.f32(float* nocapture, float) #1
declare void @llvm.nvvm.st.global.f.release.p1f32.f32(float addrspace(1)* nocapture, float) #1
declare void @llvm.nvvm.st.shared.f.release.p3f32.f32(float addrspace(3)* nocapture, float) #1
declare void @llvm.nvvm.st.gen.f.sys.p0f32.f32(float* nocapture, float) #1
declare void @llvm.nvvm.st.global.f.sys.p1f32.f32(float addrspace(1)* nocapture, float) #1
declare void @llvm.nvvm.st.shared.f.sys.p3f32.f32(float addrspace(3)* nocapture, float) #1
declare void @llvm.nvvm.st.gen.f.release.sys.p0f32.f32(float* nocapture, float) #1
declare void @llvm.nvvm.st.global.f.release.sys.p1f32.f32(float addrspace(1)* nocapture, float) #1
declare void @llvm.nvvm.st.shared.f.release.sys.p3f32.f32(float addrspace(3)* nocapture, float) #1
declare void @llvm.nvvm.st.gen.f.cta.p0f32.f32(float* nocapture, float) #1
declare void @llvm.nvvm.st.global.f.cta.p1f32.f32(float addrspace(1)* nocapture, float) #1
declare void @llvm.nvvm.st.shared.f.cta.p3f32.f32(float addrspace(3)* nocapture, float) #1
declare void @llvm.nvvm.st.gen.f.release.cta.p0f32.f32(float* nocapture, float) #1
declare void @llvm.nvvm.st.global.f.release.cta.p1f32.f32(float addrspace(1)* nocapture, float) #1
declare void @llvm.nvvm.st.shared.f.release.cta.p3f32.f32(float addrspace(3)* nocapture, float) #1
declare void @llvm.nvvm.st.gen.f.p0f64.f64(double* nocapture, double) #1
declare void @llvm.nvvm.st.global.f.p1f64.f64(double addrspace(1)* nocapture, double) #1
declare void @llvm.nvvm.st.shared.f.p3f64.f64(double addrspace(3)* nocapture, double) #1
declare void @llvm.nvvm.st.gen.f.release.p0f64.f64(double* nocapture, double) #1
declare void @llvm.nvvm.st.global.f.release.p1f64.f64(double addrspace(1)* nocapture, double) #1
declare void @llvm.nvvm.st.shared.f.release.p3f64.f64(double addrspace(3)* nocapture, double) #1
declare void @llvm.nvvm.st.gen.f.sys.p0f64.f64(double* nocapture, double) #1
declare void @llvm.nvvm.st.global.f.sys.p1f64.f64(double addrspace(1)* nocapture, double) #1
declare void @llvm.nvvm.st.shared.f.sys.p3f64.f64(double addrspace(3)* nocapture, double) #1
declare void @llvm.nvvm.st.gen.f.release.sys.p0f64.f64(double* nocapture, double) #1
declare void @llvm.nvvm.st.global.f.release.sys.p1f64.f64(double addrspace(1)* nocapture, double) #1
declare void @llvm.nvvm.st.shared.f.release.sys.p3f64.f64(double addrspace(3)* nocapture, double) #1
declare void @llvm.nvvm.st.gen.f.cta.p0f64.f64(double* nocapture, double) #1
declare void @llvm.nvvm.st.global.f.cta.p1f64.f64(double addrspace(1)* nocapture, double) #1
declare void @llvm.nvvm.st.shared.f.cta.p3f64.f64(double addrspace(3)* nocapture, double) #1
declare void @llvm.nvvm.st.gen.f.release.cta.p0f64.f64(double* nocapture, double) #1
declare void @llvm.nvvm.st.global.f.release.cta.p1f64.f64(double addrspace(1)* nocapture, double) #1
declare void @llvm.nvvm.st.shared.f.release.cta.p3f64.f64(double addrspace(3)* nocapture, double) #1
declare i32 @llvm.nvvm.ld.gen.i.volatile.i32.p0(i32* nocapture) #1
declare i32 @llvm.nvvm.ld.global.i.volatile.i32.p1(i32 addrspace(1)* nocapture) #1
declare i32 @llvm.nvvm.ld.shared.i.volatile.i32.p3(i32 addrspace(3)* nocapture) #1
declare i64 @llvm.nvvm.ld.gen.i.volatile.i64.p0(i64* nocapture) #1
declare i64 @llvm.nvvm.ld.global.i.volatile.i64.p1(i64 addrspace(1)* nocapture) #1
declare i64 @llvm.nvvm.ld.shared.i.volatile.i64.p3(i64 addrspace(3)* nocapture) #1
declare float @llvm.nvvm.ld.gen.f.volatile.f32.p0(float* nocapture) #1
declare float @llvm.nvvm.ld.global.f.volatile.f32.p1(float addrspace(1)* nocapture) #1
declare float @llvm.nvvm.ld.shared.f.volatile.f32.p3(float addrspace(3)* nocapture) #1
declare double @llvm.nvvm.ld.gen.f.volatile.f64.p0(double* nocapture) #1
declare double @llvm.nvvm.ld.global.f.volatile.f64.p1(double addrspace(1)* nocapture) #1
declare double @llvm.nvvm.ld.shared.f.volatile.f64.p3(double addrspace(3)* nocapture) #1
declare void @llvm.nvvm.st.gen.i.volatile.p0i32.i32(i32* nocapture, i32) #1
declare void @llvm.nvvm.st.global.i.volatile.p1i32.i32(i32 addrspace(1)* nocapture, i32) #1
declare void @llvm.nvvm.st.shared.i.volatile.p3i32.i32(i32 addrspace(3)* nocapture, i32) #1
declare void @llvm.nvvm.st.gen.i.volatile.p0i64.i64(i64* nocapture, i64) #1
declare void @llvm.nvvm.st.global.i.volatile.p1i64.i64(i64 addrspace(1)* nocapture, i64) #1
declare void @llvm.nvvm.st.shared.i.volatile.p3i64.i64(i64 addrspace(3)* nocapture, i64) #1
declare void @llvm.nvvm.st.gen.f.volatile.p0f32.f32(float* nocapture, float) #1
declare void @llvm.nvvm.st.global.f.volatile.p1f32.f32(float addrspace(1)* nocapture, float) #1
declare void @llvm.nvvm.st.shared.f.volatile.p3f32.f32(float addrspace(3)* nocapture, float) #1
declare void @llvm.nvvm.st.gen.f.volatile.p0f64.f64(double* nocapture, double) #1
declare void @llvm.nvvm.st.global.f.volatile.p1f64.f64(double addrspace(1)* nocapture, double) #1
declare void @llvm.nvvm.st.shared.f.volatile.p3f64.f64(double addrspace(3)* nocapture, double) #1

declare i32 @llvm.nvvm.atomic.add.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.global.i.acquire.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.shared.i.acquire.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.global.i.release.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.shared.i.release.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.gen.i.acquire.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.gen.i.release.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.global.i.release.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.gen.i.acq.rel.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.gen.i.acquire.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.gen.i.release.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.global.i.release.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.gen.i.acq.rel.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.global.i.acquire.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.shared.i.acquire.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.global.i.release.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.shared.i.release.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.acquire.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.release.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.global.i.release.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.acq.rel.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.acquire.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.release.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.global.i.release.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.acq.rel.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare float @llvm.nvvm.atomic.add.gen.f.acquire.f32.p0f32(float* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.global.f.acquire.f32.p1f32(float addrspace(1)* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.shared.f.acquire.f32.p3f32(float addrspace(3)* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.gen.f.release.f32.p0f32(float* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.global.f.release.f32.p1f32(float addrspace(1)* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.shared.f.release.f32.p3f32(float addrspace(3)* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.gen.f.acq.rel.f32.p0f32(float* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.global.f.acq.rel.f32.p1f32(float addrspace(1)* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.shared.f.acq.rel.f32.p3f32(float addrspace(3)* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.gen.f.acquire.sys.f32.p0f32(float* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.global.f.acquire.sys.f32.p1f32(float addrspace(1)* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.shared.f.acquire.sys.f32.p3f32(float addrspace(3)* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.gen.f.release.sys.f32.p0f32(float* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.global.f.release.sys.f32.p1f32(float addrspace(1)* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.shared.f.release.sys.f32.p3f32(float addrspace(3)* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.gen.f.acq.rel.sys.f32.p0f32(float* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.global.f.acq.rel.sys.f32.p1f32(float addrspace(1)* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.shared.f.acq.rel.sys.f32.p3f32(float addrspace(3)* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.gen.f.acquire.cta.f32.p0f32(float* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.global.f.acquire.cta.f32.p1f32(float addrspace(1)* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.shared.f.acquire.cta.f32.p3f32(float addrspace(3)* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.gen.f.release.cta.f32.p0f32(float* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.global.f.release.cta.f32.p1f32(float addrspace(1)* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.shared.f.release.cta.f32.p3f32(float addrspace(3)* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.gen.f.acq.rel.cta.f32.p0f32(float* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.global.f.acq.rel.cta.f32.p1f32(float addrspace(1)* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.shared.f.acq.rel.cta.f32.p3f32(float addrspace(3)* nocapture, float) #1
declare double @llvm.nvvm.atomic.add.gen.f.acquire.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.global.f.acquire.f64.p1f64(double addrspace(1)* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.shared.f.acquire.f64.p3f64(double addrspace(3)* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.gen.f.release.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.global.f.release.f64.p1f64(double addrspace(1)* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.shared.f.release.f64.p3f64(double addrspace(3)* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.gen.f.acq.rel.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.global.f.acq.rel.f64.p1f64(double addrspace(1)* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.shared.f.acq.rel.f64.p3f64(double addrspace(3)* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.gen.f.acquire.sys.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.global.f.acquire.sys.f64.p1f64(double addrspace(1)* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.shared.f.acquire.sys.f64.p3f64(double addrspace(3)* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.gen.f.release.sys.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.global.f.release.sys.f64.p1f64(double addrspace(1)* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.shared.f.release.sys.f64.p3f64(double addrspace(3)* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.gen.f.acq.rel.sys.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.global.f.acq.rel.sys.f64.p1f64(double addrspace(1)* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.shared.f.acq.rel.sys.f64.p3f64(double addrspace(3)* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.gen.f.acquire.cta.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.global.f.acquire.cta.f64.p1f64(double addrspace(1)* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.shared.f.acquire.cta.f64.p3f64(double addrspace(3)* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.gen.f.release.cta.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.global.f.release.cta.f64.p1f64(double addrspace(1)* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.shared.f.release.cta.f64.p3f64(double addrspace(3)* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.gen.f.acq.rel.cta.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.global.f.acq.rel.cta.f64.p1f64(double addrspace(1)* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.shared.f.acq.rel.cta.f64.p3f64(double addrspace(3)* nocapture, double) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.global.i.acquire.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.shared.i.acquire.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.global.i.release.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.shared.i.release.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.acquire.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.release.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.global.i.release.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.acq.rel.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.acquire.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.release.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.global.i.release.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.acq.rel.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.global.i.acquire.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.shared.i.acquire.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.global.i.release.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.shared.i.release.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.acquire.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.release.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.global.i.release.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.acq.rel.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.acquire.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.release.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.global.i.release.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.acq.rel.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.i.acquire.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.i.acquire.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.i.release.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.i.release.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.acquire.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.release.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.i.release.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.acq.rel.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.acquire.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.release.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.i.release.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.acq.rel.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.i.acquire.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.i.acquire.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.i.release.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.i.release.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.acquire.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.release.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.i.release.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.acq.rel.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.acquire.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.release.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.i.release.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.acq.rel.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.ui.acquire.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.ui.acquire.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.ui.release.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.ui.release.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.ui.acq.rel.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.ui.acq.rel.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.acquire.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.ui.acquire.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.ui.acquire.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.release.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.ui.release.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.ui.release.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.acq.rel.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.ui.acq.rel.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.ui.acq.rel.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.acquire.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.ui.acquire.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.ui.acquire.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.release.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.ui.release.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.ui.release.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.acq.rel.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.global.ui.acq.rel.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.shared.ui.acq.rel.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.ui.acquire.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.ui.acquire.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.ui.release.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.ui.release.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.ui.acq.rel.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.ui.acq.rel.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.acquire.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.ui.acquire.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.ui.acquire.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.release.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.ui.release.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.ui.release.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.acq.rel.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.ui.acq.rel.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.ui.acq.rel.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.acquire.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.ui.acquire.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.ui.acquire.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.release.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.ui.release.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.ui.release.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.acq.rel.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.global.ui.acq.rel.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.shared.ui.acq.rel.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.i.acquire.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.i.acquire.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.i.release.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.i.release.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.acquire.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.release.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.i.release.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.acq.rel.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.acquire.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.release.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.i.release.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.acq.rel.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.i.acquire.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.i.acquire.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.i.release.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.i.release.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.acquire.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.release.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.i.release.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.acq.rel.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.acquire.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.release.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.i.release.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.acq.rel.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.ui.acquire.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.ui.acquire.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.ui.release.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.ui.release.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.ui.acq.rel.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.ui.acq.rel.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.acquire.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.ui.acquire.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.ui.acquire.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.release.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.ui.release.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.ui.release.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.acq.rel.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.ui.acq.rel.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.ui.acq.rel.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.acquire.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.ui.acquire.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.ui.acquire.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.release.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.ui.release.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.ui.release.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.acq.rel.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.global.ui.acq.rel.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.shared.ui.acq.rel.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.ui.acquire.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.ui.acquire.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.ui.release.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.ui.release.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.ui.acq.rel.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.ui.acq.rel.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.acquire.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.ui.acquire.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.ui.acquire.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.release.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.ui.release.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.ui.release.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.acq.rel.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.ui.acq.rel.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.ui.acq.rel.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.acquire.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.ui.acquire.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.ui.acquire.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.release.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.ui.release.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.ui.release.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.acq.rel.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.global.ui.acq.rel.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.shared.ui.acq.rel.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.global.i.acquire.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.shared.i.acquire.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.global.i.release.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.shared.i.release.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.acquire.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.release.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.global.i.release.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.acq.rel.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.acquire.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.release.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.global.i.release.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.acq.rel.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.global.i.acquire.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.shared.i.acquire.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.global.i.release.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.shared.i.release.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.acquire.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.release.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.global.i.release.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.acq.rel.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.acquire.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.release.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.global.i.release.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.acq.rel.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.global.i.acquire.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.shared.i.acquire.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.global.i.release.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.shared.i.release.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.acquire.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.release.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.global.i.release.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.acq.rel.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.acquire.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.release.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.global.i.release.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.acq.rel.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.global.i.acquire.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.shared.i.acquire.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.global.i.release.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.shared.i.release.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.acquire.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.release.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.global.i.release.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.acq.rel.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.acquire.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.release.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.global.i.release.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.acq.rel.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.global.i.acquire.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.shared.i.acquire.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.global.i.release.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.shared.i.release.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.acquire.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.release.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.global.i.release.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.acq.rel.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.acquire.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.release.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.global.i.release.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.acq.rel.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.global.i.acquire.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.shared.i.acquire.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.global.i.release.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.shared.i.release.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.acquire.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.release.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.global.i.release.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.acq.rel.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.acquire.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.release.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.global.i.release.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.acq.rel.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.global.i.acquire.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.shared.i.acquire.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.global.i.release.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.shared.i.release.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.acquire.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.release.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.global.i.release.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.acq.rel.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.acquire.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.release.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.global.i.release.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.acq.rel.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.global.i.acquire.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.shared.i.acquire.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.global.i.release.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.shared.i.release.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.acquire.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.release.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.global.i.release.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.acq.rel.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.acquire.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.release.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.global.i.release.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.acq.rel.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.global.i.acquire.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.shared.i.acquire.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.global.i.release.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.shared.i.release.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.acquire.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.release.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.global.i.release.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.acq.rel.sys.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.acquire.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.release.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.global.i.release.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.acq.rel.cta.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.global.i.acquire.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.shared.i.acquire.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.global.i.release.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.shared.i.release.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.acquire.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.release.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.global.i.release.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.acq.rel.sys.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.acquire.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.release.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.global.i.release.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.acq.rel.cta.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.acquire.i32.p0i32(i32* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.global.i.acquire.i32.p1i32(i32 addrspace(1)* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.shared.i.acquire.i32.p3i32(i32 addrspace(3)* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.release.i32.p0i32(i32* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.global.i.release.i32.p1i32(i32 addrspace(1)* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.shared.i.release.i32.p3i32(i32 addrspace(3)* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.global.i.acq.rel.i32.p1i32(i32 addrspace(1)* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.shared.i.acq.rel.i32.p3i32(i32 addrspace(3)* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.acquire.sys.i32.p0i32(i32* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.global.i.acquire.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.shared.i.acquire.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.release.sys.i32.p0i32(i32* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.global.i.release.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.shared.i.release.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.acq.rel.sys.i32.p0i32(i32* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.global.i.acq.rel.sys.i32.p1i32(i32 addrspace(1)* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.shared.i.acq.rel.sys.i32.p3i32(i32 addrspace(3)* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.acquire.cta.i32.p0i32(i32* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.global.i.acquire.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.shared.i.acquire.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.release.cta.i32.p0i32(i32* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.global.i.release.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.shared.i.release.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.acq.rel.cta.i32.p0i32(i32* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.global.i.acq.rel.cta.i32.p1i32(i32 addrspace(1)* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.shared.i.acq.rel.cta.i32.p3i32(i32 addrspace(3)* nocapture, i32, i32) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.acquire.i64.p0i64(i64* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.global.i.acquire.i64.p1i64(i64 addrspace(1)* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.shared.i.acquire.i64.p3i64(i64 addrspace(3)* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.release.i64.p0i64(i64* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.global.i.release.i64.p1i64(i64 addrspace(1)* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.shared.i.release.i64.p3i64(i64 addrspace(3)* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.global.i.acq.rel.i64.p1i64(i64 addrspace(1)* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.shared.i.acq.rel.i64.p3i64(i64 addrspace(3)* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.acquire.sys.i64.p0i64(i64* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.global.i.acquire.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.shared.i.acquire.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.release.sys.i64.p0i64(i64* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.global.i.release.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.shared.i.release.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.acq.rel.sys.i64.p0i64(i64* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.global.i.acq.rel.sys.i64.p1i64(i64 addrspace(1)* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.shared.i.acq.rel.sys.i64.p3i64(i64 addrspace(3)* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.acquire.cta.i64.p0i64(i64* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.global.i.acquire.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.shared.i.acquire.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.release.cta.i64.p0i64(i64* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.global.i.release.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.shared.i.release.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.acq.rel.cta.i64.p0i64(i64* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.global.i.acq.rel.cta.i64.p1i64(i64 addrspace(1)* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.shared.i.acq.rel.cta.i64.p3i64(i64 addrspace(3)* nocapture, i64, i64) #1

attributes #1 = { argmemonly nounwind }
