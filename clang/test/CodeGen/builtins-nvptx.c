// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx-unknown-unknown -target-cpu sm_80 -target-feature +ptx70 \
// RUN:            -fcuda-is-device -S -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX70_SM80 -check-prefix=LP32 %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_80 -target-feature +ptx70 \
// RUN:            -fcuda-is-device -S -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX70_SM80 -check-prefix=LP64 %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx-unknown-unknown -target-cpu sm_60 \
// RUN:            -fcuda-is-device -S -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=LP32 %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_60 \
// RUN:            -fcuda-is-device -S -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=LP64 %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_61 \
// RUN:            -fcuda-is-device -S -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=LP64 %s
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_53 \
// RUN:   -DERROR_CHECK -fcuda-is-device -S -o /dev/null -x cuda -verify %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx-unknown-unknown -target-cpu sm_86 -target-feature +ptx72 \
// RUN:            -fcuda-is-device -S -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX72_SM86 -check-prefix=LP32 %s
// RUN: %clang_cc1 -ffp-contract=off -triple nvptx64-unknown-unknown -target-cpu sm_86 -target-feature +ptx72 \
// RUN:            -fcuda-is-device -S -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK_PTX72_SM86 -check-prefix=LP64 %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

__device__ int read_tid() {

// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.tid.w()

  int x = __nvvm_read_ptx_sreg_tid_x();
  int y = __nvvm_read_ptx_sreg_tid_y();
  int z = __nvvm_read_ptx_sreg_tid_z();
  int w = __nvvm_read_ptx_sreg_tid_w();

  return x + y + z + w;

}

__device__ int read_ntid() {

// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ntid.w()

  int x = __nvvm_read_ptx_sreg_ntid_x();
  int y = __nvvm_read_ptx_sreg_ntid_y();
  int z = __nvvm_read_ptx_sreg_ntid_z();
  int w = __nvvm_read_ptx_sreg_ntid_w();

  return x + y + z + w;

}

__device__ int read_ctaid() {

// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.w()

  int x = __nvvm_read_ptx_sreg_ctaid_x();
  int y = __nvvm_read_ptx_sreg_ctaid_y();
  int z = __nvvm_read_ptx_sreg_ctaid_z();
  int w = __nvvm_read_ptx_sreg_ctaid_w();

  return x + y + z + w;

}

__device__ int read_nctaid() {

// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nctaid.w()

  int x = __nvvm_read_ptx_sreg_nctaid_x();
  int y = __nvvm_read_ptx_sreg_nctaid_y();
  int z = __nvvm_read_ptx_sreg_nctaid_z();
  int w = __nvvm_read_ptx_sreg_nctaid_w();

  return x + y + z + w;

}

__device__ int read_ids() {

// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.laneid()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.warpid()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nwarpid()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.smid()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nsmid()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.gridid()

  int a = __nvvm_read_ptx_sreg_laneid();
  int b = __nvvm_read_ptx_sreg_warpid();
  int c = __nvvm_read_ptx_sreg_nwarpid();
  int d = __nvvm_read_ptx_sreg_smid();
  int e = __nvvm_read_ptx_sreg_nsmid();
  int f = __nvvm_read_ptx_sreg_gridid();

  return a + b + c + d + e + f;

}

__device__ int read_lanemasks() {

// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.lanemask.eq()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.lanemask.le()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.lanemask.lt()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.lanemask.ge()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.lanemask.gt()

  int a = __nvvm_read_ptx_sreg_lanemask_eq();
  int b = __nvvm_read_ptx_sreg_lanemask_le();
  int c = __nvvm_read_ptx_sreg_lanemask_lt();
  int d = __nvvm_read_ptx_sreg_lanemask_ge();
  int e = __nvvm_read_ptx_sreg_lanemask_gt();

  return a + b + c + d + e;

}

__device__ long long read_clocks() {

// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.clock()
// CHECK: call i64 @llvm.nvvm.read.ptx.sreg.clock64()

  int a = __nvvm_read_ptx_sreg_clock();
  long long b = __nvvm_read_ptx_sreg_clock64();

  return a + b;
}

__device__ int read_pms() {

// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.pm0()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.pm1()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.pm2()
// CHECK: call i32 @llvm.nvvm.read.ptx.sreg.pm3()

  int a = __nvvm_read_ptx_sreg_pm0();
  int b = __nvvm_read_ptx_sreg_pm1();
  int c = __nvvm_read_ptx_sreg_pm2();
  int d = __nvvm_read_ptx_sreg_pm3();

  return a + b + c + d;

}

__device__ void sync() {

// CHECK: call void @llvm.nvvm.bar.sync(i32 0)

  __nvvm_bar_sync(0);

}


// NVVM intrinsics

// The idea is not to test all intrinsics, just that Clang is recognizing the
// builtins defined in BuiltinsNVPTX.def
__device__ void nvvm_math(float f1, float f2, double d1, double d2) {
// CHECK: call float @llvm.nvvm.fmax.f
  float t1 = __nvvm_fmax_f(f1, f2);
// CHECK: call float @llvm.nvvm.fmin.f
  float t2 = __nvvm_fmin_f(f1, f2);
// CHECK: call float @llvm.nvvm.sqrt.rn.f
  float t3 = __nvvm_sqrt_rn_f(f1);
// CHECK: call float @llvm.nvvm.rcp.rn.f
  float t4 = __nvvm_rcp_rn_f(f2);
// CHECK: call float @llvm.nvvm.add.rn.f
  float t5 = __nvvm_add_rn_f(f1, f2);

// CHECK: call double @llvm.nvvm.fmax.d
  double td1 = __nvvm_fmax_d(d1, d2);
// CHECK: call double @llvm.nvvm.fmin.d
  double td2 = __nvvm_fmin_d(d1, d2);
// CHECK: call double @llvm.nvvm.sqrt.rn.d
  double td3 = __nvvm_sqrt_rn_d(d1);
// CHECK: call double @llvm.nvvm.rcp.rn.d
  double td4 = __nvvm_rcp_rn_d(d2);

// CHECK: call void @llvm.nvvm.membar.cta()
  __nvvm_membar_cta();
// CHECK: call void @llvm.nvvm.membar.gl()
  __nvvm_membar_gl();
// CHECK: call void @llvm.nvvm.membar.sys()
  __nvvm_membar_sys();
// CHECK: call void @llvm.nvvm.barrier0()
  __syncthreads();
}

__device__ int di;
__shared__ int si;
__device__ long dl;
__shared__ long sl;
__device__ long long dll;
__shared__ long long sll;

// Check for atomic intrinsics
// CHECK-LABEL: nvvm_atom
__device__ void nvvm_atom(float *fp, float f, double *dfp, double df, int *ip,
                          int i, unsigned int *uip, unsigned ui, long *lp,
                          long l, long long *llp, long long ll) {
  // CHECK: atomicrmw add ptr {{.*}} seq_cst, align 4
  __nvvm_atom_add_gen_i(ip, i);
  // CHECK: atomicrmw add ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_add_gen_l(&dl, l);
  // CHECK: atomicrmw add ptr {{.*}} seq_cst, align 8
  __nvvm_atom_add_gen_ll(&sll, ll);

  // CHECK: atomicrmw sub ptr {{.*}} seq_cst, align 4
  __nvvm_atom_sub_gen_i(ip, i);
  // CHECK: atomicrmw sub ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_sub_gen_l(&dl, l);
  // CHECK: atomicrmw sub ptr {{.*}} seq_cst, align 8
  __nvvm_atom_sub_gen_ll(&sll, ll);

  // CHECK: atomicrmw and ptr {{.*}} seq_cst, align 4
  __nvvm_atom_and_gen_i(ip, i);
  // CHECK: atomicrmw and ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_and_gen_l(&dl, l);
  // CHECK: atomicrmw and ptr {{.*}} seq_cst, align 8
  __nvvm_atom_and_gen_ll(&sll, ll);

  // CHECK: atomicrmw or ptr {{.*}} seq_cst, align 4
  __nvvm_atom_or_gen_i(ip, i);
  // CHECK: atomicrmw or ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_or_gen_l(&dl, l);
  // CHECK: atomicrmw or ptr {{.*}} seq_cst, align 8
  __nvvm_atom_or_gen_ll(&sll, ll);

  // CHECK: atomicrmw xor ptr {{.*}} seq_cst, align 4
  __nvvm_atom_xor_gen_i(ip, i);
  // CHECK: atomicrmw xor ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_xor_gen_l(&dl, l);
  // CHECK: atomicrmw xor ptr {{.*}} seq_cst, align 8
  __nvvm_atom_xor_gen_ll(&sll, ll);

  // CHECK: atomicrmw xchg ptr {{.*}} seq_cst, align 4
  __nvvm_atom_xchg_gen_i(ip, i);
  // CHECK: atomicrmw xchg ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_xchg_gen_l(&dl, l);
  // CHECK: atomicrmw xchg ptr {{.*}} seq_cst, align 8
  __nvvm_atom_xchg_gen_ll(&sll, ll);
  // CHECK: call float @llvm.nvvm.atomic.exch.gen.f.f32.p0
  __nvvm_atom_xchg_gen_f(fp, f);
  // CHECK: call double @llvm.nvvm.atomic.exch.gen.f.f64.p0
  __nvvm_atom_xchg_gen_d(dfp, df);

  // CHECK: atomicrmw max ptr {{.*}} seq_cst, align 4
  __nvvm_atom_max_gen_i(ip, i);
  // CHECK: atomicrmw umax ptr {{.*}} seq_cst, align 4
  __nvvm_atom_max_gen_ui((unsigned int *)ip, i);
  // CHECK: atomicrmw max ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_max_gen_l(&dl, l);
  // CHECK: atomicrmw umax ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_max_gen_ul((unsigned long *)&dl, l);
  // CHECK: atomicrmw max ptr {{.*}} seq_cst, align 8
  __nvvm_atom_max_gen_ll(&sll, ll);
  // CHECK: atomicrmw umax ptr {{.*}} seq_cst, align 8
  __nvvm_atom_max_gen_ull((unsigned long long *)&sll, ll);

  // CHECK: atomicrmw min ptr {{.*}} seq_cst, align 4
  __nvvm_atom_min_gen_i(ip, i);
  // CHECK: atomicrmw umin ptr {{.*}} seq_cst, align 4
  __nvvm_atom_min_gen_ui((unsigned int *)ip, i);
  // CHECK: atomicrmw min ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_min_gen_l(&dl, l);
  // CHECK: atomicrmw umin ptr {{.*}} seq_cst, align {{4|8}}
  __nvvm_atom_min_gen_ul((unsigned long *)&dl, l);
  // CHECK: atomicrmw min ptr {{.*}} seq_cst, align 8
  __nvvm_atom_min_gen_ll(&sll, ll);
  // CHECK: atomicrmw umin ptr {{.*}} seq_cst, align 8
  __nvvm_atom_min_gen_ull((unsigned long long *)&sll, ll);

  // CHECK: cmpxchg ptr {{.*}} seq_cst seq_cst, align 4
  // CHECK-NEXT: extractvalue { i32, i1 } {{%[0-9]+}}, 0
  __nvvm_atom_cas_gen_i(ip, 0, i);
  // CHECK: cmpxchg ptr {{.*}} seq_cst seq_cst, align {{4|8}}
  // CHECK-NEXT: extractvalue { {{i32|i64}}, i1 } {{%[0-9]+}}, 0
  __nvvm_atom_cas_gen_l(&dl, 0, l);
  // CHECK: cmpxchg ptr {{.*}} seq_cst seq_cst, align 8
  // CHECK-NEXT: extractvalue { i64, i1 } {{%[0-9]+}}, 0
  __nvvm_atom_cas_gen_ll(&sll, 0, ll);
  // CHECK: call float @llvm.nvvm.atomic.cas.gen.f.f32.p0
  __nvvm_atom_cas_gen_f(fp, 0, f);
  // CHECK: call double @llvm.nvvm.atomic.cas.gen.f.f64.p0
  __nvvm_atom_cas_gen_d(dfp, 0, df);

  // CHECK: atomicrmw fadd ptr {{.*}} seq_cst, align 4
  __nvvm_atom_add_gen_f(fp, f);

  // CHECK: call i32 @llvm.nvvm.atomic.load.inc.32.p0
  __nvvm_atom_inc_gen_ui(uip, ui);

  // CHECK: call i32 @llvm.nvvm.atomic.load.dec.32.p0
  __nvvm_atom_dec_gen_ui(uip, ui);

  // CHECK: call i32 @llvm.nvvm.ld.gen.i.volatile.i32.p0
  __nvvm_volatile_ld_gen_i(ip);
  // CHECK: call i32 @llvm.nvvm.ld.global.i.volatile.i32.p1
  __nvvm_volatile_ld_global_i((__attribute__((address_space(1))) int *)ip);
  // CHECK: call i32 @llvm.nvvm.ld.shared.i.volatile.i32.p3
  __nvvm_volatile_ld_shared_i((__attribute__((address_space(3))) int *)ip);

  // CHECK: call void @llvm.nvvm.st.gen.i.volatile.p0.i32
  __nvvm_volatile_st_gen_i(ip, i);
  // CHECK: call void @llvm.nvvm.st.global.i.volatile.p1.i32
  __nvvm_volatile_st_global_i((__attribute__((address_space(1))) int *)ip, i);
  // CHECK: call void @llvm.nvvm.st.shared.i.volatile.p3.i32
  __nvvm_volatile_st_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // LP32: call i32 @llvm.nvvm.ld.gen.i.volatile.i32.p0
  // LP64: call i64 @llvm.nvvm.ld.gen.i.volatile.i64.p0
  __nvvm_volatile_ld_gen_l(&dl);
  // LP32: call i32 @llvm.nvvm.ld.global.i.volatile.i32.p1
  // LP64: call i64 @llvm.nvvm.ld.global.i.volatile.i64.p1
  __nvvm_volatile_ld_global_l((__attribute__((address_space(1))) long *)&dl);
  // LP32: call i32 @llvm.nvvm.ld.shared.i.volatile.i32.p3
  // LP64: call i64 @llvm.nvvm.ld.shared.i.volatile.i64.p3
  __nvvm_volatile_ld_shared_l((__attribute__((address_space(3))) long *)&dl);

  // LP32: call void @llvm.nvvm.st.gen.i.volatile.p0.i32
  // LP64: call void @llvm.nvvm.st.gen.i.volatile.p0.i64
  __nvvm_volatile_st_gen_l(&dl, l);
  // LP32: call void @llvm.nvvm.st.global.i.volatile.p1.i32
  // LP64: call void @llvm.nvvm.st.global.i.volatile.p1.i64
  __nvvm_volatile_st_global_l((__attribute__((address_space(1))) long *)&dl, l);
  // LP32: call void @llvm.nvvm.st.shared.i.volatile.p3.i32
  // LP64: call void @llvm.nvvm.st.shared.i.volatile.p3.i64
  __nvvm_volatile_st_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK: call i64 @llvm.nvvm.ld.gen.i.volatile.i64.p0
  __nvvm_volatile_ld_gen_ll(&dll);
  // CHECK: call i64 @llvm.nvvm.ld.global.i.volatile.i64.p1
  __nvvm_volatile_ld_global_ll((__attribute__((address_space(1))) long long *)&dll);
  // CHECK: call i64 @llvm.nvvm.ld.shared.i.volatile.i64.p3
  __nvvm_volatile_ld_shared_ll((__attribute__((address_space(3))) long long *)&dll);

  // CHECK: call void @llvm.nvvm.st.gen.i.volatile.p0.i64
  __nvvm_volatile_st_gen_ll(&dll, ll);
  // CHECK: call void @llvm.nvvm.st.global.i.volatile.p1.i64
  __nvvm_volatile_st_global_ll((__attribute__((address_space(1))) long long *)&dll, ll);
  // CHECK: call void @llvm.nvvm.st.shared.i.volatile.p3.i64
  __nvvm_volatile_st_shared_ll((__attribute__((address_space(3))) long long *)&dll, ll);

  // CHECK: call float @llvm.nvvm.ld.gen.f.volatile.f32.p0
  __nvvm_volatile_ld_gen_f(fp);
  // CHECK: call float @llvm.nvvm.ld.global.f.volatile.f32.p1
  __nvvm_volatile_ld_global_f((__attribute__((address_space(1))) float *)fp);
  // CHECK: call float @llvm.nvvm.ld.shared.f.volatile.f32.p3
  __nvvm_volatile_ld_shared_f((__attribute__((address_space(3))) float *)fp);

  // CHECK: call void @llvm.nvvm.st.gen.f.volatile.p0.f32
  __nvvm_volatile_st_gen_f(fp, f);
  // CHECK: call void @llvm.nvvm.st.global.f.volatile.p1.f32
  __nvvm_volatile_st_global_f((__attribute__((address_space(1))) float *)fp, f);
  // CHECK: call void @llvm.nvvm.st.shared.f.volatile.p3.f32
  __nvvm_volatile_st_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK: call double @llvm.nvvm.ld.gen.f.volatile.f64.p0
  __nvvm_volatile_ld_gen_d(dfp);
  // CHECK: call double @llvm.nvvm.ld.global.f.volatile.f64.p1
  __nvvm_volatile_ld_global_d((__attribute__((address_space(1))) double *)dfp);
  // CHECK: call double @llvm.nvvm.ld.shared.f.volatile.f64.p3
  __nvvm_volatile_ld_shared_d((__attribute__((address_space(3))) double *)dfp);

  // CHECK: call void @llvm.nvvm.st.gen.f.volatile.p0.f64
  __nvvm_volatile_st_gen_d(dfp, df);
  // CHECK: call void @llvm.nvvm.st.global.f.volatile.p1.f64
  __nvvm_volatile_st_global_d((__attribute__((address_space(1))) double *)dfp, df);
  // CHECK: call void @llvm.nvvm.st.shared.f.volatile.p3.f64
  __nvvm_volatile_st_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  //////////////////////////////////////////////////////////////////
  // Atomics with scope (only supported on sm_60+).

#if ERROR_CHECK || __CUDA_ARCH__ >= 600

  // CHECK: call i32 @llvm.nvvm.atomic.add.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_add_gen_i' needs target feature sm_60}}
  __nvvm_atom_cta_add_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.add.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.add.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_add_gen_l' needs target feature sm_60}}
  __nvvm_atom_cta_add_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.add.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_add_gen_ll' needs target feature sm_60}}
  __nvvm_atom_cta_add_gen_ll(&sll, ll);
  // CHECK: call i32 @llvm.nvvm.atomic.add.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_add_gen_i' needs target feature sm_60}}
  __nvvm_atom_sys_add_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.add.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.add.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_add_gen_l' needs target feature sm_60}}
  __nvvm_atom_sys_add_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.add.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_add_gen_ll' needs target feature sm_60}}
  __nvvm_atom_sys_add_gen_ll(&sll, ll);

  // CHECK: call float @llvm.nvvm.atomic.add.gen.f.cta.f32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_add_gen_f' needs target feature sm_60}}
  __nvvm_atom_cta_add_gen_f(fp, f);
  // CHECK: call double @llvm.nvvm.atomic.add.gen.f.cta.f64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_add_gen_d' needs target feature sm_60}}
  __nvvm_atom_cta_add_gen_d(dfp, df);
  // CHECK: call float @llvm.nvvm.atomic.add.gen.f.sys.f32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_add_gen_f' needs target feature sm_60}}
  __nvvm_atom_sys_add_gen_f(fp, f);
  // CHECK: call double @llvm.nvvm.atomic.add.gen.f.sys.f64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_add_gen_d' needs target feature sm_60}}
  __nvvm_atom_sys_add_gen_d(dfp, df);

  // CHECK: call i32 @llvm.nvvm.atomic.exch.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_xchg_gen_i' needs target feature sm_60}}
  __nvvm_atom_cta_xchg_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.exch.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.exch.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_xchg_gen_l' needs target feature sm_60}}
  __nvvm_atom_cta_xchg_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.exch.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_xchg_gen_ll' needs target feature sm_60}}
  __nvvm_atom_cta_xchg_gen_ll(&sll, ll);
  // CHECK: call float @llvm.nvvm.atomic.exch.gen.f.cta.f32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_xchg_gen_f' needs target feature sm_60}}
  __nvvm_atom_cta_xchg_gen_f(fp, f);
  // CHECK: call double @llvm.nvvm.atomic.exch.gen.f.cta.f64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_xchg_gen_d' needs target feature sm_60}}
  __nvvm_atom_cta_xchg_gen_d(dfp, df);

  // CHECK: call i32 @llvm.nvvm.atomic.exch.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_xchg_gen_i' needs target feature sm_60}}
  __nvvm_atom_sys_xchg_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.exch.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.exch.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_xchg_gen_l' needs target feature sm_60}}
  __nvvm_atom_sys_xchg_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.exch.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_xchg_gen_ll' needs target feature sm_60}}
  __nvvm_atom_sys_xchg_gen_ll(&sll, ll);
  // CHECK: call float @llvm.nvvm.atomic.exch.gen.f.sys.f32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_xchg_gen_f' needs target feature sm_60}}
  __nvvm_atom_sys_xchg_gen_f(fp, f);
  // CHECK: call double @llvm.nvvm.atomic.exch.gen.f.sys.f64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_xchg_gen_d' needs target feature sm_60}}
  __nvvm_atom_sys_xchg_gen_d(dfp, df);

  // CHECK: call i32 @llvm.nvvm.atomic.max.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_max_gen_i' needs target feature sm_60}}
  __nvvm_atom_cta_max_gen_i(ip, i);
  // CHECK: call i32 @llvm.nvvm.atomic.max.gen.ui.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_max_gen_ui' needs target feature sm_60}}
  __nvvm_atom_cta_max_gen_ui((unsigned int *)ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.max.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.max.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_max_gen_l' needs target feature sm_60}}
  __nvvm_atom_cta_max_gen_l(&dl, l);
  // LP32: call i32 @llvm.nvvm.atomic.max.gen.ui.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.max.gen.ui.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_max_gen_ul' needs target feature sm_60}}
  __nvvm_atom_cta_max_gen_ul((unsigned long *)lp, l);
  // CHECK: call i64 @llvm.nvvm.atomic.max.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_max_gen_ll' needs target feature sm_60}}
  __nvvm_atom_cta_max_gen_ll(&sll, ll);
  // CHECK: call i64 @llvm.nvvm.atomic.max.gen.ui.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_max_gen_ull' needs target feature sm_60}}
  __nvvm_atom_cta_max_gen_ull((unsigned long long *)llp, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.max.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_max_gen_i' needs target feature sm_60}}
  __nvvm_atom_sys_max_gen_i(ip, i);
  // CHECK: call i32 @llvm.nvvm.atomic.max.gen.ui.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_max_gen_ui' needs target feature sm_60}}
  __nvvm_atom_sys_max_gen_ui((unsigned int *)ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.max.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.max.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_max_gen_l' needs target feature sm_60}}
  __nvvm_atom_sys_max_gen_l(&dl, l);
  // LP32: call i32 @llvm.nvvm.atomic.max.gen.ui.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.max.gen.ui.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_max_gen_ul' needs target feature sm_60}}
  __nvvm_atom_sys_max_gen_ul((unsigned long *)lp, l);
  // CHECK: call i64 @llvm.nvvm.atomic.max.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_max_gen_ll' needs target feature sm_60}}
  __nvvm_atom_sys_max_gen_ll(&sll, ll);
  // CHECK: call i64 @llvm.nvvm.atomic.max.gen.ui.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_max_gen_ull' needs target feature sm_60}}
  __nvvm_atom_sys_max_gen_ull((unsigned long long *)llp, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.min.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_min_gen_i' needs target feature sm_60}}
  __nvvm_atom_cta_min_gen_i(ip, i);
  // CHECK: call i32 @llvm.nvvm.atomic.min.gen.ui.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_min_gen_ui' needs target feature sm_60}}
  __nvvm_atom_cta_min_gen_ui((unsigned int *)ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.min.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.min.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_min_gen_l' needs target feature sm_60}}
  __nvvm_atom_cta_min_gen_l(&dl, l);
  // LP32: call i32 @llvm.nvvm.atomic.min.gen.ui.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.min.gen.ui.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_min_gen_ul' needs target feature sm_60}}
  __nvvm_atom_cta_min_gen_ul((unsigned long *)lp, l);
  // CHECK: call i64 @llvm.nvvm.atomic.min.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_min_gen_ll' needs target feature sm_60}}
  __nvvm_atom_cta_min_gen_ll(&sll, ll);
  // CHECK: call i64 @llvm.nvvm.atomic.min.gen.ui.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_min_gen_ull' needs target feature sm_60}}
  __nvvm_atom_cta_min_gen_ull((unsigned long long *)llp, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.min.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_min_gen_i' needs target feature sm_60}}
  __nvvm_atom_sys_min_gen_i(ip, i);
  // CHECK: call i32 @llvm.nvvm.atomic.min.gen.ui.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_min_gen_ui' needs target feature sm_60}}
  __nvvm_atom_sys_min_gen_ui((unsigned int *)ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.min.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.min.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_min_gen_l' needs target feature sm_60}}
  __nvvm_atom_sys_min_gen_l(&dl, l);
  // LP32: call i32 @llvm.nvvm.atomic.min.gen.ui.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.min.gen.ui.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_min_gen_ul' needs target feature sm_60}}
  __nvvm_atom_sys_min_gen_ul((unsigned long *)lp, l);
  // CHECK: call i64 @llvm.nvvm.atomic.min.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_min_gen_ll' needs target feature sm_60}}
  __nvvm_atom_sys_min_gen_ll(&sll, ll);
  // CHECK: call i64 @llvm.nvvm.atomic.min.gen.ui.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_min_gen_ull' needs target feature sm_60}}
  __nvvm_atom_sys_min_gen_ull((unsigned long long *)llp, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.inc.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_inc_gen_ui' needs target feature sm_60}}
  __nvvm_atom_cta_inc_gen_ui((unsigned int *)ip, i);
  // CHECK: call i32 @llvm.nvvm.atomic.inc.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_inc_gen_ui' needs target feature sm_60}}
  __nvvm_atom_sys_inc_gen_ui((unsigned int *)ip, i);

  // CHECK: call i32 @llvm.nvvm.atomic.dec.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_dec_gen_ui' needs target feature sm_60}}
  __nvvm_atom_cta_dec_gen_ui((unsigned int *)ip, i);
  // CHECK: call i32 @llvm.nvvm.atomic.dec.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_dec_gen_ui' needs target feature sm_60}}
  __nvvm_atom_sys_dec_gen_ui((unsigned int *)ip, i);

  // CHECK: call i32 @llvm.nvvm.atomic.and.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_and_gen_i' needs target feature sm_60}}
  __nvvm_atom_cta_and_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.and.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.and.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_and_gen_l' needs target feature sm_60}}
  __nvvm_atom_cta_and_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.and.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_and_gen_ll' needs target feature sm_60}}
  __nvvm_atom_cta_and_gen_ll(&sll, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.and.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_and_gen_i' needs target feature sm_60}}
  __nvvm_atom_sys_and_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.and.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.and.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_and_gen_l' needs target feature sm_60}}
  __nvvm_atom_sys_and_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.and.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_and_gen_ll' needs target feature sm_60}}
  __nvvm_atom_sys_and_gen_ll(&sll, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.or.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_or_gen_i' needs target feature sm_60}}
  __nvvm_atom_cta_or_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.or.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.or.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_or_gen_l' needs target feature sm_60}}
  __nvvm_atom_cta_or_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.or.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_or_gen_ll' needs target feature sm_60}}
  __nvvm_atom_cta_or_gen_ll(&sll, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.or.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_or_gen_i' needs target feature sm_60}}
  __nvvm_atom_sys_or_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.or.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.or.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_or_gen_l' needs target feature sm_60}}
  __nvvm_atom_sys_or_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.or.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_or_gen_ll' needs target feature sm_60}}
  __nvvm_atom_sys_or_gen_ll(&sll, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.xor.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_xor_gen_i' needs target feature sm_60}}
  __nvvm_atom_cta_xor_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.xor.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.xor.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_xor_gen_l' needs target feature sm_60}}
  __nvvm_atom_cta_xor_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.xor.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_xor_gen_ll' needs target feature sm_60}}
  __nvvm_atom_cta_xor_gen_ll(&sll, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.xor.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_xor_gen_i' needs target feature sm_60}}
  __nvvm_atom_sys_xor_gen_i(ip, i);
  // LP32: call i32 @llvm.nvvm.atomic.xor.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.xor.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_xor_gen_l' needs target feature sm_60}}
  __nvvm_atom_sys_xor_gen_l(&dl, l);
  // CHECK: call i64 @llvm.nvvm.atomic.xor.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_xor_gen_ll' needs target feature sm_60}}
  __nvvm_atom_sys_xor_gen_ll(&sll, ll);

  // CHECK: call i32 @llvm.nvvm.atomic.cas.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_cta_cas_gen_i' needs target feature sm_60}}
  __nvvm_atom_cta_cas_gen_i(ip, i, 0);
  // LP32: call i32 @llvm.nvvm.atomic.cas.gen.i.cta.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.cas.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_cas_gen_l' needs target feature sm_60}}
  __nvvm_atom_cta_cas_gen_l(&dl, l, 0);
  // CHECK: call i64 @llvm.nvvm.atomic.cas.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_cta_cas_gen_ll' needs target feature sm_60}}
  __nvvm_atom_cta_cas_gen_ll(&sll, ll, 0);

  // CHECK: call i32 @llvm.nvvm.atomic.cas.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_sys_cas_gen_i' needs target feature sm_60}}
  __nvvm_atom_sys_cas_gen_i(ip, i, 0);
  // LP32: call i32 @llvm.nvvm.atomic.cas.gen.i.sys.i32.p0
  // LP64: call i64 @llvm.nvvm.atomic.cas.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_cas_gen_l' needs target feature sm_60}}
  __nvvm_atom_sys_cas_gen_l(&dl, l, 0);
  // CHECK: call i64 @llvm.nvvm.atomic.cas.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_sys_cas_gen_ll' needs target feature sm_60}}
  __nvvm_atom_sys_cas_gen_ll(&sll, ll, 0);
#endif

  //////////////////////////////////////////////////////////////////
  // Atomics with semantics (only supported on sm_70+).

#if ERROR_CHECK || __CUDA_ARCH__ >= 700

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.gen.i.i32.p0
  // expected-error@+1 {{'__nvvm_ld_gen_i' needs target feature sm_70}}
  __nvvm_ld_gen_i(ip);
  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.global.i.i32.p1
  // expected-error@+1 {{'__nvvm_ld_global_i' needs target feature sm_70}}
  __nvvm_ld_global_i((__attribute__((address_space(1))) int *)ip);
  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.shared.i.i32.p3
  // expected-error@+1 {{'__nvvm_ld_shared_i' needs target feature sm_70}}
  __nvvm_ld_shared_i((__attribute__((address_space(3))) int *)ip);
  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.gen.i.acquire.i32.p0
  // expected-error@+1 {{'__nvvm_acquire_ld_gen_i' needs target feature sm_70}}
  __nvvm_acquire_ld_gen_i(ip);
  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.global.i.acquire.i32.p1
  // expected-error@+1 {{'__nvvm_acquire_ld_global_i' needs target feature sm_70}}
  __nvvm_acquire_ld_global_i((__attribute__((address_space(1))) int *)ip);
  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.shared.i.acquire.i32.p3
  // expected-error@+1 {{'__nvvm_acquire_ld_shared_i' needs target feature sm_70}}
  __nvvm_acquire_ld_shared_i((__attribute__((address_space(3))) int *)ip);
  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.gen.i.sys.i32.p0
  // expected-error@+1 {{'__nvvm_sys_ld_gen_i' needs target feature sm_70}}
  __nvvm_sys_ld_gen_i(ip);
  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.global.i.sys.i32.p1
  // expected-error@+1 {{'__nvvm_sys_ld_global_i' needs target feature sm_70}}
  __nvvm_sys_ld_global_i((__attribute__((address_space(1))) int *)ip);
  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.shared.i.sys.i32.p3
  // expected-error@+1 {{'__nvvm_sys_ld_shared_i' needs target feature sm_70}}
  __nvvm_sys_ld_shared_i((__attribute__((address_space(3))) int *)ip);
  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.gen.i.acquire.sys.i32.p0
  // expected-error@+1 {{'__nvvm_acquire_sys_ld_gen_i' needs target feature sm_70}}
  __nvvm_acquire_sys_ld_gen_i(ip);
  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.global.i.acquire.sys.i32.p1
  // expected-error@+1 {{'__nvvm_acquire_sys_ld_global_i' needs target feature sm_70}}
  __nvvm_acquire_sys_ld_global_i((__attribute__((address_space(1))) int *)ip);
  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.shared.i.acquire.sys.i32.p3
  // expected-error@+1 {{'__nvvm_acquire_sys_ld_shared_i' needs target feature sm_70}}
  __nvvm_acquire_sys_ld_shared_i((__attribute__((address_space(3))) int *)ip);
  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.gen.i.cta.i32.p0
  // expected-error@+1 {{'__nvvm_cta_ld_gen_i' needs target feature sm_70}}
  __nvvm_cta_ld_gen_i(ip);
  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.global.i.cta.i32.p1
  // expected-error@+1 {{'__nvvm_cta_ld_global_i' needs target feature sm_70}}
  __nvvm_cta_ld_global_i((__attribute__((address_space(1))) int *)ip);
  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.shared.i.cta.i32.p3
  // expected-error@+1 {{'__nvvm_cta_ld_shared_i' needs target feature sm_70}}
  __nvvm_cta_ld_shared_i((__attribute__((address_space(3))) int *)ip);
  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.gen.i.acquire.cta.i32.p0
  // expected-error@+1 {{'__nvvm_acquire_cta_ld_gen_i' needs target feature sm_70}}
  __nvvm_acquire_cta_ld_gen_i(ip);
  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.global.i.acquire.cta.i32.p1
  // expected-error@+1 {{'__nvvm_acquire_cta_ld_global_i' needs target feature sm_70}}
  __nvvm_acquire_cta_ld_global_i((__attribute__((address_space(1))) int *)ip);
  // CHECK_SM70_LP64: call i32 @llvm.nvvm.ld.shared.i.acquire.cta.i32.p3
  // expected-error@+1 {{'__nvvm_acquire_cta_ld_shared_i' needs target feature sm_70}}
  __nvvm_acquire_cta_ld_shared_i((__attribute__((address_space(3))) int *)ip);

  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.p0.i32
  // expected-error@+1 {{'__nvvm_st_gen_i' needs target feature sm_70}}
  __nvvm_st_gen_i(ip, i);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.p1.i32
  // expected-error@+1 {{'__nvvm_st_global_i' needs target feature sm_70}}
  __nvvm_st_global_i((__attribute__((address_space(1))) int *)ip, i);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.p3.i32
  // expected-error@+1 {{'__nvvm_st_shared_i' needs target feature sm_70}}
  __nvvm_st_shared_i((__attribute__((address_space(3))) int *)ip, i);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.release.p0.i32
  // expected-error@+1 {{'__nvvm_release_st_gen_i' needs target feature sm_70}}
  __nvvm_release_st_gen_i(ip, i);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.release.p1.i32
  // expected-error@+1 {{'__nvvm_release_st_global_i' needs target feature sm_70}}
  __nvvm_release_st_global_i((__attribute__((address_space(1))) int *)ip, i);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.release.p3.i32
  // expected-error@+1 {{'__nvvm_release_st_shared_i' needs target feature sm_70}}
  __nvvm_release_st_shared_i((__attribute__((address_space(3))) int *)ip, i);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.sys.p0.i32
  // expected-error@+1 {{'__nvvm_sys_st_gen_i' needs target feature sm_70}}
  __nvvm_sys_st_gen_i(ip, i);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.sys.p1.i32
  // expected-error@+1 {{'__nvvm_sys_st_global_i' needs target feature sm_70}}
  __nvvm_sys_st_global_i((__attribute__((address_space(1))) int *)ip, i);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.sys.p3.i32
  // expected-error@+1 {{'__nvvm_sys_st_shared_i' needs target feature sm_70}}
  __nvvm_sys_st_shared_i((__attribute__((address_space(3))) int *)ip, i);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.release.sys.p0.i32
  // expected-error@+1 {{'__nvvm_release_sys_st_gen_i' needs target feature sm_70}}
  __nvvm_release_sys_st_gen_i(ip, i);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.release.sys.p1.i32
  // expected-error@+1 {{'__nvvm_release_sys_st_global_i' needs target feature sm_70}}
  __nvvm_release_sys_st_global_i((__attribute__((address_space(1))) int *)ip, i);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.release.sys.p3.i32
  // expected-error@+1 {{'__nvvm_release_sys_st_shared_i' needs target feature sm_70}}
  __nvvm_release_sys_st_shared_i((__attribute__((address_space(3))) int *)ip, i);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.cta.p0.i32
  // expected-error@+1 {{'__nvvm_cta_st_gen_i' needs target feature sm_70}}
  __nvvm_cta_st_gen_i(ip, i);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.cta.p1.i32
  // expected-error@+1 {{'__nvvm_cta_st_global_i' needs target feature sm_70}}
  __nvvm_cta_st_global_i((__attribute__((address_space(1))) int *)ip, i);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.cta.p3.i32
  // expected-error@+1 {{'__nvvm_cta_st_shared_i' needs target feature sm_70}}
  __nvvm_cta_st_shared_i((__attribute__((address_space(3))) int *)ip, i);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.release.cta.p0.i32
  // expected-error@+1 {{'__nvvm_release_cta_st_gen_i' needs target feature sm_70}}
  __nvvm_release_cta_st_gen_i(ip, i);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.release.cta.p1.i32
  // expected-error@+1 {{'__nvvm_release_cta_st_global_i' needs target feature sm_70}}
  __nvvm_release_cta_st_global_i((__attribute__((address_space(1))) int *)ip, i);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.release.cta.p3.i32
  // expected-error@+1 {{'__nvvm_release_cta_st_shared_i' needs target feature sm_70}}
  __nvvm_release_cta_st_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.gen.i.i64.p0
  // expected-error@+1 {{'__nvvm_ld_gen_l' needs target feature sm_70}}
  __nvvm_ld_gen_l(&dl);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.global.i.i64.p1
  // expected-error@+1 {{'__nvvm_ld_global_l' needs target feature sm_70}}
  __nvvm_ld_global_l((__attribute__((address_space(1))) long *)&dl);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.shared.i.i64.p3
  // expected-error@+1 {{'__nvvm_ld_shared_l' needs target feature sm_70}}
  __nvvm_ld_shared_l((__attribute__((address_space(3))) long *)&dl);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.gen.i.acquire.i64.p0
  // expected-error@+1 {{'__nvvm_acquire_ld_gen_l' needs target feature sm_70}}
  __nvvm_acquire_ld_gen_l(&dl);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.global.i.acquire.i64.p1
  // expected-error@+1 {{'__nvvm_acquire_ld_global_l' needs target feature sm_70}}
  __nvvm_acquire_ld_global_l((__attribute__((address_space(1))) long *)&dl);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.shared.i.acquire.i64.p3
  // expected-error@+1 {{'__nvvm_acquire_ld_shared_l' needs target feature sm_70}}
  __nvvm_acquire_ld_shared_l((__attribute__((address_space(3))) long *)&dl);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_sys_ld_gen_l' needs target feature sm_70}}
  __nvvm_sys_ld_gen_l(&dl);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.global.i.sys.i64.p1
  // expected-error@+1 {{'__nvvm_sys_ld_global_l' needs target feature sm_70}}
  __nvvm_sys_ld_global_l((__attribute__((address_space(1))) long *)&dl);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.shared.i.sys.i64.p3
  // expected-error@+1 {{'__nvvm_sys_ld_shared_l' needs target feature sm_70}}
  __nvvm_sys_ld_shared_l((__attribute__((address_space(3))) long *)&dl);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.gen.i.acquire.sys.i64.p0
  // expected-error@+1 {{'__nvvm_acquire_sys_ld_gen_l' needs target feature sm_70}}
  __nvvm_acquire_sys_ld_gen_l(&dl);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.global.i.acquire.sys.i64.p1
  // expected-error@+1 {{'__nvvm_acquire_sys_ld_global_l' needs target feature sm_70}}
  __nvvm_acquire_sys_ld_global_l((__attribute__((address_space(1))) long *)&dl);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.shared.i.acquire.sys.i64.p3
  // expected-error@+1 {{'__nvvm_acquire_sys_ld_shared_l' needs target feature sm_70}}
  __nvvm_acquire_sys_ld_shared_l((__attribute__((address_space(3))) long *)&dl);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_cta_ld_gen_l' needs target feature sm_70}}
  __nvvm_cta_ld_gen_l(&dl);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.global.i.cta.i64.p1
  // expected-error@+1 {{'__nvvm_cta_ld_global_l' needs target feature sm_70}}
  __nvvm_cta_ld_global_l((__attribute__((address_space(1))) long *)&dl);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.shared.i.cta.i64.p3
  // expected-error@+1 {{'__nvvm_cta_ld_shared_l' needs target feature sm_70}}
  __nvvm_cta_ld_shared_l((__attribute__((address_space(3))) long *)&dl);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.gen.i.acquire.cta.i64.p0
  // expected-error@+1 {{'__nvvm_acquire_cta_ld_gen_l' needs target feature sm_70}}
  __nvvm_acquire_cta_ld_gen_l(&dl);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.global.i.acquire.cta.i64.p1
  // expected-error@+1 {{'__nvvm_acquire_cta_ld_global_l' needs target feature sm_70}}
  __nvvm_acquire_cta_ld_global_l((__attribute__((address_space(1))) long *)&dl);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.shared.i.acquire.cta.i64.p3
  // expected-error@+1 {{'__nvvm_acquire_cta_ld_shared_l' needs target feature sm_70}}
  __nvvm_acquire_cta_ld_shared_l((__attribute__((address_space(3))) long *)&dl);

  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.p0.i64
  // expected-error@+1 {{'__nvvm_st_gen_l' needs target feature sm_70}}
  __nvvm_st_gen_l(&dl, l);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.p1.i64
  // expected-error@+1 {{'__nvvm_st_global_l' needs target feature sm_70}}
  __nvvm_st_global_l((__attribute__((address_space(1))) long *)&dl, l);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.p3.i64
  // expected-error@+1 {{'__nvvm_st_shared_l' needs target feature sm_70}}
  __nvvm_st_shared_l((__attribute__((address_space(3))) long *)&dl, l);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.release.p0.i64
  // expected-error@+1 {{'__nvvm_release_st_gen_l' needs target feature sm_70}}
  __nvvm_release_st_gen_l(&dl, l);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.release.p1.i64
  // expected-error@+1 {{'__nvvm_release_st_global_l' needs target feature sm_70}}
  __nvvm_release_st_global_l((__attribute__((address_space(1))) long *)&dl, l);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.release.p3.i64
  // expected-error@+1 {{'__nvvm_release_st_shared_l' needs target feature sm_70}}
  __nvvm_release_st_shared_l((__attribute__((address_space(3))) long *)&dl, l);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.sys.p0.i64
  // expected-error@+1 {{'__nvvm_sys_st_gen_l' needs target feature sm_70}}
  __nvvm_sys_st_gen_l(&dl, l);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.sys.p1.i64
  // expected-error@+1 {{'__nvvm_sys_st_global_l' needs target feature sm_70}}
  __nvvm_sys_st_global_l((__attribute__((address_space(1))) long *)&dl, l);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.sys.p3.i64
  // expected-error@+1 {{'__nvvm_sys_st_shared_l' needs target feature sm_70}}
  __nvvm_sys_st_shared_l((__attribute__((address_space(3))) long *)&dl, l);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.release.sys.p0.i64
  // expected-error@+1 {{'__nvvm_release_sys_st_gen_l' needs target feature sm_70}}
  __nvvm_release_sys_st_gen_l(&dl, l);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.release.sys.p1.i64
  // expected-error@+1 {{'__nvvm_release_sys_st_global_l' needs target feature sm_70}}
  __nvvm_release_sys_st_global_l((__attribute__((address_space(1))) long *)&dl, l);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.release.sys.p3.i64
  // expected-error@+1 {{'__nvvm_release_sys_st_shared_l' needs target feature sm_70}}
  __nvvm_release_sys_st_shared_l((__attribute__((address_space(3))) long *)&dl, l);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.cta.p0.i64
  // expected-error@+1 {{'__nvvm_cta_st_gen_l' needs target feature sm_70}}
  __nvvm_cta_st_gen_l(&dl, l);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.cta.p1.i64
  // expected-error@+1 {{'__nvvm_cta_st_global_l' needs target feature sm_70}}
  __nvvm_cta_st_global_l((__attribute__((address_space(1))) long *)&dl, l);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.cta.p3.i64
  // expected-error@+1 {{'__nvvm_cta_st_shared_l' needs target feature sm_70}}
  __nvvm_cta_st_shared_l((__attribute__((address_space(3))) long *)&dl, l);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.release.cta.p0.i64
  // expected-error@+1 {{'__nvvm_release_cta_st_gen_l' needs target feature sm_70}}
  __nvvm_release_cta_st_gen_l(&dl, l);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.release.cta.p1.i64
  // expected-error@+1 {{'__nvvm_release_cta_st_global_l' needs target feature sm_70}}
  __nvvm_release_cta_st_global_l((__attribute__((address_space(1))) long *)&dl, l);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.release.cta.p3.i64
  // expected-error@+1 {{'__nvvm_release_cta_st_shared_l' needs target feature sm_70}}
  __nvvm_release_cta_st_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.gen.i.i64.p0
  // expected-error@+1 {{'__nvvm_ld_gen_ll' needs target feature sm_70}}
  __nvvm_ld_gen_ll(&dll);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.global.i.i64.p1
  // expected-error@+1 {{'__nvvm_ld_global_ll' needs target feature sm_70}}
  __nvvm_ld_global_ll((__attribute__((address_space(1))) long long *)&dll);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.shared.i.i64.p3
  // expected-error@+1 {{'__nvvm_ld_shared_ll' needs target feature sm_70}}
  __nvvm_ld_shared_ll((__attribute__((address_space(3))) long long *)&dll);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.gen.i.acquire.i64.p0
  // expected-error@+1 {{'__nvvm_acquire_ld_gen_ll' needs target feature sm_70}}
  __nvvm_acquire_ld_gen_ll(&dll);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.global.i.acquire.i64.p1
  // expected-error@+1 {{'__nvvm_acquire_ld_global_ll' needs target feature sm_70}}
  __nvvm_acquire_ld_global_ll((__attribute__((address_space(1))) long long *)&dll);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.shared.i.acquire.i64.p3
  // expected-error@+1 {{'__nvvm_acquire_ld_shared_ll' needs target feature sm_70}}
  __nvvm_acquire_ld_shared_ll((__attribute__((address_space(3))) long long *)&dll);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.gen.i.sys.i64.p0
  // expected-error@+1 {{'__nvvm_sys_ld_gen_ll' needs target feature sm_70}}
  __nvvm_sys_ld_gen_ll(&dll);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.global.i.sys.i64.p1
  // expected-error@+1 {{'__nvvm_sys_ld_global_ll' needs target feature sm_70}}
  __nvvm_sys_ld_global_ll((__attribute__((address_space(1))) long long *)&dll);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.shared.i.sys.i64.p3
  // expected-error@+1 {{'__nvvm_sys_ld_shared_ll' needs target feature sm_70}}
  __nvvm_sys_ld_shared_ll((__attribute__((address_space(3))) long long *)&dll);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.gen.i.acquire.sys.i64.p0
  // expected-error@+1 {{'__nvvm_acquire_sys_ld_gen_ll' needs target feature sm_70}}
  __nvvm_acquire_sys_ld_gen_ll(&dll);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.global.i.acquire.sys.i64.p1
  // expected-error@+1 {{'__nvvm_acquire_sys_ld_global_ll' needs target feature sm_70}}
  __nvvm_acquire_sys_ld_global_ll((__attribute__((address_space(1))) long long *)&dll);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.shared.i.acquire.sys.i64.p3
  // expected-error@+1 {{'__nvvm_acquire_sys_ld_shared_ll' needs target feature sm_70}}
  __nvvm_acquire_sys_ld_shared_ll((__attribute__((address_space(3))) long long *)&dll);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.gen.i.cta.i64.p0
  // expected-error@+1 {{'__nvvm_cta_ld_gen_ll' needs target feature sm_70}}
  __nvvm_cta_ld_gen_ll(&dll);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.global.i.cta.i64.p1
  // expected-error@+1 {{'__nvvm_cta_ld_global_ll' needs target feature sm_70}}
  __nvvm_cta_ld_global_ll((__attribute__((address_space(1))) long long *)&dll);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.shared.i.cta.i64.p3
  // expected-error@+1 {{'__nvvm_cta_ld_shared_ll' needs target feature sm_70}}
  __nvvm_cta_ld_shared_ll((__attribute__((address_space(3))) long long *)&dll);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.gen.i.acquire.cta.i64.p0
  // expected-error@+1 {{'__nvvm_acquire_cta_ld_gen_ll' needs target feature sm_70}}
  __nvvm_acquire_cta_ld_gen_ll(&dll);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.global.i.acquire.cta.i64.p1
  // expected-error@+1 {{'__nvvm_acquire_cta_ld_global_ll' needs target feature sm_70}}
  __nvvm_acquire_cta_ld_global_ll((__attribute__((address_space(1))) long long *)&dll);
  // CHECK_SM70_LP64: call i64 @llvm.nvvm.ld.shared.i.acquire.cta.i64.p3
  // expected-error@+1 {{'__nvvm_acquire_cta_ld_shared_ll' needs target feature sm_70}}
  __nvvm_acquire_cta_ld_shared_ll((__attribute__((address_space(3))) long long *)&dll);

  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.p0.i64
  // expected-error@+1 {{'__nvvm_st_gen_ll' needs target feature sm_70}}
  __nvvm_st_gen_ll(&dll, ll);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.p1.i64
  // expected-error@+1 {{'__nvvm_st_global_ll' needs target feature sm_70}}
  __nvvm_st_global_ll((__attribute__((address_space(1))) long long *)&dll, ll);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.p3.i64
  // expected-error@+1 {{'__nvvm_st_shared_ll' needs target feature sm_70}}
  __nvvm_st_shared_ll((__attribute__((address_space(3))) long long *)&dll, ll);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.release.p0.i64
  // expected-error@+1 {{'__nvvm_release_st_gen_ll' needs target feature sm_70}}
  __nvvm_release_st_gen_ll(&dll, ll);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.release.p1.i64
  // expected-error@+1 {{'__nvvm_release_st_global_ll' needs target feature sm_70}}
  __nvvm_release_st_global_ll((__attribute__((address_space(1))) long long *)&dll, ll);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.release.p3.i64
  // expected-error@+1 {{'__nvvm_release_st_shared_ll' needs target feature sm_70}}
  __nvvm_release_st_shared_ll((__attribute__((address_space(3))) long long *)&dll, ll);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.sys.p0.i64
  // expected-error@+1 {{'__nvvm_sys_st_gen_ll' needs target feature sm_70}}
  __nvvm_sys_st_gen_ll(&dll, ll);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.sys.p1.i64
  // expected-error@+1 {{'__nvvm_sys_st_global_ll' needs target feature sm_70}}
  __nvvm_sys_st_global_ll((__attribute__((address_space(1))) long long *)&dll, ll);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.sys.p3.i64
  // expected-error@+1 {{'__nvvm_sys_st_shared_ll' needs target feature sm_70}}
  __nvvm_sys_st_shared_ll((__attribute__((address_space(3))) long long *)&dll, ll);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.release.sys.p0.i64
  // expected-error@+1 {{'__nvvm_release_sys_st_gen_ll' needs target feature sm_70}}
  __nvvm_release_sys_st_gen_ll(&dll, ll);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.release.sys.p1.i64
  // expected-error@+1 {{'__nvvm_release_sys_st_global_ll' needs target feature sm_70}}
  __nvvm_release_sys_st_global_ll((__attribute__((address_space(1))) long long *)&dll, ll);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.release.sys.p3.i64
  // expected-error@+1 {{'__nvvm_release_sys_st_shared_ll' needs target feature sm_70}}
  __nvvm_release_sys_st_shared_ll((__attribute__((address_space(3))) long long *)&dll, ll);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.cta.p0.i64
  // expected-error@+1 {{'__nvvm_cta_st_gen_ll' needs target feature sm_70}}
  __nvvm_cta_st_gen_ll(&dll, ll);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.cta.p1.i64
  // expected-error@+1 {{'__nvvm_cta_st_global_ll' needs target feature sm_70}}
  __nvvm_cta_st_global_ll((__attribute__((address_space(1))) long long *)&dll, ll);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.cta.p3.i64
  // expected-error@+1 {{'__nvvm_cta_st_shared_ll' needs target feature sm_70}}
  __nvvm_cta_st_shared_ll((__attribute__((address_space(3))) long long *)&dll, ll);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.i.release.cta.p0.i64
  // expected-error@+1 {{'__nvvm_release_cta_st_gen_ll' needs target feature sm_70}}
  __nvvm_release_cta_st_gen_ll(&dll, ll);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.i.release.cta.p1.i64
  // expected-error@+1 {{'__nvvm_release_cta_st_global_ll' needs target feature sm_70}}
  __nvvm_release_cta_st_global_ll((__attribute__((address_space(1))) long long *)&dll, ll);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.i.release.cta.p3.i64
  // expected-error@+1 {{'__nvvm_release_cta_st_shared_ll' needs target feature sm_70}}
  __nvvm_release_cta_st_shared_ll((__attribute__((address_space(3))) long long *)&dll, ll);

  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.gen.f.f32.p0
  // expected-error@+1 {{'__nvvm_ld_gen_f' needs target feature sm_70}}
  __nvvm_ld_gen_f(fp);
  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.global.f.f32.p1
  // expected-error@+1 {{'__nvvm_ld_global_f' needs target feature sm_70}}
  __nvvm_ld_global_f((__attribute__((address_space(1))) float *)fp);
  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.shared.f.f32.p3
  // expected-error@+1 {{'__nvvm_ld_shared_f' needs target feature sm_70}}
  __nvvm_ld_shared_f((__attribute__((address_space(3))) float *)fp);
  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.gen.f.acquire.f32.p0
  // expected-error@+1 {{'__nvvm_acquire_ld_gen_f' needs target feature sm_70}}
  __nvvm_acquire_ld_gen_f(fp);
  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.global.f.acquire.f32.p1
  // expected-error@+1 {{'__nvvm_acquire_ld_global_f' needs target feature sm_70}}
  __nvvm_acquire_ld_global_f((__attribute__((address_space(1))) float *)fp);
  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.shared.f.acquire.f32.p3
  // expected-error@+1 {{'__nvvm_acquire_ld_shared_f' needs target feature sm_70}}
  __nvvm_acquire_ld_shared_f((__attribute__((address_space(3))) float *)fp);
  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.gen.f.sys.f32.p0
  // expected-error@+1 {{'__nvvm_sys_ld_gen_f' needs target feature sm_70}}
  __nvvm_sys_ld_gen_f(fp);
  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.global.f.sys.f32.p1
  // expected-error@+1 {{'__nvvm_sys_ld_global_f' needs target feature sm_70}}
  __nvvm_sys_ld_global_f((__attribute__((address_space(1))) float *)fp);
  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.shared.f.sys.f32.p3
  // expected-error@+1 {{'__nvvm_sys_ld_shared_f' needs target feature sm_70}}
  __nvvm_sys_ld_shared_f((__attribute__((address_space(3))) float *)fp);
  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.gen.f.acquire.sys.f32.p0
  // expected-error@+1 {{'__nvvm_acquire_sys_ld_gen_f' needs target feature sm_70}}
  __nvvm_acquire_sys_ld_gen_f(fp);
  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.global.f.acquire.sys.f32.p1
  // expected-error@+1 {{'__nvvm_acquire_sys_ld_global_f' needs target feature sm_70}}
  __nvvm_acquire_sys_ld_global_f((__attribute__((address_space(1))) float *)fp);
  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.shared.f.acquire.sys.f32.p3
  // expected-error@+1 {{'__nvvm_acquire_sys_ld_shared_f' needs target feature sm_70}}
  __nvvm_acquire_sys_ld_shared_f((__attribute__((address_space(3))) float *)fp);
  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.gen.f.cta.f32.p0
  // expected-error@+1 {{'__nvvm_cta_ld_gen_f' needs target feature sm_70}}
  __nvvm_cta_ld_gen_f(fp);
  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.global.f.cta.f32.p1
  // expected-error@+1 {{'__nvvm_cta_ld_global_f' needs target feature sm_70}}
  __nvvm_cta_ld_global_f((__attribute__((address_space(1))) float *)fp);
  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.shared.f.cta.f32.p3
  // expected-error@+1 {{'__nvvm_cta_ld_shared_f' needs target feature sm_70}}
  __nvvm_cta_ld_shared_f((__attribute__((address_space(3))) float *)fp);
  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.gen.f.acquire.cta.f32.p0
  // expected-error@+1 {{'__nvvm_acquire_cta_ld_gen_f' needs target feature sm_70}}
  __nvvm_acquire_cta_ld_gen_f(fp);
  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.global.f.acquire.cta.f32.p1
  // expected-error@+1 {{'__nvvm_acquire_cta_ld_global_f' needs target feature sm_70}}
  __nvvm_acquire_cta_ld_global_f((__attribute__((address_space(1))) float *)fp);
  // CHECK_SM70_LP64: call float @llvm.nvvm.ld.shared.f.acquire.cta.f32.p3
  // expected-error@+1 {{'__nvvm_acquire_cta_ld_shared_f' needs target feature sm_70}}
  __nvvm_acquire_cta_ld_shared_f((__attribute__((address_space(3))) float *)fp);

  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.f.p0.f32
  // expected-error@+1 {{'__nvvm_st_gen_f' needs target feature sm_70}}
  __nvvm_st_gen_f(fp, f);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.f.p1.f32
  // expected-error@+1 {{'__nvvm_st_global_f' needs target feature sm_70}}
  __nvvm_st_global_f((__attribute__((address_space(1))) float *)fp, f);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.f.p3.f32
  // expected-error@+1 {{'__nvvm_st_shared_f' needs target feature sm_70}}
  __nvvm_st_shared_f((__attribute__((address_space(3))) float *)fp, f);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.f.release.p0.f32
  // expected-error@+1 {{'__nvvm_release_st_gen_f' needs target feature sm_70}}
  __nvvm_release_st_gen_f(fp, f);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.f.release.p1.f32
  // expected-error@+1 {{'__nvvm_release_st_global_f' needs target feature sm_70}}
  __nvvm_release_st_global_f((__attribute__((address_space(1))) float *)fp, f);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.f.release.p3.f32
  // expected-error@+1 {{'__nvvm_release_st_shared_f' needs target feature sm_70}}
  __nvvm_release_st_shared_f((__attribute__((address_space(3))) float *)fp, f);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.f.sys.p0.f32
  // expected-error@+1 {{'__nvvm_sys_st_gen_f' needs target feature sm_70}}
  __nvvm_sys_st_gen_f(fp, f);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.f.sys.p1.f32
  // expected-error@+1 {{'__nvvm_sys_st_global_f' needs target feature sm_70}}
  __nvvm_sys_st_global_f((__attribute__((address_space(1))) float *)fp, f);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.f.sys.p3.f32
  // expected-error@+1 {{'__nvvm_sys_st_shared_f' needs target feature sm_70}}
  __nvvm_sys_st_shared_f((__attribute__((address_space(3))) float *)fp, f);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.f.release.sys.p0.f32
  // expected-error@+1 {{'__nvvm_release_sys_st_gen_f' needs target feature sm_70}}
  __nvvm_release_sys_st_gen_f(fp, f);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.f.release.sys.p1.f32
  // expected-error@+1 {{'__nvvm_release_sys_st_global_f' needs target feature sm_70}}
  __nvvm_release_sys_st_global_f((__attribute__((address_space(1))) float *)fp, f);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.f.release.sys.p3.f32
  // expected-error@+1 {{'__nvvm_release_sys_st_shared_f' needs target feature sm_70}}
  __nvvm_release_sys_st_shared_f((__attribute__((address_space(3))) float *)fp, f);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.f.cta.p0.f32
  // expected-error@+1 {{'__nvvm_cta_st_gen_f' needs target feature sm_70}}
  __nvvm_cta_st_gen_f(fp, f);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.f.cta.p1.f32
  // expected-error@+1 {{'__nvvm_cta_st_global_f' needs target feature sm_70}}
  __nvvm_cta_st_global_f((__attribute__((address_space(1))) float *)fp, f);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.f.cta.p3.f32
  // expected-error@+1 {{'__nvvm_cta_st_shared_f' needs target feature sm_70}}
  __nvvm_cta_st_shared_f((__attribute__((address_space(3))) float *)fp, f);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.f.release.cta.p0.f32
  // expected-error@+1 {{'__nvvm_release_cta_st_gen_f' needs target feature sm_70}}
  __nvvm_release_cta_st_gen_f(fp, f);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.f.release.cta.p1.f32
  // expected-error@+1 {{'__nvvm_release_cta_st_global_f' needs target feature sm_70}}
  __nvvm_release_cta_st_global_f((__attribute__((address_space(1))) float *)fp, f);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.f.release.cta.p3.f32
  // expected-error@+1 {{'__nvvm_release_cta_st_shared_f' needs target feature sm_70}}
  __nvvm_release_cta_st_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.gen.f.f64.p0
  // expected-error@+1 {{'__nvvm_ld_gen_d' needs target feature sm_70}}
  __nvvm_ld_gen_d(dfp);
  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.global.f.f64.p1
  // expected-error@+1 {{'__nvvm_ld_global_d' needs target feature sm_70}}
  __nvvm_ld_global_d((__attribute__((address_space(1))) double *)dfp);
  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.shared.f.f64.p3
  // expected-error@+1 {{'__nvvm_ld_shared_d' needs target feature sm_70}}
  __nvvm_ld_shared_d((__attribute__((address_space(3))) double *)dfp);
  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.gen.f.acquire.f64.p0
  // expected-error@+1 {{'__nvvm_acquire_ld_gen_d' needs target feature sm_70}}
  __nvvm_acquire_ld_gen_d(dfp);
  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.global.f.acquire.f64.p1
  // expected-error@+1 {{'__nvvm_acquire_ld_global_d' needs target feature sm_70}}
  __nvvm_acquire_ld_global_d((__attribute__((address_space(1))) double *)dfp);
  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.shared.f.acquire.f64.p3
  // expected-error@+1 {{'__nvvm_acquire_ld_shared_d' needs target feature sm_70}}
  __nvvm_acquire_ld_shared_d((__attribute__((address_space(3))) double *)dfp);
  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.gen.f.sys.f64.p0
  // expected-error@+1 {{'__nvvm_sys_ld_gen_d' needs target feature sm_70}}
  __nvvm_sys_ld_gen_d(dfp);
  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.global.f.sys.f64.p1
  // expected-error@+1 {{'__nvvm_sys_ld_global_d' needs target feature sm_70}}
  __nvvm_sys_ld_global_d((__attribute__((address_space(1))) double *)dfp);
  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.shared.f.sys.f64.p3
  // expected-error@+1 {{'__nvvm_sys_ld_shared_d' needs target feature sm_70}}
  __nvvm_sys_ld_shared_d((__attribute__((address_space(3))) double *)dfp);
  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.gen.f.acquire.sys.f64.p0
  // expected-error@+1 {{'__nvvm_acquire_sys_ld_gen_d' needs target feature sm_70}}
  __nvvm_acquire_sys_ld_gen_d(dfp);
  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.global.f.acquire.sys.f64.p1
  // expected-error@+1 {{'__nvvm_acquire_sys_ld_global_d' needs target feature sm_70}}
  __nvvm_acquire_sys_ld_global_d((__attribute__((address_space(1))) double *)dfp);
  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.shared.f.acquire.sys.f64.p3
  // expected-error@+1 {{'__nvvm_acquire_sys_ld_shared_d' needs target feature sm_70}}
  __nvvm_acquire_sys_ld_shared_d((__attribute__((address_space(3))) double *)dfp);
  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.gen.f.cta.f64.p0
  // expected-error@+1 {{'__nvvm_cta_ld_gen_d' needs target feature sm_70}}
  __nvvm_cta_ld_gen_d(dfp);
  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.global.f.cta.f64.p1
  // expected-error@+1 {{'__nvvm_cta_ld_global_d' needs target feature sm_70}}
  __nvvm_cta_ld_global_d((__attribute__((address_space(1))) double *)dfp);
  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.shared.f.cta.f64.p3
  // expected-error@+1 {{'__nvvm_cta_ld_shared_d' needs target feature sm_70}}
  __nvvm_cta_ld_shared_d((__attribute__((address_space(3))) double *)dfp);
  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.gen.f.acquire.cta.f64.p0
  // expected-error@+1 {{'__nvvm_acquire_cta_ld_gen_d' needs target feature sm_70}}
  __nvvm_acquire_cta_ld_gen_d(dfp);
  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.global.f.acquire.cta.f64.p1
  // expected-error@+1 {{'__nvvm_acquire_cta_ld_global_d' needs target feature sm_70}}
  __nvvm_acquire_cta_ld_global_d((__attribute__((address_space(1))) double *)dfp);
  // CHECK_SM70_LP64: call double @llvm.nvvm.ld.shared.f.acquire.cta.f64.p3
  // expected-error@+1 {{'__nvvm_acquire_cta_ld_shared_d' needs target feature sm_70}}
  __nvvm_acquire_cta_ld_shared_d((__attribute__((address_space(3))) double *)dfp);

  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.f.p0.f64
  // expected-error@+1 {{'__nvvm_st_gen_d' needs target feature sm_70}}
  __nvvm_st_gen_d(dfp, df);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.f.p1.f64
  // expected-error@+1 {{'__nvvm_st_global_d' needs target feature sm_70}}
  __nvvm_st_global_d((__attribute__((address_space(1))) double *)dfp, df);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.f.p3.f64
  // expected-error@+1 {{'__nvvm_st_shared_d' needs target feature sm_70}}
  __nvvm_st_shared_d((__attribute__((address_space(3))) double *)dfp, df);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.f.release.p0.f64
  // expected-error@+1 {{'__nvvm_release_st_gen_d' needs target feature sm_70}}
  __nvvm_release_st_gen_d(dfp, df);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.f.release.p1.f64
  // expected-error@+1 {{'__nvvm_release_st_global_d' needs target feature sm_70}}
  __nvvm_release_st_global_d((__attribute__((address_space(1))) double *)dfp, df);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.f.release.p3.f64
  // expected-error@+1 {{'__nvvm_release_st_shared_d' needs target feature sm_70}}
  __nvvm_release_st_shared_d((__attribute__((address_space(3))) double *)dfp, df);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.f.sys.p0.f64
  // expected-error@+1 {{'__nvvm_sys_st_gen_d' needs target feature sm_70}}
  __nvvm_sys_st_gen_d(dfp, df);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.f.sys.p1.f64
  // expected-error@+1 {{'__nvvm_sys_st_global_d' needs target feature sm_70}}
  __nvvm_sys_st_global_d((__attribute__((address_space(1))) double *)dfp, df);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.f.sys.p3.f64
  // expected-error@+1 {{'__nvvm_sys_st_shared_d' needs target feature sm_70}}
  __nvvm_sys_st_shared_d((__attribute__((address_space(3))) double *)dfp, df);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.f.release.sys.p0.f64
  // expected-error@+1 {{'__nvvm_release_sys_st_gen_d' needs target feature sm_70}}
  __nvvm_release_sys_st_gen_d(dfp, df);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.f.release.sys.p1.f64
  // expected-error@+1 {{'__nvvm_release_sys_st_global_d' needs target feature sm_70}}
  __nvvm_release_sys_st_global_d((__attribute__((address_space(1))) double *)dfp, df);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.f.release.sys.p3.f64
  // expected-error@+1 {{'__nvvm_release_sys_st_shared_d' needs target feature sm_70}}
  __nvvm_release_sys_st_shared_d((__attribute__((address_space(3))) double *)dfp, df);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.f.cta.p0.f64
  // expected-error@+1 {{'__nvvm_cta_st_gen_d' needs target feature sm_70}}
  __nvvm_cta_st_gen_d(dfp, df);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.f.cta.p1.f64
  // expected-error@+1 {{'__nvvm_cta_st_global_d' needs target feature sm_70}}
  __nvvm_cta_st_global_d((__attribute__((address_space(1))) double *)dfp, df);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.f.cta.p3.f64
  // expected-error@+1 {{'__nvvm_cta_st_shared_d' needs target feature sm_70}}
  __nvvm_cta_st_shared_d((__attribute__((address_space(3))) double *)dfp, df);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.gen.f.release.cta.p0.f64
  // expected-error@+1 {{'__nvvm_release_cta_st_gen_d' needs target feature sm_70}}
  __nvvm_release_cta_st_gen_d(dfp, df);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.global.f.release.cta.p1.f64
  // expected-error@+1 {{'__nvvm_release_cta_st_global_d' needs target feature sm_70}}
  __nvvm_release_cta_st_global_d((__attribute__((address_space(1))) double *)dfp, df);
  // CHECK_SM70_LP64: call void @llvm.nvvm.st.shared.f.release.cta.p3.f64
  // expected-error@+1 {{'__nvvm_release_cta_st_shared_d' needs target feature sm_70}}
  __nvvm_release_cta_st_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.gen.i.acquire.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_add_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_add_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.global.i.acquire.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_add_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_add_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.shared.i.acquire.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_add_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_add_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.gen.i.release.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_add_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_add_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.global.i.release.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_add_global_i' needs target feature sm_70}}
  __nvvm_atom_release_add_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.shared.i.release.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_add_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_add_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.gen.i.acq.rel.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_add_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_add_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.global.i.acq.rel.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_add_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_add_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.shared.i.acq.rel.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_add_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_add_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.gen.i.acquire.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_add_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_add_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.global.i.acquire.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_add_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_add_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.shared.i.acquire.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_add_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_add_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.gen.i.release.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_add_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_add_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.global.i.release.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_add_global_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_add_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.shared.i.release.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_add_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_add_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.gen.i.acq.rel.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_add_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_add_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.global.i.acq.rel.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_add_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_add_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.shared.i.acq.rel.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_add_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_add_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.gen.i.acquire.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_add_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_add_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.global.i.acquire.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_add_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_add_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.shared.i.acquire.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_add_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_add_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.gen.i.release.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_add_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_add_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.global.i.release.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_add_global_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_add_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.shared.i.release.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_add_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_add_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.gen.i.acq.rel.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_add_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_add_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.global.i.acq.rel.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_add_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_add_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.add.shared.i.acq.rel.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_add_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_add_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.gen.i.acquire.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_add_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_add_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.global.i.acquire.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_add_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_add_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.shared.i.acquire.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_add_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_add_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.gen.i.release.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_add_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_add_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.global.i.release.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_add_global_l' needs target feature sm_70}}
  __nvvm_atom_release_add_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.shared.i.release.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_add_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_add_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.gen.i.acq.rel.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_add_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_add_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.global.i.acq.rel.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_add_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_add_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.shared.i.acq.rel.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_add_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_add_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.gen.i.acquire.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_add_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_add_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.global.i.acquire.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_add_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_add_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.shared.i.acquire.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_add_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_add_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.gen.i.release.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_add_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_add_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.global.i.release.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_add_global_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_add_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.shared.i.release.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_add_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_add_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.gen.i.acq.rel.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_add_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_add_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.global.i.acq.rel.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_add_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_add_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.shared.i.acq.rel.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_add_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_add_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.gen.i.acquire.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_add_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_add_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.global.i.acquire.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_add_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_add_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.shared.i.acquire.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_add_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_add_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.gen.i.release.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_add_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_add_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.global.i.release.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_add_global_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_add_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.shared.i.release.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_add_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_add_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.gen.i.acq.rel.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_add_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_add_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.global.i.acq.rel.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_add_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_add_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.add.shared.i.acq.rel.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_add_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_add_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.gen.f.acquire.f32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_add_gen_f' needs target feature sm_70}}
  __nvvm_atom_acquire_add_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.global.f.acquire.f32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_add_global_f' needs target feature sm_70}}
  __nvvm_atom_acquire_add_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.shared.f.acquire.f32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_add_shared_f' needs target feature sm_70}}
  __nvvm_atom_acquire_add_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.gen.f.release.f32.p0
  // expected-error@+1 {{'__nvvm_atom_release_add_gen_f' needs target feature sm_70}}
  __nvvm_atom_release_add_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.global.f.release.f32.p1
  // expected-error@+1 {{'__nvvm_atom_release_add_global_f' needs target feature sm_70}}
  __nvvm_atom_release_add_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.shared.f.release.f32.p3
  // expected-error@+1 {{'__nvvm_atom_release_add_shared_f' needs target feature sm_70}}
  __nvvm_atom_release_add_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.gen.f.acq.rel.f32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_add_gen_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_add_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.global.f.acq.rel.f32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_add_global_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_add_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.shared.f.acq.rel.f32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_add_shared_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_add_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.gen.f.acquire.sys.f32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_add_gen_f' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_add_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.global.f.acquire.sys.f32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_add_global_f' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_add_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.shared.f.acquire.sys.f32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_add_shared_f' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_add_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.gen.f.release.sys.f32.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_add_gen_f' needs target feature sm_70}}
  __nvvm_atom_release_sys_add_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.global.f.release.sys.f32.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_add_global_f' needs target feature sm_70}}
  __nvvm_atom_release_sys_add_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.shared.f.release.sys.f32.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_add_shared_f' needs target feature sm_70}}
  __nvvm_atom_release_sys_add_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.gen.f.acq.rel.sys.f32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_add_gen_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_add_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.global.f.acq.rel.sys.f32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_add_global_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_add_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.shared.f.acq.rel.sys.f32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_add_shared_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_add_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.gen.f.acquire.cta.f32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_add_gen_f' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_add_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.global.f.acquire.cta.f32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_add_global_f' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_add_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.shared.f.acquire.cta.f32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_add_shared_f' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_add_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.gen.f.release.cta.f32.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_add_gen_f' needs target feature sm_70}}
  __nvvm_atom_release_cta_add_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.global.f.release.cta.f32.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_add_global_f' needs target feature sm_70}}
  __nvvm_atom_release_cta_add_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.shared.f.release.cta.f32.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_add_shared_f' needs target feature sm_70}}
  __nvvm_atom_release_cta_add_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.gen.f.acq.rel.cta.f32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_add_gen_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_add_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.global.f.acq.rel.cta.f32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_add_global_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_add_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.add.shared.f.acq.rel.cta.f32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_add_shared_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_add_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.gen.f.acquire.f64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_add_gen_d' needs target feature sm_70}}
  __nvvm_atom_acquire_add_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.global.f.acquire.f64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_add_global_d' needs target feature sm_70}}
  __nvvm_atom_acquire_add_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.shared.f.acquire.f64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_add_shared_d' needs target feature sm_70}}
  __nvvm_atom_acquire_add_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.gen.f.release.f64.p0
  // expected-error@+1 {{'__nvvm_atom_release_add_gen_d' needs target feature sm_70}}
  __nvvm_atom_release_add_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.global.f.release.f64.p1
  // expected-error@+1 {{'__nvvm_atom_release_add_global_d' needs target feature sm_70}}
  __nvvm_atom_release_add_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.shared.f.release.f64.p3
  // expected-error@+1 {{'__nvvm_atom_release_add_shared_d' needs target feature sm_70}}
  __nvvm_atom_release_add_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.gen.f.acq.rel.f64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_add_gen_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_add_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.global.f.acq.rel.f64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_add_global_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_add_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.shared.f.acq.rel.f64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_add_shared_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_add_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.gen.f.acquire.sys.f64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_add_gen_d' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_add_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.global.f.acquire.sys.f64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_add_global_d' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_add_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.shared.f.acquire.sys.f64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_add_shared_d' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_add_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.gen.f.release.sys.f64.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_add_gen_d' needs target feature sm_70}}
  __nvvm_atom_release_sys_add_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.global.f.release.sys.f64.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_add_global_d' needs target feature sm_70}}
  __nvvm_atom_release_sys_add_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.shared.f.release.sys.f64.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_add_shared_d' needs target feature sm_70}}
  __nvvm_atom_release_sys_add_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.gen.f.acq.rel.sys.f64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_add_gen_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_add_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.global.f.acq.rel.sys.f64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_add_global_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_add_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.shared.f.acq.rel.sys.f64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_add_shared_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_add_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.gen.f.acquire.cta.f64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_add_gen_d' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_add_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.global.f.acquire.cta.f64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_add_global_d' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_add_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.shared.f.acquire.cta.f64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_add_shared_d' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_add_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.gen.f.release.cta.f64.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_add_gen_d' needs target feature sm_70}}
  __nvvm_atom_release_cta_add_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.global.f.release.cta.f64.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_add_global_d' needs target feature sm_70}}
  __nvvm_atom_release_cta_add_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.shared.f.release.cta.f64.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_add_shared_d' needs target feature sm_70}}
  __nvvm_atom_release_cta_add_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.gen.f.acq.rel.cta.f64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_add_gen_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_add_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.global.f.acq.rel.cta.f64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_add_global_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_add_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.add.shared.f.acq.rel.cta.f64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_add_shared_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_add_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.gen.i.acquire.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_xchg_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_xchg_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.global.i.acquire.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_xchg_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_xchg_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.shared.i.acquire.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_xchg_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_xchg_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.gen.i.release.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_xchg_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_xchg_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.global.i.release.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_xchg_global_i' needs target feature sm_70}}
  __nvvm_atom_release_xchg_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.shared.i.release.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_xchg_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_xchg_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.gen.i.acq.rel.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xchg_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xchg_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.global.i.acq.rel.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xchg_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xchg_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.shared.i.acq.rel.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xchg_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xchg_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.gen.i.acquire.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xchg_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xchg_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.global.i.acquire.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xchg_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xchg_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.shared.i.acquire.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xchg_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xchg_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.gen.i.release.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_xchg_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_xchg_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.global.i.release.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_xchg_global_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_xchg_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.shared.i.release.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_xchg_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_xchg_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.gen.i.acq.rel.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xchg_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xchg_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.global.i.acq.rel.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xchg_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xchg_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.shared.i.acq.rel.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xchg_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xchg_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.gen.i.acquire.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xchg_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xchg_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.global.i.acquire.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xchg_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xchg_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.shared.i.acquire.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xchg_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xchg_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.gen.i.release.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_xchg_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_xchg_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.global.i.release.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_xchg_global_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_xchg_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.shared.i.release.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_xchg_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_xchg_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.gen.i.acq.rel.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xchg_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xchg_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.global.i.acq.rel.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xchg_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xchg_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.exch.shared.i.acq.rel.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xchg_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xchg_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.gen.i.acquire.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_xchg_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_xchg_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.global.i.acquire.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_xchg_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_xchg_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.shared.i.acquire.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_xchg_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_xchg_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.gen.i.release.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_xchg_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_xchg_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.global.i.release.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_xchg_global_l' needs target feature sm_70}}
  __nvvm_atom_release_xchg_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.shared.i.release.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_xchg_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_xchg_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.gen.i.acq.rel.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xchg_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xchg_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.global.i.acq.rel.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xchg_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xchg_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.shared.i.acq.rel.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xchg_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xchg_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.gen.i.acquire.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xchg_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xchg_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.global.i.acquire.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xchg_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xchg_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.shared.i.acquire.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xchg_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xchg_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.gen.i.release.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_xchg_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_xchg_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.global.i.release.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_xchg_global_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_xchg_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.shared.i.release.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_xchg_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_xchg_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.gen.i.acq.rel.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xchg_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xchg_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.global.i.acq.rel.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xchg_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xchg_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.shared.i.acq.rel.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xchg_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xchg_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.gen.i.acquire.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xchg_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xchg_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.global.i.acquire.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xchg_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xchg_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.shared.i.acquire.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xchg_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xchg_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.gen.i.release.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_xchg_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_xchg_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.global.i.release.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_xchg_global_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_xchg_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.shared.i.release.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_xchg_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_xchg_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.gen.i.acq.rel.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xchg_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xchg_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.global.i.acq.rel.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xchg_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xchg_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.exch.shared.i.acq.rel.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xchg_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xchg_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.gen.f.acquire.f32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_xchg_gen_f' needs target feature sm_70}}
  __nvvm_atom_acquire_xchg_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.global.f.acquire.f32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_xchg_global_f' needs target feature sm_70}}
  __nvvm_atom_acquire_xchg_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.shared.f.acquire.f32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_xchg_shared_f' needs target feature sm_70}}
  __nvvm_atom_acquire_xchg_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.gen.f.release.f32.p0
  // expected-error@+1 {{'__nvvm_atom_release_xchg_gen_f' needs target feature sm_70}}
  __nvvm_atom_release_xchg_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.global.f.release.f32.p1
  // expected-error@+1 {{'__nvvm_atom_release_xchg_global_f' needs target feature sm_70}}
  __nvvm_atom_release_xchg_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.shared.f.release.f32.p3
  // expected-error@+1 {{'__nvvm_atom_release_xchg_shared_f' needs target feature sm_70}}
  __nvvm_atom_release_xchg_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.gen.f.acq.rel.f32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xchg_gen_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xchg_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.global.f.acq.rel.f32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xchg_global_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xchg_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.shared.f.acq.rel.f32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xchg_shared_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xchg_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.gen.f.acquire.sys.f32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xchg_gen_f' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xchg_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.global.f.acquire.sys.f32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xchg_global_f' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xchg_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.shared.f.acquire.sys.f32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xchg_shared_f' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xchg_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.gen.f.release.sys.f32.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_xchg_gen_f' needs target feature sm_70}}
  __nvvm_atom_release_sys_xchg_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.global.f.release.sys.f32.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_xchg_global_f' needs target feature sm_70}}
  __nvvm_atom_release_sys_xchg_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.shared.f.release.sys.f32.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_xchg_shared_f' needs target feature sm_70}}
  __nvvm_atom_release_sys_xchg_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.gen.f.acq.rel.sys.f32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xchg_gen_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xchg_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.global.f.acq.rel.sys.f32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xchg_global_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xchg_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.shared.f.acq.rel.sys.f32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xchg_shared_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xchg_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.gen.f.acquire.cta.f32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xchg_gen_f' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xchg_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.global.f.acquire.cta.f32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xchg_global_f' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xchg_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.shared.f.acquire.cta.f32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xchg_shared_f' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xchg_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.gen.f.release.cta.f32.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_xchg_gen_f' needs target feature sm_70}}
  __nvvm_atom_release_cta_xchg_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.global.f.release.cta.f32.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_xchg_global_f' needs target feature sm_70}}
  __nvvm_atom_release_cta_xchg_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.shared.f.release.cta.f32.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_xchg_shared_f' needs target feature sm_70}}
  __nvvm_atom_release_cta_xchg_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.gen.f.acq.rel.cta.f32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xchg_gen_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xchg_gen_f(fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.global.f.acq.rel.cta.f32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xchg_global_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xchg_global_f((__attribute__((address_space(1))) float *)fp, f);

  // CHECK_SM70_LP64: call float @llvm.nvvm.atomic.exch.shared.f.acq.rel.cta.f32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xchg_shared_f' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xchg_shared_f((__attribute__((address_space(3))) float *)fp, f);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.gen.f.acquire.f64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_xchg_gen_d' needs target feature sm_70}}
  __nvvm_atom_acquire_xchg_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.global.f.acquire.f64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_xchg_global_d' needs target feature sm_70}}
  __nvvm_atom_acquire_xchg_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.shared.f.acquire.f64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_xchg_shared_d' needs target feature sm_70}}
  __nvvm_atom_acquire_xchg_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.gen.f.release.f64.p0
  // expected-error@+1 {{'__nvvm_atom_release_xchg_gen_d' needs target feature sm_70}}
  __nvvm_atom_release_xchg_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.global.f.release.f64.p1
  // expected-error@+1 {{'__nvvm_atom_release_xchg_global_d' needs target feature sm_70}}
  __nvvm_atom_release_xchg_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.shared.f.release.f64.p3
  // expected-error@+1 {{'__nvvm_atom_release_xchg_shared_d' needs target feature sm_70}}
  __nvvm_atom_release_xchg_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.gen.f.acq.rel.f64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xchg_gen_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xchg_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.global.f.acq.rel.f64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xchg_global_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xchg_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.shared.f.acq.rel.f64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xchg_shared_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xchg_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.gen.f.acquire.sys.f64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xchg_gen_d' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xchg_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.global.f.acquire.sys.f64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xchg_global_d' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xchg_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.shared.f.acquire.sys.f64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xchg_shared_d' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xchg_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.gen.f.release.sys.f64.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_xchg_gen_d' needs target feature sm_70}}
  __nvvm_atom_release_sys_xchg_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.global.f.release.sys.f64.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_xchg_global_d' needs target feature sm_70}}
  __nvvm_atom_release_sys_xchg_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.shared.f.release.sys.f64.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_xchg_shared_d' needs target feature sm_70}}
  __nvvm_atom_release_sys_xchg_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.gen.f.acq.rel.sys.f64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xchg_gen_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xchg_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.global.f.acq.rel.sys.f64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xchg_global_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xchg_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.shared.f.acq.rel.sys.f64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xchg_shared_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xchg_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.gen.f.acquire.cta.f64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xchg_gen_d' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xchg_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.global.f.acquire.cta.f64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xchg_global_d' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xchg_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.shared.f.acquire.cta.f64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xchg_shared_d' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xchg_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.gen.f.release.cta.f64.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_xchg_gen_d' needs target feature sm_70}}
  __nvvm_atom_release_cta_xchg_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.global.f.release.cta.f64.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_xchg_global_d' needs target feature sm_70}}
  __nvvm_atom_release_cta_xchg_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.shared.f.release.cta.f64.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_xchg_shared_d' needs target feature sm_70}}
  __nvvm_atom_release_cta_xchg_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.gen.f.acq.rel.cta.f64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xchg_gen_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xchg_gen_d(dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.global.f.acq.rel.cta.f64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xchg_global_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xchg_global_d((__attribute__((address_space(1))) double *)dfp, df);

  // CHECK_SM70_LP64: call double @llvm.nvvm.atomic.exch.shared.f.acq.rel.cta.f64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xchg_shared_d' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xchg_shared_d((__attribute__((address_space(3))) double *)dfp, df);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.i.acquire.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_max_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_max_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.i.acquire.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_max_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_max_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.i.acquire.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_max_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_max_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.i.release.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_max_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_max_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.i.release.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_max_global_i' needs target feature sm_70}}
  __nvvm_atom_release_max_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.i.release.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_max_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_max_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.i.acq.rel.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_max_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_max_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.i.acq.rel.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_max_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_max_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.i.acq.rel.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_max_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_max_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.i.acquire.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_max_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_max_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.i.acquire.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_max_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_max_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.i.acquire.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_max_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_max_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.i.release.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_max_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_max_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.i.release.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_max_global_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_max_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.i.release.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_max_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_max_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.i.acq.rel.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_max_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_max_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.i.acq.rel.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_max_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_max_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.i.acq.rel.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_max_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_max_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.i.acquire.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_max_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_max_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.i.acquire.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_max_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_max_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.i.acquire.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_max_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_max_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.i.release.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_max_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_max_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.i.release.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_max_global_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_max_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.i.release.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_max_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_max_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.i.acq.rel.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_max_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_max_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.i.acq.rel.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_max_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_max_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.i.acq.rel.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_max_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_max_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.i.acquire.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_max_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_max_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.i.acquire.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_max_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_max_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.i.acquire.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_max_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_max_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.i.release.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_max_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_max_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.i.release.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_max_global_l' needs target feature sm_70}}
  __nvvm_atom_release_max_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.i.release.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_max_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_max_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.i.acq.rel.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_max_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_max_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.i.acq.rel.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_max_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_max_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.i.acq.rel.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_max_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_max_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.i.acquire.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_max_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_max_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.i.acquire.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_max_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_max_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.i.acquire.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_max_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_max_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.i.release.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_max_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_max_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.i.release.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_max_global_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_max_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.i.release.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_max_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_max_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.i.acq.rel.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_max_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_max_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.i.acq.rel.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_max_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_max_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.i.acq.rel.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_max_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_max_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.i.acquire.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_max_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_max_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.i.acquire.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_max_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_max_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.i.acquire.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_max_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_max_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.i.release.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_max_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_max_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.i.release.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_max_global_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_max_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.i.release.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_max_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_max_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.i.acq.rel.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_max_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_max_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.i.acq.rel.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_max_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_max_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.i.acq.rel.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_max_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_max_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.ui.acquire.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_max_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_max_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.ui.acquire.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_max_global_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_max_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.ui.acquire.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_max_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_max_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.ui.release.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_max_gen_ui' needs target feature sm_70}}
  __nvvm_atom_release_max_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.ui.release.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_max_global_ui' needs target feature sm_70}}
  __nvvm_atom_release_max_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.ui.release.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_max_shared_ui' needs target feature sm_70}}
  __nvvm_atom_release_max_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.ui.acq.rel.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_max_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_max_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.ui.acq.rel.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_max_global_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_max_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.ui.acq.rel.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_max_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_max_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.ui.acquire.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_max_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_max_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.ui.acquire.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_max_global_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_max_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.ui.acquire.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_max_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_max_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.ui.release.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_max_gen_ui' needs target feature sm_70}}
  __nvvm_atom_release_sys_max_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.ui.release.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_max_global_ui' needs target feature sm_70}}
  __nvvm_atom_release_sys_max_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.ui.release.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_max_shared_ui' needs target feature sm_70}}
  __nvvm_atom_release_sys_max_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.ui.acq.rel.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_max_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_max_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.ui.acq.rel.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_max_global_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_max_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.ui.acq.rel.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_max_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_max_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.ui.acquire.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_max_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_max_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.ui.acquire.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_max_global_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_max_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.ui.acquire.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_max_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_max_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.ui.release.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_max_gen_ui' needs target feature sm_70}}
  __nvvm_atom_release_cta_max_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.ui.release.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_max_global_ui' needs target feature sm_70}}
  __nvvm_atom_release_cta_max_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.ui.release.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_max_shared_ui' needs target feature sm_70}}
  __nvvm_atom_release_cta_max_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.gen.ui.acq.rel.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_max_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_max_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.global.ui.acq.rel.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_max_global_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_max_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.max.shared.ui.acq.rel.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_max_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_max_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.ui.acquire.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_max_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_max_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.ui.acquire.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_max_global_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_max_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.ui.acquire.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_max_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_max_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.ui.release.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_max_gen_ul' needs target feature sm_70}}
  __nvvm_atom_release_max_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.ui.release.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_max_global_ul' needs target feature sm_70}}
  __nvvm_atom_release_max_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.ui.release.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_max_shared_ul' needs target feature sm_70}}
  __nvvm_atom_release_max_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.ui.acq.rel.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_max_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_max_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.ui.acq.rel.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_max_global_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_max_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.ui.acq.rel.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_max_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_max_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.ui.acquire.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_max_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_max_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.ui.acquire.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_max_global_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_max_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.ui.acquire.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_max_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_max_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.ui.release.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_max_gen_ul' needs target feature sm_70}}
  __nvvm_atom_release_sys_max_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.ui.release.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_max_global_ul' needs target feature sm_70}}
  __nvvm_atom_release_sys_max_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.ui.release.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_max_shared_ul' needs target feature sm_70}}
  __nvvm_atom_release_sys_max_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.ui.acq.rel.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_max_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_max_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.ui.acq.rel.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_max_global_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_max_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.ui.acq.rel.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_max_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_max_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.ui.acquire.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_max_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_max_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.ui.acquire.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_max_global_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_max_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.ui.acquire.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_max_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_max_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.ui.release.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_max_gen_ul' needs target feature sm_70}}
  __nvvm_atom_release_cta_max_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.ui.release.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_max_global_ul' needs target feature sm_70}}
  __nvvm_atom_release_cta_max_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.ui.release.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_max_shared_ul' needs target feature sm_70}}
  __nvvm_atom_release_cta_max_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.gen.ui.acq.rel.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_max_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_max_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.global.ui.acq.rel.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_max_global_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_max_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.max.shared.ui.acq.rel.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_max_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_max_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.i.acquire.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_min_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_min_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.i.acquire.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_min_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_min_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.i.acquire.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_min_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_min_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.i.release.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_min_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_min_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.i.release.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_min_global_i' needs target feature sm_70}}
  __nvvm_atom_release_min_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.i.release.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_min_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_min_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.i.acq.rel.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_min_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_min_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.i.acq.rel.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_min_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_min_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.i.acq.rel.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_min_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_min_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.i.acquire.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_min_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_min_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.i.acquire.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_min_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_min_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.i.acquire.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_min_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_min_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.i.release.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_min_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_min_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.i.release.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_min_global_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_min_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.i.release.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_min_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_min_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.i.acq.rel.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_min_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_min_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.i.acq.rel.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_min_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_min_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.i.acq.rel.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_min_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_min_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.i.acquire.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_min_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_min_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.i.acquire.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_min_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_min_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.i.acquire.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_min_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_min_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.i.release.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_min_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_min_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.i.release.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_min_global_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_min_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.i.release.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_min_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_min_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.i.acq.rel.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_min_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_min_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.i.acq.rel.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_min_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_min_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.i.acq.rel.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_min_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_min_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.i.acquire.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_min_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_min_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.i.acquire.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_min_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_min_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.i.acquire.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_min_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_min_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.i.release.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_min_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_min_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.i.release.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_min_global_l' needs target feature sm_70}}
  __nvvm_atom_release_min_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.i.release.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_min_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_min_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.i.acq.rel.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_min_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_min_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.i.acq.rel.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_min_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_min_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.i.acq.rel.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_min_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_min_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.i.acquire.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_min_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_min_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.i.acquire.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_min_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_min_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.i.acquire.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_min_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_min_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.i.release.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_min_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_min_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.i.release.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_min_global_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_min_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.i.release.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_min_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_min_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.i.acq.rel.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_min_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_min_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.i.acq.rel.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_min_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_min_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.i.acq.rel.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_min_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_min_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.i.acquire.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_min_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_min_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.i.acquire.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_min_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_min_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.i.acquire.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_min_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_min_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.i.release.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_min_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_min_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.i.release.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_min_global_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_min_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.i.release.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_min_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_min_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.i.acq.rel.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_min_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_min_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.i.acq.rel.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_min_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_min_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.i.acq.rel.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_min_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_min_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.ui.acquire.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_min_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_min_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.ui.acquire.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_min_global_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_min_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.ui.acquire.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_min_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_min_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.ui.release.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_min_gen_ui' needs target feature sm_70}}
  __nvvm_atom_release_min_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.ui.release.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_min_global_ui' needs target feature sm_70}}
  __nvvm_atom_release_min_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.ui.release.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_min_shared_ui' needs target feature sm_70}}
  __nvvm_atom_release_min_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.ui.acq.rel.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_min_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_min_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.ui.acq.rel.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_min_global_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_min_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.ui.acq.rel.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_min_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_min_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.ui.acquire.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_min_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_min_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.ui.acquire.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_min_global_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_min_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.ui.acquire.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_min_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_min_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.ui.release.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_min_gen_ui' needs target feature sm_70}}
  __nvvm_atom_release_sys_min_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.ui.release.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_min_global_ui' needs target feature sm_70}}
  __nvvm_atom_release_sys_min_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.ui.release.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_min_shared_ui' needs target feature sm_70}}
  __nvvm_atom_release_sys_min_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.ui.acq.rel.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_min_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_min_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.ui.acq.rel.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_min_global_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_min_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.ui.acq.rel.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_min_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_min_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.ui.acquire.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_min_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_min_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.ui.acquire.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_min_global_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_min_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.ui.acquire.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_min_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_min_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.ui.release.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_min_gen_ui' needs target feature sm_70}}
  __nvvm_atom_release_cta_min_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.ui.release.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_min_global_ui' needs target feature sm_70}}
  __nvvm_atom_release_cta_min_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.ui.release.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_min_shared_ui' needs target feature sm_70}}
  __nvvm_atom_release_cta_min_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.gen.ui.acq.rel.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_min_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_min_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.global.ui.acq.rel.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_min_global_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_min_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.min.shared.ui.acq.rel.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_min_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_min_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.ui.acquire.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_min_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_min_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.ui.acquire.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_min_global_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_min_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.ui.acquire.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_min_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_min_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.ui.release.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_min_gen_ul' needs target feature sm_70}}
  __nvvm_atom_release_min_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.ui.release.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_min_global_ul' needs target feature sm_70}}
  __nvvm_atom_release_min_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.ui.release.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_min_shared_ul' needs target feature sm_70}}
  __nvvm_atom_release_min_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.ui.acq.rel.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_min_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_min_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.ui.acq.rel.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_min_global_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_min_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.ui.acq.rel.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_min_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_min_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.ui.acquire.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_min_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_min_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.ui.acquire.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_min_global_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_min_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.ui.acquire.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_min_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_min_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.ui.release.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_min_gen_ul' needs target feature sm_70}}
  __nvvm_atom_release_sys_min_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.ui.release.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_min_global_ul' needs target feature sm_70}}
  __nvvm_atom_release_sys_min_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.ui.release.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_min_shared_ul' needs target feature sm_70}}
  __nvvm_atom_release_sys_min_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.ui.acq.rel.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_min_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_min_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.ui.acq.rel.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_min_global_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_min_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.ui.acq.rel.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_min_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_min_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.ui.acquire.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_min_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_min_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.ui.acquire.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_min_global_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_min_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.ui.acquire.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_min_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_min_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.ui.release.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_min_gen_ul' needs target feature sm_70}}
  __nvvm_atom_release_cta_min_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.ui.release.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_min_global_ul' needs target feature sm_70}}
  __nvvm_atom_release_cta_min_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.ui.release.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_min_shared_ul' needs target feature sm_70}}
  __nvvm_atom_release_cta_min_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.gen.ui.acq.rel.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_min_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_min_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.global.ui.acq.rel.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_min_global_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_min_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.min.shared.ui.acq.rel.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_min_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_min_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.gen.i.acquire.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_inc_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_inc_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.global.i.acquire.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_inc_global_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_inc_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.shared.i.acquire.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_inc_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_inc_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.gen.i.release.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_inc_gen_ui' needs target feature sm_70}}
  __nvvm_atom_release_inc_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.global.i.release.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_inc_global_ui' needs target feature sm_70}}
  __nvvm_atom_release_inc_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.shared.i.release.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_inc_shared_ui' needs target feature sm_70}}
  __nvvm_atom_release_inc_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.gen.i.acq.rel.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_inc_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_inc_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.global.i.acq.rel.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_inc_global_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_inc_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.shared.i.acq.rel.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_inc_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_inc_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.gen.i.acquire.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_inc_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_inc_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.global.i.acquire.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_inc_global_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_inc_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.shared.i.acquire.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_inc_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_inc_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.gen.i.release.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_inc_gen_ui' needs target feature sm_70}}
  __nvvm_atom_release_sys_inc_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.global.i.release.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_inc_global_ui' needs target feature sm_70}}
  __nvvm_atom_release_sys_inc_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.shared.i.release.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_inc_shared_ui' needs target feature sm_70}}
  __nvvm_atom_release_sys_inc_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.gen.i.acq.rel.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_inc_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_inc_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.global.i.acq.rel.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_inc_global_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_inc_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.shared.i.acq.rel.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_inc_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_inc_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.gen.i.acquire.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_inc_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_inc_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.global.i.acquire.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_inc_global_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_inc_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.shared.i.acquire.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_inc_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_inc_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.gen.i.release.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_inc_gen_ui' needs target feature sm_70}}
  __nvvm_atom_release_cta_inc_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.global.i.release.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_inc_global_ui' needs target feature sm_70}}
  __nvvm_atom_release_cta_inc_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.shared.i.release.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_inc_shared_ui' needs target feature sm_70}}
  __nvvm_atom_release_cta_inc_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.gen.i.acq.rel.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_inc_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_inc_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.global.i.acq.rel.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_inc_global_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_inc_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.inc.shared.i.acq.rel.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_inc_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_inc_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.gen.i.acquire.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_inc_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_inc_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.global.i.acquire.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_inc_global_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_inc_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.shared.i.acquire.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_inc_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_inc_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.gen.i.release.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_inc_gen_ul' needs target feature sm_70}}
  __nvvm_atom_release_inc_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.global.i.release.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_inc_global_ul' needs target feature sm_70}}
  __nvvm_atom_release_inc_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.shared.i.release.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_inc_shared_ul' needs target feature sm_70}}
  __nvvm_atom_release_inc_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.gen.i.acq.rel.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_inc_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_inc_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.global.i.acq.rel.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_inc_global_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_inc_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.shared.i.acq.rel.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_inc_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_inc_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.gen.i.acquire.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_inc_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_inc_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.global.i.acquire.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_inc_global_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_inc_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.shared.i.acquire.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_inc_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_inc_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.gen.i.release.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_inc_gen_ul' needs target feature sm_70}}
  __nvvm_atom_release_sys_inc_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.global.i.release.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_inc_global_ul' needs target feature sm_70}}
  __nvvm_atom_release_sys_inc_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.shared.i.release.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_inc_shared_ul' needs target feature sm_70}}
  __nvvm_atom_release_sys_inc_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.gen.i.acq.rel.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_inc_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_inc_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.global.i.acq.rel.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_inc_global_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_inc_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.shared.i.acq.rel.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_inc_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_inc_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.gen.i.acquire.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_inc_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_inc_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.global.i.acquire.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_inc_global_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_inc_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.shared.i.acquire.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_inc_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_inc_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.gen.i.release.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_inc_gen_ul' needs target feature sm_70}}
  __nvvm_atom_release_cta_inc_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.global.i.release.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_inc_global_ul' needs target feature sm_70}}
  __nvvm_atom_release_cta_inc_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.shared.i.release.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_inc_shared_ul' needs target feature sm_70}}
  __nvvm_atom_release_cta_inc_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.gen.i.acq.rel.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_inc_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_inc_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.global.i.acq.rel.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_inc_global_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_inc_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.inc.shared.i.acq.rel.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_inc_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_inc_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.gen.i.acquire.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_dec_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_dec_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.global.i.acquire.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_dec_global_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_dec_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.shared.i.acquire.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_dec_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_dec_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.gen.i.release.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_dec_gen_ui' needs target feature sm_70}}
  __nvvm_atom_release_dec_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.global.i.release.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_dec_global_ui' needs target feature sm_70}}
  __nvvm_atom_release_dec_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.shared.i.release.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_dec_shared_ui' needs target feature sm_70}}
  __nvvm_atom_release_dec_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.gen.i.acq.rel.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_dec_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_dec_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.global.i.acq.rel.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_dec_global_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_dec_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.shared.i.acq.rel.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_dec_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_dec_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.gen.i.acquire.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_dec_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_dec_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.global.i.acquire.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_dec_global_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_dec_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.shared.i.acquire.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_dec_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_dec_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.gen.i.release.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_dec_gen_ui' needs target feature sm_70}}
  __nvvm_atom_release_sys_dec_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.global.i.release.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_dec_global_ui' needs target feature sm_70}}
  __nvvm_atom_release_sys_dec_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.shared.i.release.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_dec_shared_ui' needs target feature sm_70}}
  __nvvm_atom_release_sys_dec_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.gen.i.acq.rel.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_dec_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_dec_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.global.i.acq.rel.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_dec_global_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_dec_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.shared.i.acq.rel.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_dec_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_dec_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.gen.i.acquire.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_dec_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_dec_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.global.i.acquire.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_dec_global_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_dec_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.shared.i.acquire.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_dec_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_dec_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.gen.i.release.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_dec_gen_ui' needs target feature sm_70}}
  __nvvm_atom_release_cta_dec_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.global.i.release.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_dec_global_ui' needs target feature sm_70}}
  __nvvm_atom_release_cta_dec_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.shared.i.release.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_dec_shared_ui' needs target feature sm_70}}
  __nvvm_atom_release_cta_dec_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.gen.i.acq.rel.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_dec_gen_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_dec_gen_ui((unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.global.i.acq.rel.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_dec_global_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_dec_global_ui((__attribute__((address_space(1))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.dec.shared.i.acq.rel.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_dec_shared_ui' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_dec_shared_ui((__attribute__((address_space(3))) unsigned int *)(unsigned int *)ip, i);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.gen.i.acquire.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_dec_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_dec_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.global.i.acquire.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_dec_global_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_dec_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.shared.i.acquire.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_dec_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_dec_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.gen.i.release.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_dec_gen_ul' needs target feature sm_70}}
  __nvvm_atom_release_dec_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.global.i.release.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_dec_global_ul' needs target feature sm_70}}
  __nvvm_atom_release_dec_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.shared.i.release.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_dec_shared_ul' needs target feature sm_70}}
  __nvvm_atom_release_dec_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.gen.i.acq.rel.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_dec_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_dec_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.global.i.acq.rel.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_dec_global_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_dec_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.shared.i.acq.rel.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_dec_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_dec_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.gen.i.acquire.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_dec_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_dec_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.global.i.acquire.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_dec_global_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_dec_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.shared.i.acquire.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_dec_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_dec_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.gen.i.release.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_dec_gen_ul' needs target feature sm_70}}
  __nvvm_atom_release_sys_dec_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.global.i.release.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_dec_global_ul' needs target feature sm_70}}
  __nvvm_atom_release_sys_dec_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.shared.i.release.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_dec_shared_ul' needs target feature sm_70}}
  __nvvm_atom_release_sys_dec_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.gen.i.acq.rel.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_dec_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_dec_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.global.i.acq.rel.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_dec_global_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_dec_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.shared.i.acq.rel.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_dec_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_dec_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.gen.i.acquire.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_dec_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_dec_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.global.i.acquire.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_dec_global_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_dec_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.shared.i.acquire.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_dec_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_dec_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.gen.i.release.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_dec_gen_ul' needs target feature sm_70}}
  __nvvm_atom_release_cta_dec_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.global.i.release.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_dec_global_ul' needs target feature sm_70}}
  __nvvm_atom_release_cta_dec_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.shared.i.release.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_dec_shared_ul' needs target feature sm_70}}
  __nvvm_atom_release_cta_dec_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.gen.i.acq.rel.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_dec_gen_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_dec_gen_ul((unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.global.i.acq.rel.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_dec_global_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_dec_global_ul((__attribute__((address_space(1))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.dec.shared.i.acq.rel.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_dec_shared_ul' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_dec_shared_ul((__attribute__((address_space(3))) unsigned long *)(unsigned long *)lp, l);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.gen.i.acquire.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_and_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_and_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.global.i.acquire.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_and_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_and_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.shared.i.acquire.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_and_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_and_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.gen.i.release.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_and_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_and_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.global.i.release.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_and_global_i' needs target feature sm_70}}
  __nvvm_atom_release_and_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.shared.i.release.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_and_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_and_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.gen.i.acq.rel.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_and_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_and_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.global.i.acq.rel.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_and_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_and_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.shared.i.acq.rel.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_and_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_and_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.gen.i.acquire.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_and_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_and_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.global.i.acquire.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_and_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_and_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.shared.i.acquire.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_and_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_and_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.gen.i.release.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_and_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_and_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.global.i.release.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_and_global_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_and_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.shared.i.release.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_and_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_and_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.gen.i.acq.rel.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_and_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_and_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.global.i.acq.rel.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_and_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_and_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.shared.i.acq.rel.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_and_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_and_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.gen.i.acquire.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_and_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_and_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.global.i.acquire.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_and_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_and_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.shared.i.acquire.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_and_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_and_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.gen.i.release.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_and_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_and_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.global.i.release.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_and_global_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_and_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.shared.i.release.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_and_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_and_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.gen.i.acq.rel.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_and_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_and_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.global.i.acq.rel.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_and_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_and_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.and.shared.i.acq.rel.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_and_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_and_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.gen.i.acquire.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_and_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_and_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.global.i.acquire.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_and_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_and_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.shared.i.acquire.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_and_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_and_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.gen.i.release.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_and_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_and_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.global.i.release.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_and_global_l' needs target feature sm_70}}
  __nvvm_atom_release_and_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.shared.i.release.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_and_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_and_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.gen.i.acq.rel.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_and_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_and_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.global.i.acq.rel.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_and_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_and_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.shared.i.acq.rel.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_and_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_and_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.gen.i.acquire.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_and_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_and_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.global.i.acquire.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_and_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_and_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.shared.i.acquire.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_and_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_and_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.gen.i.release.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_and_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_and_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.global.i.release.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_and_global_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_and_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.shared.i.release.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_and_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_and_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.gen.i.acq.rel.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_and_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_and_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.global.i.acq.rel.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_and_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_and_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.shared.i.acq.rel.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_and_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_and_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.gen.i.acquire.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_and_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_and_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.global.i.acquire.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_and_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_and_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.shared.i.acquire.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_and_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_and_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.gen.i.release.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_and_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_and_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.global.i.release.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_and_global_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_and_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.shared.i.release.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_and_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_and_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.gen.i.acq.rel.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_and_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_and_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.global.i.acq.rel.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_and_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_and_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.and.shared.i.acq.rel.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_and_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_and_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.gen.i.acquire.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_or_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_or_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.global.i.acquire.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_or_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_or_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.shared.i.acquire.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_or_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_or_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.gen.i.release.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_or_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_or_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.global.i.release.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_or_global_i' needs target feature sm_70}}
  __nvvm_atom_release_or_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.shared.i.release.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_or_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_or_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.gen.i.acq.rel.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_or_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_or_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.global.i.acq.rel.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_or_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_or_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.shared.i.acq.rel.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_or_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_or_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.gen.i.acquire.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_or_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_or_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.global.i.acquire.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_or_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_or_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.shared.i.acquire.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_or_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_or_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.gen.i.release.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_or_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_or_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.global.i.release.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_or_global_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_or_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.shared.i.release.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_or_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_or_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.gen.i.acq.rel.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_or_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_or_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.global.i.acq.rel.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_or_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_or_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.shared.i.acq.rel.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_or_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_or_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.gen.i.acquire.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_or_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_or_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.global.i.acquire.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_or_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_or_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.shared.i.acquire.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_or_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_or_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.gen.i.release.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_or_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_or_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.global.i.release.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_or_global_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_or_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.shared.i.release.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_or_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_or_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.gen.i.acq.rel.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_or_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_or_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.global.i.acq.rel.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_or_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_or_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.or.shared.i.acq.rel.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_or_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_or_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.gen.i.acquire.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_or_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_or_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.global.i.acquire.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_or_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_or_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.shared.i.acquire.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_or_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_or_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.gen.i.release.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_or_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_or_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.global.i.release.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_or_global_l' needs target feature sm_70}}
  __nvvm_atom_release_or_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.shared.i.release.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_or_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_or_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.gen.i.acq.rel.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_or_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_or_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.global.i.acq.rel.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_or_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_or_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.shared.i.acq.rel.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_or_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_or_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.gen.i.acquire.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_or_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_or_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.global.i.acquire.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_or_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_or_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.shared.i.acquire.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_or_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_or_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.gen.i.release.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_or_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_or_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.global.i.release.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_or_global_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_or_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.shared.i.release.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_or_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_or_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.gen.i.acq.rel.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_or_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_or_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.global.i.acq.rel.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_or_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_or_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.shared.i.acq.rel.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_or_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_or_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.gen.i.acquire.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_or_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_or_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.global.i.acquire.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_or_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_or_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.shared.i.acquire.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_or_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_or_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.gen.i.release.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_or_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_or_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.global.i.release.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_or_global_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_or_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.shared.i.release.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_or_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_or_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.gen.i.acq.rel.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_or_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_or_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.global.i.acq.rel.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_or_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_or_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.or.shared.i.acq.rel.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_or_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_or_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.gen.i.acquire.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_xor_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_xor_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.global.i.acquire.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_xor_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_xor_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.shared.i.acquire.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_xor_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_xor_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.gen.i.release.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_xor_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_xor_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.global.i.release.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_xor_global_i' needs target feature sm_70}}
  __nvvm_atom_release_xor_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.shared.i.release.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_xor_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_xor_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.gen.i.acq.rel.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xor_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xor_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.global.i.acq.rel.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xor_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xor_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.shared.i.acq.rel.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xor_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xor_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.gen.i.acquire.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xor_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xor_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.global.i.acquire.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xor_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xor_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.shared.i.acquire.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xor_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xor_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.gen.i.release.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_xor_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_xor_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.global.i.release.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_xor_global_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_xor_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.shared.i.release.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_xor_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_xor_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.gen.i.acq.rel.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xor_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xor_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.global.i.acq.rel.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xor_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xor_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.shared.i.acq.rel.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xor_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xor_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.gen.i.acquire.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xor_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xor_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.global.i.acquire.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xor_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xor_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.shared.i.acquire.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xor_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xor_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.gen.i.release.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_xor_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_xor_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.global.i.release.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_xor_global_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_xor_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.shared.i.release.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_xor_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_xor_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.gen.i.acq.rel.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xor_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xor_gen_i(ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.global.i.acq.rel.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xor_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xor_global_i((__attribute__((address_space(1))) int *)ip, i);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.xor.shared.i.acq.rel.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xor_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xor_shared_i((__attribute__((address_space(3))) int *)ip, i);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.gen.i.acquire.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_xor_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_xor_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.global.i.acquire.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_xor_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_xor_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.shared.i.acquire.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_xor_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_xor_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.gen.i.release.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_xor_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_xor_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.global.i.release.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_xor_global_l' needs target feature sm_70}}
  __nvvm_atom_release_xor_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.shared.i.release.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_xor_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_xor_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.gen.i.acq.rel.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xor_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xor_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.global.i.acq.rel.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xor_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xor_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.shared.i.acq.rel.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_xor_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_xor_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.gen.i.acquire.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xor_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xor_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.global.i.acquire.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xor_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xor_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.shared.i.acquire.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_xor_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_xor_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.gen.i.release.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_xor_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_xor_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.global.i.release.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_xor_global_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_xor_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.shared.i.release.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_xor_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_xor_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.gen.i.acq.rel.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xor_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xor_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.global.i.acq.rel.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xor_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xor_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.shared.i.acq.rel.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_xor_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_xor_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.gen.i.acquire.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xor_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xor_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.global.i.acquire.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xor_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xor_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.shared.i.acquire.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_xor_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_xor_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.gen.i.release.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_xor_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_xor_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.global.i.release.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_xor_global_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_xor_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.shared.i.release.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_xor_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_xor_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.gen.i.acq.rel.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xor_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xor_gen_l(&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.global.i.acq.rel.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xor_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xor_global_l((__attribute__((address_space(1))) long *)&dl, l);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.xor.shared.i.acq.rel.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_xor_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_xor_shared_l((__attribute__((address_space(3))) long *)&dl, l);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.gen.i.acquire.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cas_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cas_gen_i(ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.global.i.acquire.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cas_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cas_global_i((__attribute__((address_space(1))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.shared.i.acquire.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cas_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cas_shared_i((__attribute__((address_space(3))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.gen.i.release.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_cas_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_cas_gen_i(ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.global.i.release.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_cas_global_i' needs target feature sm_70}}
  __nvvm_atom_release_cas_global_i((__attribute__((address_space(1))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.shared.i.release.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_cas_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_cas_shared_i((__attribute__((address_space(3))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.gen.i.acq.rel.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cas_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cas_gen_i(ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.global.i.acq.rel.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cas_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cas_global_i((__attribute__((address_space(1))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.shared.i.acq.rel.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cas_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cas_shared_i((__attribute__((address_space(3))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.gen.i.acquire.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_cas_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_cas_gen_i(ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.global.i.acquire.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_cas_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_cas_global_i((__attribute__((address_space(1))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.shared.i.acquire.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_cas_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_cas_shared_i((__attribute__((address_space(3))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.gen.i.release.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_cas_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_cas_gen_i(ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.global.i.release.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_cas_global_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_cas_global_i((__attribute__((address_space(1))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.shared.i.release.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_cas_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_sys_cas_shared_i((__attribute__((address_space(3))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.gen.i.acq.rel.sys.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_cas_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_cas_gen_i(ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.global.i.acq.rel.sys.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_cas_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_cas_global_i((__attribute__((address_space(1))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.shared.i.acq.rel.sys.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_cas_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_cas_shared_i((__attribute__((address_space(3))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.gen.i.acquire.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_cas_gen_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_cas_gen_i(ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.global.i.acquire.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_cas_global_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_cas_global_i((__attribute__((address_space(1))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.shared.i.acquire.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_cas_shared_i' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_cas_shared_i((__attribute__((address_space(3))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.gen.i.release.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_cas_gen_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_cas_gen_i(ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.global.i.release.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_cas_global_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_cas_global_i((__attribute__((address_space(1))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.shared.i.release.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_cas_shared_i' needs target feature sm_70}}
  __nvvm_atom_release_cta_cas_shared_i((__attribute__((address_space(3))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.gen.i.acq.rel.cta.i32.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_cas_gen_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_cas_gen_i(ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.global.i.acq.rel.cta.i32.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_cas_global_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_cas_global_i((__attribute__((address_space(1))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i32 @llvm.nvvm.atomic.cas.shared.i.acq.rel.cta.i32.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_cas_shared_i' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_cas_shared_i((__attribute__((address_space(3))) int *)ip, i, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.gen.i.acquire.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cas_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cas_gen_l(&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.global.i.acquire.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cas_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cas_global_l((__attribute__((address_space(1))) long *)&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.shared.i.acquire.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cas_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cas_shared_l((__attribute__((address_space(3))) long *)&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.gen.i.release.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_cas_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_cas_gen_l(&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.global.i.release.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_cas_global_l' needs target feature sm_70}}
  __nvvm_atom_release_cas_global_l((__attribute__((address_space(1))) long *)&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.shared.i.release.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_cas_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_cas_shared_l((__attribute__((address_space(3))) long *)&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.gen.i.acq.rel.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cas_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cas_gen_l(&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.global.i.acq.rel.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cas_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cas_global_l((__attribute__((address_space(1))) long *)&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.shared.i.acq.rel.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cas_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cas_shared_l((__attribute__((address_space(3))) long *)&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.gen.i.acquire.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_cas_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_cas_gen_l(&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.global.i.acquire.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_cas_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_cas_global_l((__attribute__((address_space(1))) long *)&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.shared.i.acquire.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_sys_cas_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_sys_cas_shared_l((__attribute__((address_space(3))) long *)&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.gen.i.release.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_sys_cas_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_cas_gen_l(&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.global.i.release.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_sys_cas_global_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_cas_global_l((__attribute__((address_space(1))) long *)&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.shared.i.release.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_sys_cas_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_sys_cas_shared_l((__attribute__((address_space(3))) long *)&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.gen.i.acq.rel.sys.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_cas_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_cas_gen_l(&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.global.i.acq.rel.sys.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_cas_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_cas_global_l((__attribute__((address_space(1))) long *)&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.shared.i.acq.rel.sys.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_sys_cas_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_sys_cas_shared_l((__attribute__((address_space(3))) long *)&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.gen.i.acquire.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_cas_gen_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_cas_gen_l(&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.global.i.acquire.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_cas_global_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_cas_global_l((__attribute__((address_space(1))) long *)&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.shared.i.acquire.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acquire_cta_cas_shared_l' needs target feature sm_70}}
  __nvvm_atom_acquire_cta_cas_shared_l((__attribute__((address_space(3))) long *)&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.gen.i.release.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_release_cta_cas_gen_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_cas_gen_l(&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.global.i.release.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_release_cta_cas_global_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_cas_global_l((__attribute__((address_space(1))) long *)&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.shared.i.release.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_release_cta_cas_shared_l' needs target feature sm_70}}
  __nvvm_atom_release_cta_cas_shared_l((__attribute__((address_space(3))) long *)&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.gen.i.acq.rel.cta.i64.p0
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_cas_gen_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_cas_gen_l(&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.global.i.acq.rel.cta.i64.p1
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_cas_global_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_cas_global_l((__attribute__((address_space(1))) long *)&dl, l, 0);

  // CHECK_SM70_LP64: call i64 @llvm.nvvm.atomic.cas.shared.i.acq.rel.cta.i64.p3
  // expected-error@+1 {{'__nvvm_atom_acq_rel_cta_cas_shared_l' needs target feature sm_70}}
  __nvvm_atom_acq_rel_cta_cas_shared_l((__attribute__((address_space(3))) long *)&dl, l, 0);

#endif

  // CHECK: ret
}

// CHECK-LABEL: nvvm_ldg
__device__ void nvvm_ldg(const void *p) {
  // CHECK: call i8 @llvm.nvvm.ldg.global.i.i8.p0(ptr {{%[0-9]+}}, i32 1)
  // CHECK: call i8 @llvm.nvvm.ldg.global.i.i8.p0(ptr {{%[0-9]+}}, i32 1)
  // CHECK: call i8 @llvm.nvvm.ldg.global.i.i8.p0(ptr {{%[0-9]+}}, i32 1)
  __nvvm_ldg_c((const char *)p);
  __nvvm_ldg_uc((const unsigned char *)p);
  __nvvm_ldg_sc((const signed char *)p);

  // CHECK: call i16 @llvm.nvvm.ldg.global.i.i16.p0(ptr {{%[0-9]+}}, i32 2)
  // CHECK: call i16 @llvm.nvvm.ldg.global.i.i16.p0(ptr {{%[0-9]+}}, i32 2)
  __nvvm_ldg_s((const short *)p);
  __nvvm_ldg_us((const unsigned short *)p);

  // CHECK: call i32 @llvm.nvvm.ldg.global.i.i32.p0(ptr {{%[0-9]+}}, i32 4)
  // CHECK: call i32 @llvm.nvvm.ldg.global.i.i32.p0(ptr {{%[0-9]+}}, i32 4)
  __nvvm_ldg_i((const int *)p);
  __nvvm_ldg_ui((const unsigned int *)p);

  // LP32: call i32 @llvm.nvvm.ldg.global.i.i32.p0(ptr {{%[0-9]+}}, i32 4)
  // LP32: call i32 @llvm.nvvm.ldg.global.i.i32.p0(ptr {{%[0-9]+}}, i32 4)
  // LP64: call i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr {{%[0-9]+}}, i32 8)
  // LP64: call i64 @llvm.nvvm.ldg.global.i.i64.p0(ptr {{%[0-9]+}}, i32 8)
  __nvvm_ldg_l((const long *)p);
  __nvvm_ldg_ul((const unsigned long *)p);

  // CHECK: call float @llvm.nvvm.ldg.global.f.f32.p0(ptr {{%[0-9]+}}, i32 4)
  __nvvm_ldg_f((const float *)p);
  // CHECK: call double @llvm.nvvm.ldg.global.f.f64.p0(ptr {{%[0-9]+}}, i32 8)
  __nvvm_ldg_d((const double *)p);

  // In practice, the pointers we pass to __ldg will be aligned as appropriate
  // for the CUDA <type>N vector types (e.g. short4), which are not the same as
  // the LLVM vector types.  However, each LLVM vector type has an alignment
  // less than or equal to its corresponding CUDA type, so we're OK.
  //
  // PTX Interoperability section 2.2: "For a vector with an even number of
  // elements, its alignment is set to number of elements times the alignment of
  // its member: n*alignof(t)."

  // CHECK: call <2 x i8> @llvm.nvvm.ldg.global.i.v2i8.p0(ptr {{%[0-9]+}}, i32 2)
  // CHECK: call <2 x i8> @llvm.nvvm.ldg.global.i.v2i8.p0(ptr {{%[0-9]+}}, i32 2)
  // CHECK: call <2 x i8> @llvm.nvvm.ldg.global.i.v2i8.p0(ptr {{%[0-9]+}}, i32 2)
  typedef char char2 __attribute__((ext_vector_type(2)));
  typedef unsigned char uchar2 __attribute__((ext_vector_type(2)));
  typedef signed char schar2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_c2((const char2 *)p);
  __nvvm_ldg_uc2((const uchar2 *)p);
  __nvvm_ldg_sc2((const schar2 *)p);

  // CHECK: call <4 x i8> @llvm.nvvm.ldg.global.i.v4i8.p0(ptr {{%[0-9]+}}, i32 4)
  // CHECK: call <4 x i8> @llvm.nvvm.ldg.global.i.v4i8.p0(ptr {{%[0-9]+}}, i32 4)
  // CHECK: call <4 x i8> @llvm.nvvm.ldg.global.i.v4i8.p0(ptr {{%[0-9]+}}, i32 4)
  typedef char char4 __attribute__((ext_vector_type(4)));
  typedef unsigned char uchar4 __attribute__((ext_vector_type(4)));
  typedef signed char schar4 __attribute__((ext_vector_type(4)));
  __nvvm_ldg_c4((const char4 *)p);
  __nvvm_ldg_uc4((const uchar4 *)p);
  __nvvm_ldg_sc4((const schar4 *)p);

  // CHECK: call <2 x i16> @llvm.nvvm.ldg.global.i.v2i16.p0(ptr {{%[0-9]+}}, i32 4)
  // CHECK: call <2 x i16> @llvm.nvvm.ldg.global.i.v2i16.p0(ptr {{%[0-9]+}}, i32 4)
  typedef short short2 __attribute__((ext_vector_type(2)));
  typedef unsigned short ushort2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_s2((const short2 *)p);
  __nvvm_ldg_us2((const ushort2 *)p);

  // CHECK: call <4 x i16> @llvm.nvvm.ldg.global.i.v4i16.p0(ptr {{%[0-9]+}}, i32 8)
  // CHECK: call <4 x i16> @llvm.nvvm.ldg.global.i.v4i16.p0(ptr {{%[0-9]+}}, i32 8)
  typedef short short4 __attribute__((ext_vector_type(4)));
  typedef unsigned short ushort4 __attribute__((ext_vector_type(4)));
  __nvvm_ldg_s4((const short4 *)p);
  __nvvm_ldg_us4((const ushort4 *)p);

  // CHECK: call <2 x i32> @llvm.nvvm.ldg.global.i.v2i32.p0(ptr {{%[0-9]+}}, i32 8)
  // CHECK: call <2 x i32> @llvm.nvvm.ldg.global.i.v2i32.p0(ptr {{%[0-9]+}}, i32 8)
  typedef int int2 __attribute__((ext_vector_type(2)));
  typedef unsigned int uint2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_i2((const int2 *)p);
  __nvvm_ldg_ui2((const uint2 *)p);

  // CHECK: call <4 x i32> @llvm.nvvm.ldg.global.i.v4i32.p0(ptr {{%[0-9]+}}, i32 16)
  // CHECK: call <4 x i32> @llvm.nvvm.ldg.global.i.v4i32.p0(ptr {{%[0-9]+}}, i32 16)
  typedef int int4 __attribute__((ext_vector_type(4)));
  typedef unsigned int uint4 __attribute__((ext_vector_type(4)));
  __nvvm_ldg_i4((const int4 *)p);
  __nvvm_ldg_ui4((const uint4 *)p);

  // LP32: call <2 x i32> @llvm.nvvm.ldg.global.i.v2i32.p0(ptr {{%[0-9]+}}, i32 8)
  // LP32: call <2 x i32> @llvm.nvvm.ldg.global.i.v2i32.p0(ptr {{%[0-9]+}}, i32 8)
  // LP64: call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0(ptr {{%[0-9]+}}, i32 16)
  // LP64: call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0(ptr {{%[0-9]+}}, i32 16)
  typedef long long2 __attribute__((ext_vector_type(2)));
  typedef unsigned long ulong2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_l2((const long2 *)p);
  __nvvm_ldg_ul2((const ulong2 *)p);

  // CHECK: call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0(ptr {{%[0-9]+}}, i32 16)
  // CHECK: call <2 x i64> @llvm.nvvm.ldg.global.i.v2i64.p0(ptr {{%[0-9]+}}, i32 16)
  typedef long long longlong2 __attribute__((ext_vector_type(2)));
  typedef unsigned long long ulonglong2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_ll2((const longlong2 *)p);
  __nvvm_ldg_ull2((const ulonglong2 *)p);

  // CHECK: call <2 x float> @llvm.nvvm.ldg.global.f.v2f32.p0(ptr {{%[0-9]+}}, i32 8)
  typedef float float2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_f2((const float2 *)p);

  // CHECK: call <4 x float> @llvm.nvvm.ldg.global.f.v4f32.p0(ptr {{%[0-9]+}}, i32 16)
  typedef float float4 __attribute__((ext_vector_type(4)));
  __nvvm_ldg_f4((const float4 *)p);

  // CHECK: call <2 x double> @llvm.nvvm.ldg.global.f.v2f64.p0(ptr {{%[0-9]+}}, i32 16)
  typedef double double2 __attribute__((ext_vector_type(2)));
  __nvvm_ldg_d2((const double2 *)p);
}

// CHECK-LABEL: nvvm_ldu
__device__ void nvvm_ldu(const void *p) {
  // CHECK: call i8 @llvm.nvvm.ldu.global.i.i8.p0(ptr {{%[0-9]+}}, i32 1)
  // CHECK: call i8 @llvm.nvvm.ldu.global.i.i8.p0(ptr {{%[0-9]+}}, i32 1)
  // CHECK: call i8 @llvm.nvvm.ldu.global.i.i8.p0(ptr {{%[0-9]+}}, i32 1)
  __nvvm_ldu_c((const char *)p);
  __nvvm_ldu_uc((const unsigned char *)p);
  __nvvm_ldu_sc((const signed char *)p);

  // CHECK: call i16 @llvm.nvvm.ldu.global.i.i16.p0(ptr {{%[0-9]+}}, i32 2)
  // CHECK: call i16 @llvm.nvvm.ldu.global.i.i16.p0(ptr {{%[0-9]+}}, i32 2)
  __nvvm_ldu_s((const short *)p);
  __nvvm_ldu_us((const unsigned short *)p);

  // CHECK: call i32 @llvm.nvvm.ldu.global.i.i32.p0(ptr {{%[0-9]+}}, i32 4)
  // CHECK: call i32 @llvm.nvvm.ldu.global.i.i32.p0(ptr {{%[0-9]+}}, i32 4)
  __nvvm_ldu_i((const int *)p);
  __nvvm_ldu_ui((const unsigned int *)p);

  // LP32: call i32 @llvm.nvvm.ldu.global.i.i32.p0(ptr {{%[0-9]+}}, i32 4)
  // LP32: call i32 @llvm.nvvm.ldu.global.i.i32.p0(ptr {{%[0-9]+}}, i32 4)
  // LP64: call i64 @llvm.nvvm.ldu.global.i.i64.p0(ptr {{%[0-9]+}}, i32 8)
  // LP64: call i64 @llvm.nvvm.ldu.global.i.i64.p0(ptr {{%[0-9]+}}, i32 8)
  __nvvm_ldu_l((const long *)p);
  __nvvm_ldu_ul((const unsigned long *)p);

  // CHECK: call float @llvm.nvvm.ldu.global.f.f32.p0(ptr {{%[0-9]+}}, i32 4)
  __nvvm_ldu_f((const float *)p);
  // CHECK: call double @llvm.nvvm.ldu.global.f.f64.p0(ptr {{%[0-9]+}}, i32 8)
  __nvvm_ldu_d((const double *)p);

  // CHECK: call <2 x i8> @llvm.nvvm.ldu.global.i.v2i8.p0(ptr {{%[0-9]+}}, i32 2)
  // CHECK: call <2 x i8> @llvm.nvvm.ldu.global.i.v2i8.p0(ptr {{%[0-9]+}}, i32 2)
  // CHECK: call <2 x i8> @llvm.nvvm.ldu.global.i.v2i8.p0(ptr {{%[0-9]+}}, i32 2)
  typedef char char2 __attribute__((ext_vector_type(2)));
  typedef unsigned char uchar2 __attribute__((ext_vector_type(2)));
  typedef signed char schar2 __attribute__((ext_vector_type(2)));
  __nvvm_ldu_c2((const char2 *)p);
  __nvvm_ldu_uc2((const uchar2 *)p);
  __nvvm_ldu_sc2((const schar2 *)p);

  // CHECK: call <4 x i8> @llvm.nvvm.ldu.global.i.v4i8.p0(ptr {{%[0-9]+}}, i32 4)
  // CHECK: call <4 x i8> @llvm.nvvm.ldu.global.i.v4i8.p0(ptr {{%[0-9]+}}, i32 4)
  // CHECK: call <4 x i8> @llvm.nvvm.ldu.global.i.v4i8.p0(ptr {{%[0-9]+}}, i32 4)
  typedef char char4 __attribute__((ext_vector_type(4)));
  typedef unsigned char uchar4 __attribute__((ext_vector_type(4)));
  typedef signed char schar4 __attribute__((ext_vector_type(4)));
  __nvvm_ldu_c4((const char4 *)p);
  __nvvm_ldu_uc4((const uchar4 *)p);
  __nvvm_ldu_sc4((const schar4 *)p);

  // CHECK: call <2 x i16> @llvm.nvvm.ldu.global.i.v2i16.p0(ptr {{%[0-9]+}}, i32 4)
  // CHECK: call <2 x i16> @llvm.nvvm.ldu.global.i.v2i16.p0(ptr {{%[0-9]+}}, i32 4)
  typedef short short2 __attribute__((ext_vector_type(2)));
  typedef unsigned short ushort2 __attribute__((ext_vector_type(2)));
  __nvvm_ldu_s2((const short2 *)p);
  __nvvm_ldu_us2((const ushort2 *)p);

  // CHECK: call <4 x i16> @llvm.nvvm.ldu.global.i.v4i16.p0(ptr {{%[0-9]+}}, i32 8)
  // CHECK: call <4 x i16> @llvm.nvvm.ldu.global.i.v4i16.p0(ptr {{%[0-9]+}}, i32 8)
  typedef short short4 __attribute__((ext_vector_type(4)));
  typedef unsigned short ushort4 __attribute__((ext_vector_type(4)));
  __nvvm_ldu_s4((const short4 *)p);
  __nvvm_ldu_us4((const ushort4 *)p);

  // CHECK: call <2 x i32> @llvm.nvvm.ldu.global.i.v2i32.p0(ptr {{%[0-9]+}}, i32 8)
  // CHECK: call <2 x i32> @llvm.nvvm.ldu.global.i.v2i32.p0(ptr {{%[0-9]+}}, i32 8)
  typedef int int2 __attribute__((ext_vector_type(2)));
  typedef unsigned int uint2 __attribute__((ext_vector_type(2)));
  __nvvm_ldu_i2((const int2 *)p);
  __nvvm_ldu_ui2((const uint2 *)p);

  // CHECK: call <4 x i32> @llvm.nvvm.ldu.global.i.v4i32.p0(ptr {{%[0-9]+}}, i32 16)
  // CHECK: call <4 x i32> @llvm.nvvm.ldu.global.i.v4i32.p0(ptr {{%[0-9]+}}, i32 16)
  typedef int int4 __attribute__((ext_vector_type(4)));
  typedef unsigned int uint4 __attribute__((ext_vector_type(4)));
  __nvvm_ldu_i4((const int4 *)p);
  __nvvm_ldu_ui4((const uint4 *)p);

  // LP32: call <2 x i32> @llvm.nvvm.ldu.global.i.v2i32.p0(ptr {{%[0-9]+}}, i32 8)
  // LP32: call <2 x i32> @llvm.nvvm.ldu.global.i.v2i32.p0(ptr {{%[0-9]+}}, i32 8)
  // LP64: call <2 x i64> @llvm.nvvm.ldu.global.i.v2i64.p0(ptr {{%[0-9]+}}, i32 16)
  // LP64: call <2 x i64> @llvm.nvvm.ldu.global.i.v2i64.p0(ptr {{%[0-9]+}}, i32 16)
  typedef long long2 __attribute__((ext_vector_type(2)));
  typedef unsigned long ulong2 __attribute__((ext_vector_type(2)));
  __nvvm_ldu_l2((const long2 *)p);
  __nvvm_ldu_ul2((const ulong2 *)p);

  // CHECK: call <2 x i64> @llvm.nvvm.ldu.global.i.v2i64.p0(ptr {{%[0-9]+}}, i32 16)
  // CHECK: call <2 x i64> @llvm.nvvm.ldu.global.i.v2i64.p0(ptr {{%[0-9]+}}, i32 16)
  typedef long long longlong2 __attribute__((ext_vector_type(2)));
  typedef unsigned long long ulonglong2 __attribute__((ext_vector_type(2)));
  __nvvm_ldu_ll2((const longlong2 *)p);
  __nvvm_ldu_ull2((const ulonglong2 *)p);

  // CHECK: call <2 x float> @llvm.nvvm.ldu.global.f.v2f32.p0(ptr {{%[0-9]+}}, i32 8)
  typedef float float2 __attribute__((ext_vector_type(2)));
  __nvvm_ldu_f2((const float2 *)p);

  // CHECK: call <4 x float> @llvm.nvvm.ldu.global.f.v4f32.p0(ptr {{%[0-9]+}}, i32 16)
  typedef float float4 __attribute__((ext_vector_type(4)));
  __nvvm_ldu_f4((const float4 *)p);

  // CHECK: call <2 x double> @llvm.nvvm.ldu.global.f.v2f64.p0(ptr {{%[0-9]+}}, i32 16)
  typedef double double2 __attribute__((ext_vector_type(2)));
  __nvvm_ldu_d2((const double2 *)p);
}

// CHECK-LABEL: nvvm_shfl
__device__ void nvvm_shfl(int i, float f, int a, int b) {
  // CHECK: call i32 @llvm.nvvm.shfl.down.i32(i32
  __nvvm_shfl_down_i32(i, a, b);
  // CHECK: call float @llvm.nvvm.shfl.down.f32(float
  __nvvm_shfl_down_f32(f, a, b);
  // CHECK: call i32 @llvm.nvvm.shfl.up.i32(i32
  __nvvm_shfl_up_i32(i, a, b);
  // CHECK: call float @llvm.nvvm.shfl.up.f32(float
  __nvvm_shfl_up_f32(f, a, b);
  // CHECK: call i32 @llvm.nvvm.shfl.bfly.i32(i32
  __nvvm_shfl_bfly_i32(i, a, b);
  // CHECK: call float @llvm.nvvm.shfl.bfly.f32(float
  __nvvm_shfl_bfly_f32(f, a, b);
  // CHECK: call i32 @llvm.nvvm.shfl.idx.i32(i32
  __nvvm_shfl_idx_i32(i, a, b);
  // CHECK: call float @llvm.nvvm.shfl.idx.f32(float
  __nvvm_shfl_idx_f32(f, a, b);
  // CHECK: ret void
}

__device__ void nvvm_vote(int pred) {
  // CHECK: call i1 @llvm.nvvm.vote.all(i1
  __nvvm_vote_all(pred);
  // CHECK: call i1 @llvm.nvvm.vote.any(i1
  __nvvm_vote_any(pred);
  // CHECK: call i1 @llvm.nvvm.vote.uni(i1
  __nvvm_vote_uni(pred);
  // CHECK: call i32 @llvm.nvvm.vote.ballot(i1
  __nvvm_vote_ballot(pred);
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_mbarrier
__device__ void nvvm_mbarrier(long long* addr, __attribute__((address_space(3))) long long* sharedAddr, int count, long long state) {
  #if __CUDA_ARCH__ >= 800
  __nvvm_mbarrier_init(addr, count);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.mbarrier.init
  __nvvm_mbarrier_init_shared(sharedAddr, count);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.mbarrier.init.shared

  __nvvm_mbarrier_inval(addr);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.mbarrier.inval
  __nvvm_mbarrier_inval_shared(sharedAddr);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.mbarrier.inval.shared

  __nvvm_mbarrier_arrive(addr);
  // CHECK_PTX70_SM80: call i64 @llvm.nvvm.mbarrier.arrive
  __nvvm_mbarrier_arrive_shared(sharedAddr);
  // CHECK_PTX70_SM80: call i64 @llvm.nvvm.mbarrier.arrive.shared
  __nvvm_mbarrier_arrive_noComplete(addr, count);
  // CHECK_PTX70_SM80: call i64 @llvm.nvvm.mbarrier.arrive.noComplete
  __nvvm_mbarrier_arrive_noComplete_shared(sharedAddr, count);
  // CHECK_PTX70_SM80: call i64 @llvm.nvvm.mbarrier.arrive.noComplete.shared

  __nvvm_mbarrier_arrive_drop(addr);
  // CHECK_PTX70_SM80: call i64 @llvm.nvvm.mbarrier.arrive.drop
  __nvvm_mbarrier_arrive_drop_shared(sharedAddr);
  // CHECK_PTX70_SM80: call i64 @llvm.nvvm.mbarrier.arrive.drop.shared
  __nvvm_mbarrier_arrive_drop_noComplete(addr, count);
  // CHECK_PTX70_SM80: call i64 @llvm.nvvm.mbarrier.arrive.drop.noComplete
  __nvvm_mbarrier_arrive_drop_noComplete_shared(sharedAddr, count);
  // CHECK_PTX70_SM80: call i64 @llvm.nvvm.mbarrier.arrive.drop.noComplete.shared

  __nvvm_mbarrier_test_wait(addr, state);
  // CHECK_PTX70_SM80: call i1 @llvm.nvvm.mbarrier.test.wait
  __nvvm_mbarrier_test_wait_shared(sharedAddr, state);
  // CHECK_PTX70_SM80: call i1 @llvm.nvvm.mbarrier.test.wait.shared

  __nvvm_mbarrier_pending_count(state);
  // CHECK_PTX70_SM80: call i32 @llvm.nvvm.mbarrier.pending.count
  #endif
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_async_copy
__device__ void nvvm_async_copy(__attribute__((address_space(3))) void* dst, __attribute__((address_space(1))) const void* src, long long* addr, __attribute__((address_space(3))) long long* sharedAddr) {
  #if __CUDA_ARCH__ >= 800
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.mbarrier.arrive
  __nvvm_cp_async_mbarrier_arrive(addr);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.mbarrier.arrive.shared
  __nvvm_cp_async_mbarrier_arrive_shared(sharedAddr);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.mbarrier.arrive.noinc
  __nvvm_cp_async_mbarrier_arrive_noinc(addr);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.mbarrier.arrive.noinc.shared
  __nvvm_cp_async_mbarrier_arrive_noinc_shared(sharedAddr);

  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.ca.shared.global.4(
  __nvvm_cp_async_ca_shared_global_4(dst, src);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.ca.shared.global.8(
  __nvvm_cp_async_ca_shared_global_8(dst, src);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.ca.shared.global.16(
  __nvvm_cp_async_ca_shared_global_16(dst, src);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.cg.shared.global.16(
  __nvvm_cp_async_cg_shared_global_16(dst, src);

  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.ca.shared.global.4.s({{.*}}, i32 2)
  __nvvm_cp_async_ca_shared_global_4(dst, src, 2);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.ca.shared.global.8.s({{.*}}, i32 2)
  __nvvm_cp_async_ca_shared_global_8(dst, src, 2);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.ca.shared.global.16.s({{.*}}, i32 2)
  __nvvm_cp_async_ca_shared_global_16(dst, src, 2);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.cg.shared.global.16.s({{.*}}, i32 2)
  __nvvm_cp_async_cg_shared_global_16(dst, src, 2);
  
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.commit.group
  __nvvm_cp_async_commit_group();
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.wait.group(i32 0)
  __nvvm_cp_async_wait_group(0);
    // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.wait.group(i32 8)
  __nvvm_cp_async_wait_group(8);
    // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.wait.group(i32 16)
  __nvvm_cp_async_wait_group(16);
  // CHECK_PTX70_SM80: call void @llvm.nvvm.cp.async.wait.all
  __nvvm_cp_async_wait_all();
  #endif
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_cvt_sm80
__device__ void nvvm_cvt_sm80() {
#if __CUDA_ARCH__ >= 800
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rn(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff2bf16x2_rn(1, 1);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rn.relu(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff2bf16x2_rn_relu(1, 1);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rz(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff2bf16x2_rz(1, 1);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rz.relu(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff2bf16x2_rz_relu(1, 1);

  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.ff2f16x2.rn(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff2f16x2_rn(1, 1);
  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.ff2f16x2.rn.relu(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff2f16x2_rn_relu(1, 1);
  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.ff2f16x2.rz(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff2f16x2_rz(1, 1);
  // CHECK_PTX70_SM80: call <2 x half> @llvm.nvvm.ff2f16x2.rz.relu(float 1.000000e+00, float 1.000000e+00)
  __nvvm_ff2f16x2_rz_relu(1, 1);

  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.f2bf16.rn(float 1.000000e+00)
  __nvvm_f2bf16_rn(1);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.f2bf16.rn.relu(float 1.000000e+00)
  __nvvm_f2bf16_rn_relu(1);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.f2bf16.rz(float 1.000000e+00)
  __nvvm_f2bf16_rz(1);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.f2bf16.rz.relu(float 1.000000e+00)
  __nvvm_f2bf16_rz_relu(1);

  // CHECK_PTX70_SM80: call i32 @llvm.nvvm.f2tf32.rna(float 1.000000e+00)
  __nvvm_f2tf32_rna(1);
#endif
  // CHECK: ret void
}

#define NAN32 0x7FBFFFFF
#define NAN16 (__bf16)0x7FBF
#define BF16 (__bf16)0.1f
#define BF16_2 (__bf16)0.2f
#define NANBF16 (__bf16)0xFFC1
#define BF16X2 {(__bf16)0.1f, (__bf16)0.1f}
#define BF16X2_2 {(__bf16)0.2f, (__bf16)0.2f}
#define NANBF16X2 {NANBF16, NANBF16}

// CHECK-LABEL: nvvm_abs_neg_bf16_bf16x2_sm80
__device__ void nvvm_abs_neg_bf16_bf16x2_sm80() {
#if __CUDA_ARCH__ >= 800

  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.abs.bf16(bfloat 0xR3DCD)
  __nvvm_abs_bf16(BF16);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.abs.bf16x2(<2 x bfloat> <bfloat 0xR3DCD, bfloat 0xR3DCD>)
  __nvvm_abs_bf16x2(BF16X2);

  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.neg.bf16(bfloat 0xR3DCD)
  __nvvm_neg_bf16(BF16);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.neg.bf16x2(<2 x bfloat> <bfloat 0xR3DCD, bfloat 0xR3DCD>)
  __nvvm_neg_bf16x2(BF16X2);
#endif
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_min_max_sm80
__device__ void nvvm_min_max_sm80() {
#if __CUDA_ARCH__ >= 800

  // CHECK_PTX70_SM80: call float @llvm.nvvm.fmin.nan.f
  __nvvm_fmin_nan_f(0.1f, (float)NAN32);
  // CHECK_PTX70_SM80: call float @llvm.nvvm.fmin.ftz.nan.f
  __nvvm_fmin_ftz_nan_f(0.1f, (float)NAN32);

  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fmin.bf16
  __nvvm_fmin_bf16(BF16, BF16_2);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fmin.ftz.bf16
  __nvvm_fmin_ftz_bf16(BF16, BF16_2);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fmin.nan.bf16
  __nvvm_fmin_nan_bf16(BF16, NANBF16);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fmin.ftz.nan.bf16
  __nvvm_fmin_ftz_nan_bf16(BF16, NANBF16);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fmin.bf16x2
  __nvvm_fmin_bf16x2(BF16X2, BF16X2_2);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fmin.ftz.bf16x2
  __nvvm_fmin_ftz_bf16x2(BF16X2, BF16X2_2);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fmin.nan.bf16x2
  __nvvm_fmin_nan_bf16x2(BF16X2, NANBF16X2);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fmin.ftz.nan.bf16x2
  __nvvm_fmin_ftz_nan_bf16x2(BF16X2, NANBF16X2);
  // CHECK_PTX70_SM80: call float @llvm.nvvm.fmax.nan.f
  __nvvm_fmax_nan_f(0.1f, 0.11f);
  // CHECK_PTX70_SM80: call float @llvm.nvvm.fmax.ftz.nan.f
  __nvvm_fmax_ftz_nan_f(0.1f, (float)NAN32);

  // CHECK_PTX70_SM80: call float @llvm.nvvm.fmax.nan.f
  __nvvm_fmax_nan_f(0.1f, (float)NAN32);
  // CHECK_PTX70_SM80: call float @llvm.nvvm.fmax.ftz.nan.f
  __nvvm_fmax_ftz_nan_f(0.1f, (float)NAN32);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fmax.bf16
  __nvvm_fmax_bf16(BF16, BF16_2);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fmax.ftz.bf16
  __nvvm_fmax_ftz_bf16(BF16, BF16_2);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fmax.nan.bf16
  __nvvm_fmax_nan_bf16(BF16, NANBF16);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fmax.ftz.nan.bf16
  __nvvm_fmax_ftz_nan_bf16(BF16, NANBF16);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fmax.bf16x2
  __nvvm_fmax_bf16x2(BF16X2, BF16X2_2);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fmax.ftz.bf16x2
  __nvvm_fmax_ftz_bf16x2(BF16X2, BF16X2_2);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fmax.nan.bf16x2
  __nvvm_fmax_nan_bf16x2(NANBF16X2, BF16X2);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fmax.ftz.nan.bf16x2
  __nvvm_fmax_ftz_nan_bf16x2(NANBF16X2, BF16X2);
  // CHECK_PTX70_SM80: call float @llvm.nvvm.fmax.nan.f
  __nvvm_fmax_nan_f(0.1f, (float)NAN32);
  // CHECK_PTX70_SM80: call float @llvm.nvvm.fmax.ftz.nan.f
  __nvvm_fmax_ftz_nan_f(0.1f, (float)NAN32);

#endif
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_fma_bf16_bf16x2_sm80
__device__ void nvvm_fma_bf16_bf16x2_sm80() {
#if __CUDA_ARCH__ >= 800
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fma.rn.bf16
  __nvvm_fma_rn_bf16(BF16, BF16_2, BF16_2);
  // CHECK_PTX70_SM80: call bfloat @llvm.nvvm.fma.rn.relu.bf16
  __nvvm_fma_rn_relu_bf16(BF16, BF16_2, BF16_2);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fma.rn.bf16x2
  __nvvm_fma_rn_bf16x2(BF16X2, BF16X2_2, BF16X2_2);
  // CHECK_PTX70_SM80: call <2 x bfloat> @llvm.nvvm.fma.rn.relu.bf16x2
  __nvvm_fma_rn_relu_bf16x2(BF16X2, BF16X2_2, BF16X2_2);
#endif
  // CHECK: ret void
}

// CHECK-LABEL: nvvm_min_max_sm86
__device__ void nvvm_min_max_sm86() {
#if __CUDA_ARCH__ >= 860

  // CHECK_PTX72_SM86: call bfloat @llvm.nvvm.fmin.xorsign.abs.bf16
  __nvvm_fmin_xorsign_abs_bf16(BF16, BF16_2);
  // CHECK_PTX72_SM86: call bfloat @llvm.nvvm.fmin.nan.xorsign.abs.bf16
  __nvvm_fmin_nan_xorsign_abs_bf16(BF16, NANBF16);
  // CHECK_PTX72_SM86: call <2 x bfloat> @llvm.nvvm.fmin.xorsign.abs.bf16x2
  __nvvm_fmin_xorsign_abs_bf16x2(BF16X2, BF16X2_2);
  // CHECK_PTX72_SM86: call <2 x bfloat> @llvm.nvvm.fmin.nan.xorsign.abs.bf16x2
  __nvvm_fmin_nan_xorsign_abs_bf16x2(BF16X2, NANBF16X2);
  // CHECK_PTX72_SM86: call float @llvm.nvvm.fmin.xorsign.abs.f
  __nvvm_fmin_xorsign_abs_f(-0.1f, 0.1f);
  // CHECK_PTX72_SM86: call float @llvm.nvvm.fmin.ftz.xorsign.abs.f
  __nvvm_fmin_ftz_xorsign_abs_f(-0.1f, 0.1f);
  // CHECK_PTX72_SM86: call float @llvm.nvvm.fmin.nan.xorsign.abs.f
  __nvvm_fmin_nan_xorsign_abs_f(-0.1f, (float)NAN32);
  // CHECK_PTX72_SM86: call float @llvm.nvvm.fmin.ftz.nan.xorsign.abs.f
  __nvvm_fmin_ftz_nan_xorsign_abs_f(-0.1f, (float)NAN32);

  // CHECK_PTX72_SM86: call bfloat @llvm.nvvm.fmax.xorsign.abs.bf16
  __nvvm_fmax_xorsign_abs_bf16(BF16, BF16_2);
  // CHECK_PTX72_SM86: call bfloat @llvm.nvvm.fmax.nan.xorsign.abs.bf16
  __nvvm_fmax_nan_xorsign_abs_bf16(BF16, NANBF16);
  // CHECK_PTX72_SM86: call <2 x bfloat> @llvm.nvvm.fmax.xorsign.abs.bf16x2
  __nvvm_fmax_xorsign_abs_bf16x2(BF16X2, BF16X2_2);
  // CHECK_PTX72_SM86: call <2 x bfloat> @llvm.nvvm.fmax.nan.xorsign.abs.bf16x2
  __nvvm_fmax_nan_xorsign_abs_bf16x2(BF16X2, NANBF16X2);
  // CHECK_PTX72_SM86: call float @llvm.nvvm.fmax.xorsign.abs.f
  __nvvm_fmax_xorsign_abs_f(-0.1f, 0.1f);
  // CHECK_PTX72_SM86: call float @llvm.nvvm.fmax.ftz.xorsign.abs.f
  __nvvm_fmax_ftz_xorsign_abs_f(-0.1f, 0.1f);
  // CHECK_PTX72_SM86: call float @llvm.nvvm.fmax.nan.xorsign.abs.f
  __nvvm_fmax_nan_xorsign_abs_f(-0.1f, (float)NAN32);
  // CHECK_PTX72_SM86: call float @llvm.nvvm.fmax.ftz.nan.xorsign.abs.f
  __nvvm_fmax_ftz_nan_xorsign_abs_f(-0.1f, (float)NAN32);
#endif
  // CHECK: ret void
}
