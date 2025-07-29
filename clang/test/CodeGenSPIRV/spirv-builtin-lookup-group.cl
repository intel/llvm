// RUN: %clang_cc1 -triple=spir64 -cl-std=CL2.0 -fdeclare-spirv-builtins -emit-llvm %s -o - | FileCheck %s

void group_async_copy(short __attribute__((opencl_local)) *dst , short const __attribute__((opencl_global)) *src,
event_t event) {
  // CHECK-LABEL: @group_async_copy(
  // CHECK: tail call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyiPU3AS3sPU3AS1Ksmm9ocl_event(
  __spirv_GroupAsyncCopy(2, dst, src, 32, 16, event);
}

void group_wait_event(event_t event) {
  // CHECK-LABEL: @group_wait_event(
  // CHECK: call spir_func void @_Z23__spirv_GroupWaitEventsiiP9ocl_event(
  __spirv_GroupWaitEvents(1, 0, &event);
}

bool group_all(bool predicate) {
  // CHECK-LABEL: @group_all(
  // CHECK: call spir_func zeroext i1 @_Z16__spirv_GroupAllib(
  return __spirv_GroupAll(2, predicate);
}

bool group_any(bool predicate) {
  // CHECK-LABEL: @group_any(
  // CHECK: call spir_func zeroext i1 @_Z16__spirv_GroupAnyib(
  return __spirv_GroupAny(2, predicate);
}

char group_broad_cast(char a) {
  // CHECK-LABEL: @group_broad_cast(
  // CHECK: call spir_func signext i8 @_Z22__spirv_GroupBroadcasticj(
  return __spirv_GroupBroadcast(2, a, 0u);
}

int group_iadd(int a) {
  // CHECK-LABEL: @group_iadd(
  // CHECK: call spir_func i32 @_Z17__spirv_GroupIAddiii(
  return __spirv_GroupIAdd(2, 2, a);
}

int group_imul_khr(short a) {
  // CHECK-LABEL: @group_imul_khr(
  // CHECK: call spir_func signext i16 @_Z20__spirv_GroupIMulKHRiis(
  return __spirv_GroupIMulKHR(2, 0, a);
}

long group_bitwise_or_khr(long a) {
  // CHECK-LABEL: @group_bitwise_or_khr(
  // CHECK: call spir_func i64 @_Z25__spirv_GroupBitwiseOrKHRiil(
  return __spirv_GroupBitwiseOrKHR(2, 0, a);
}

float group_fadd(float a) {
  // CHECK-LABEL: @group_fadd(
  // CHECK: call spir_func float @_Z17__spirv_GroupFAddiif(
  return __spirv_GroupFAdd(2, 1, a);
}

float group_fmin(float a) {
  // CHECK-LABEL: @group_fmin(
  // CHECK: call spir_func float @_Z17__spirv_GroupFMiniif(
  return __spirv_GroupFMin(2, 0, a);
}

float group_fmax(float a) {
  // CHECK-LABEL: @group_fmax(
  // CHECK: call spir_func float @_Z17__spirv_GroupFMaxiif(
  return __spirv_GroupFMax(2, 2, a);
}

float group_fmul_khr(float a) {
  // CHECK-LABEL: @group_fmul_khr(
  // CHECK: call spir_func float @_Z20__spirv_GroupFMulKHRiif(
  return __spirv_GroupFMulKHR(2, 2, a);
}

unsigned char group_umin(unsigned char a ) {
  // CHECK-LABEL: @group_umin(
  // CHECK: call spir_func zeroext i8 @_Z17__spirv_GroupUMiniih(
  return __spirv_GroupUMin(2, 0, a);
}

unsigned long group_umax(unsigned long a) {
  // CHECK-LABEL: @group_umax(
  // CHECK: call spir_func i64 @_Z17__spirv_GroupUMaxiim(
  return __spirv_GroupUMax(2, 0, a);
}

char group_smin(char a) {
  // CHECK-LABEL: @group_smin(
  // CHECK: call spir_func signext i8 @_Z17__spirv_GroupSMiniic(
  return __spirv_GroupSMin(2, 0, a);
}

short group_smax(short a) {
  // CHECK-LABEL: @group_smax(
  // CHECK: call spir_func signext i16 @_Z17__spirv_GroupSMaxiis(
  return __spirv_GroupSMax(2, 0, a);
}

bool group_logical_and_khr(bool a) {
  // CHECK-LABEL: @group_logical_and_khr(
  // CHECK: call spir_func zeroext i1 @_Z26__spirv_GroupLogicalAndKHRiib(
  return __spirv_GroupLogicalAndKHR(2, 0, a);
}

bool group_logical_or_khr(bool a) {
  // CHECK-LABEL: @group_logical_or_khr(
  // CHECK: call spir_func zeroext i1 @_Z25__spirv_GroupLogicalOrKHRiib(
  return __spirv_GroupLogicalOrKHR(2, 0, a);
}
