//==----- SYCLDeviceLibReqMask.cpp - get SYCL devicelib required Info ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass goes through input module's function list to detect all SYCL
// devicelib functions invoked. Each devicelib function invoked is included in
// one 'fallback' SPIR-V library loaded by SYCL runtime. After scanning all
// functions in input module, a mask telling which SPIR-V libraries are needed
// by input module indeed will be returned. This mask will be saved and used by
// SYCL runtime later.
//===----------------------------------------------------------------------===//

#include "SYCLDeviceLibReqMask.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Module.h"

#include <string>
#include <unordered_map>

static constexpr char DEVICELIB_FUNC_PREFIX[] = "__devicelib_";

using namespace llvm;

namespace {

using SYCLDeviceLibFuncMap = std::unordered_map<std::string, DeviceLibExt>;

// Please update SDLMap if any item is added to or removed from
// fallback device libraries in libdevice.
SYCLDeviceLibFuncMap SDLMap = {
    {"__devicelib_abs", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_acosf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_acoshf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_asinf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_asinhf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_atan2f", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_atanf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_atanhf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_cbrtf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_cosf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_coshf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_div", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_erfcf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_erff", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_exp2f", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_expf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_expm1f", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_fdimf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_fmaf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_fmodf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_frexpf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_hypotf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_ilogbf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_labs", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_ldiv", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_ldexpf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_lgammaf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_llabs", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_lldiv", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_log10f", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_log1pf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_log2f", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_logbf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_logf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_modff", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_nextafterf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_powf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_remainderf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_remquof", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_scalbnf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_sinf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_sinhf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_sqrtf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_tanf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_tanhf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_tgammaf", DeviceLibExt::cl_intel_devicelib_math},
    {"__devicelib_acos", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_acosh", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_asin", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_asinh", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_atan", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_atan2", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_atanh", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_cbrt", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_cos", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_cosh", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_erf", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_erfc", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_exp", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_exp2", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_expm1", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_fdim", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_fma", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_fmod", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_frexp", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_hypot", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_ilogb", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_ldexp", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_lgamma", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_log", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_log10", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_log1p", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_log2", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_logb", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_modf", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_nextafter", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_pow", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_remainder", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_remquo", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_scalbn", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_sin", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_sinh", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_sqrt", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_tan", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_tanh", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib_tgamma", DeviceLibExt::cl_intel_devicelib_math_fp64},
    {"__devicelib___divsc3", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib___mulsc3", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cabsf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cacosf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cacoshf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cargf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_casinf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_casinhf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_catanf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_catanhf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_ccosf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_ccoshf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cexpf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cimagf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_clogf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cpolarf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cpowf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_cprojf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_crealf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_csinf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_csinhf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_csqrtf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_ctanf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib_ctanhf", DeviceLibExt::cl_intel_devicelib_complex},
    {"__devicelib___divdc3", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib___muldc3", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_cabs", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_cacos", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_cacosh", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_carg", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_casin", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_casinh", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_catan", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_catanh", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_ccos", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_ccosh", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_cexp", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_cimag", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_clog", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_cpolar", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_cpow", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_cproj", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_creal", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_csin", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_csinh", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_csqrt", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_ctan", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_ctanh", DeviceLibExt::cl_intel_devicelib_complex_fp64},
    {"__devicelib_memcpy", DeviceLibExt::cl_intel_devicelib_cstring},
    {"__devicelib_memset", DeviceLibExt::cl_intel_devicelib_cstring},
    {"__devicelib_memcmp", DeviceLibExt::cl_intel_devicelib_cstring},
};

// Each fallback device library corresponds to one bit in "require mask" which
// is an unsigned int32. getDeviceLibBit checks which fallback device library
// is required for FuncName and returns the corresponding bit. The corresponding
// mask for each fallback device library is:
// fallback-cassert:      0x1
// fallback-cmath:        0x2
// fallback-cmath-fp64:   0x4
// fallback-complex:      0x8
// fallback-complex-fp64: 0x10
// fallback-cstring:      0x20
uint32_t getDeviceLibBits(const std::string &FuncName) {
  auto DeviceLibFuncIter = SDLMap.find(FuncName);
  return ((DeviceLibFuncIter == SDLMap.end())
              ? 0
              : 0x1 << (static_cast<uint32_t>(DeviceLibFuncIter->second) -
                        static_cast<uint32_t>(
                            DeviceLibExt::cl_intel_devicelib_assert)));
}

// For each device image module, we go through all functions which meets
// 1. The function name has prefix "__devicelib_"
// 2. The function is declaration which means it doesn't have function body
// And we don't expect non-spirv functions with "__devicelib_" prefix.
uint32_t getModuleDeviceLibReqMask(const Module &M) {
  // Device libraries will be enabled only for spir-v module.
  if (!llvm::Triple(M.getTargetTriple()).isSPIR())
    return 0;
  // 0x1 means sycl runtime will link and load libsycl-fallback-assert.spv as
  // default. In fact, default link assert spv is not necessary but dramatic
  // perf regression is observed if we don't link any device library. The perf
  // regression is caused by a clang issue.
  uint32_t ReqMask = 0x1;
  for (const Function &SF : M) {
    if (SF.getName().startswith(DEVICELIB_FUNC_PREFIX) && SF.isDeclaration()) {
      assert(SF.getCallingConv() == CallingConv::SPIR_FUNC);
      uint32_t DeviceLibBits = getDeviceLibBits(SF.getName().str());
      ReqMask |= DeviceLibBits;
    }
  }
  return ReqMask;
}
} // namespace

char SYCLDeviceLibReqMaskPass::ID = 0;
bool SYCLDeviceLibReqMaskPass::runOnModule(Module &M) {
  MReqMask = getModuleDeviceLibReqMask(M);
  return false;
}
