//===- DeviceLibFunctions.h - record the functions in each device library--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef __DEVICELIB_FUNCS_LIST__
#define __DEVICELIB_FUNCS_LIST__
#include <string>
// All __devicelib_* functions must be sorted in alphabetic order as we
// will use binary search to find some entry in them.
static std::string CmathDeviceLibFunctions[] = {
    "__devicelib_acosf",  "__devicelib_acoshf",     "__devicelib_asinf",
    "__devicelib_asinhf", "__devicelib_atan2f",     "__devicelib_atanf",
    "__devicelib_atanhf", "__devicelib_cbrtf",      "__devicelib_cosf",
    "__devicelib_coshf",  "__devicelib_erfcf",      "__devicelib_erff",
    "__devicelib_exp2f",  "__devicelib_expf",       "__devicelib_expm1f",
    "__devicelib_fdimf",  "__devicelib_fmaf",       "__devicelib_fmodf",
    "__devicelib_frexpf", "__devicelib_hypotf",     "__devicelib_ilogbf",
    "__devicelib_ldexpf", "__devicelib_lgammaf",    "__devicelib_log10f",
    "__devicelib_log1pf", "__devicelib_log2f",      "__devicelib_logbf",
    "__devicelib_logf",   "__devicelib_modff",      "__devicelib_nextafterf",
    "__devicelib_powf",   "__devicelib_remainderf", "__devicelib_remquof",
    "__devicelib_sinf",   "__devicelib_sinhf",      "__devicelib_sqrtf",
    "__devicelib_tanf",   "__devicelib_tanhf",      "__devicelib_tgammaf"};

static std::string Cmath64DeviceLibFunctions[] = {
    "__devicelib_acos",  "__devicelib_acosh",     "__devicelib_asin",
    "__devicelib_asinh", "__devicelib_atan",      "__devicelib_atan2",
    "__devicelib_atanh", "__devicelib_cbrt",      "__devicelib_cos",
    "__devicelib_cosh",  "__devicelib_erf",       "__devicelib_erfc",
    "__devicelib_exp",   "__devicelib_exp2",      "__devicelib_expm1",
    "__devicelib_fdim",  "__devicelib_fma",       "__devicelib_fmod",
    "__devicelib_frexp", "__devicelib_hypot",     "__devicelib_ilogb",
    "__devicelib_ldexp", "__devicelib_lgamma",    "__devicelib_log",
    "__devicelib_log10", "__devicelib_log1p",     "__devicelib_log2",
    "__devicelib_logb",  "__devicelib_modf",      "__devicelib_nextafter",
    "__devicelib_pow",   "__devicelib_remainder", "__devicelib_remquo",
    "__devicelib_sin",   "__devicelib_sinh",      "__devicelib_sqrt",
    "__devicelib_tan",   "__devicelib_tanh",      "__devicelib_tgamma"};

static std::string ComplexDeviceLibFunctions[] = {
    "__devicelib___divsc3", "__devicelib___mulsc3", "__devicelib_cabsf",
    "__devicelib_cacosf",   "__devicelib_cacoshf",  "__devicelib_cargf",
    "__devicelib_casinf",   "__devicelib_casinhf",  "__devicelib_catanf",
    "__devicelib_catanhf",  "__devicelib_ccosf",    "__devicelib_ccoshf",
    "__devicelib_cexpf",    "__devicelib_cimagf",   "__devicelib_clogf",
    "__devicelib_cpolarf",  "__devicelib_cpowf",    "__devicelib_cprojf",
    "__devicelib_crealf",   "__devicelib_csinf",    "__devicelib_csinhf",
    "__devicelib_csqrtf",   "__devicelib_ctanf",    "__devicelib_ctanhf"};

static std::string Complex64DeviceLibFunctions[] = {
    "__devicelib___divdc3", "__devicelib___muldc3", "__devicelib_cabs",
    "__devicelib_cacos",    "__devicelib_cacosh",   "__devicelib_carg",
    "__devicelib_casin",    "__devicelib_casinh",   "__devicelib_catan",
    "__devicelib_catanh",   "__devicelib_ccos",     "__devicelib_ccosh",
    "__devicelib_cexp",     "__devicelib_cimag",    "__devicelib_clog",
    "__devicelib_cpolar",   "__devicelib_cpow",     "__devicelib_cproj",
    "__devicelib_creal",    "__devicelib_csin",     "__devicelib_csinh",
    "__devicelib_csqrt",    "__devicelib_ctan",     "__devicelib_ctanh"};
#endif
