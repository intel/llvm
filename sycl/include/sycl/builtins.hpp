//==----------- builtins.hpp - SYCL built-in functions ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>

#include <sycl/detail/builtins/builtins.hpp>

#ifdef __SYCL_DEVICE_ONLY__
extern "C" {

extern __DPCPP_SYCL_EXTERNAL_LIBC void *memcpy(void *dest, const void *src,
                                               size_t n);
extern __DPCPP_SYCL_EXTERNAL_LIBC void *memset(void *dest, int c, size_t n);
extern __DPCPP_SYCL_EXTERNAL_LIBC int memcmp(const void *s1, const void *s2,
                                             size_t n);
extern __DPCPP_SYCL_EXTERNAL_LIBC size_t strlen(const char *s);
extern __DPCPP_SYCL_EXTERNAL_LIBC char *strcpy(char *dest, const char *src);
extern __DPCPP_SYCL_EXTERNAL_LIBC char *strncpy(char *dest, const char *src,
                                                size_t n);
extern __DPCPP_SYCL_EXTERNAL_LIBC int strcmp(const char *s1, const char *s2);
extern __DPCPP_SYCL_EXTERNAL_LIBC int strncmp(const char *s1, const char *s2,
                                              size_t n);
extern __DPCPP_SYCL_EXTERNAL_LIBC int rand();
extern __DPCPP_SYCL_EXTERNAL_LIBC void srand(unsigned int seed);
extern __DPCPP_SYCL_EXTERNAL float __imf_fadd_rd(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fadd_rn(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fadd_ru(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fadd_rz(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fsub_rd(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fsub_rn(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fsub_ru(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fsub_rz(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmul_rd(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmul_rn(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmul_ru(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmul_rz(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fdiv_rd(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fdiv_rn(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fdiv_ru(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fdiv_rz(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_frcp_rd(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_frcp_rn(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_frcp_ru(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_frcp_rz(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmaf_rd(float x, float y, float z);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmaf_rn(float x, float y, float z);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmaf_ru(float x, float y, float z);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmaf_rz(float x, float y, float z);
extern __DPCPP_SYCL_EXTERNAL float __imf_sqrtf_rd(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_sqrtf_rn(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_sqrtf_ru(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_sqrtf_rz(float x);
extern __DPCPP_SYCL_EXTERNAL double __imf_fma_rd(double x, double y, double z);
extern __DPCPP_SYCL_EXTERNAL double __imf_fma_rn(double x, double y, double z);
extern __DPCPP_SYCL_EXTERNAL double __imf_fma_ru(double x, double y, double z);
extern __DPCPP_SYCL_EXTERNAL double __imf_fma_rz(double x, double y, double z);
extern __DPCPP_SYCL_EXTERNAL double __imf_dadd_rd(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dadd_rn(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dadd_ru(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dadd_rz(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dsub_rd(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dsub_rn(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dsub_ru(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dsub_rz(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dmul_rd(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dmul_rn(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dmul_ru(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dmul_rz(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_ddiv_rd(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_ddiv_rn(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_ddiv_ru(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_ddiv_rz(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_drcp_rd(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_drcp_rn(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_drcp_ru(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_drcp_rz(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sqrt_rd(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sqrt_rn(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sqrt_ru(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sqrt_rz(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabs2(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabs4(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsss2(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsss4(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vneg2(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vneg4(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vnegss2(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vnegss4(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsdiffs2(unsigned int x,
                                                           unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsdiffs4(unsigned int x,
                                                           unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsdiffu2(unsigned int x,
                                                           unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsdiffu4(unsigned int x,
                                                           unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vadd2(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vadd4(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vaddss2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vaddss4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vaddus2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vaddus4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsub2(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsub4(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsubss2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsubss4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsubus2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsubus4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vavgs2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vavgs4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vavgu2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vavgu4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vhaddu2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vhaddu4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpeq2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpeq4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpne2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpne4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpges2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpges4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgeu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgeu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgts2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgts4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgtu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgtu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmples2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmples4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpleu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpleu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmplts2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmplts4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpltu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpltu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmaxs2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmaxs4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmaxu2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmaxu4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmins2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmins4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vminu2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vminu4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vseteq2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vseteq4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetne2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetne4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetges2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetges4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgeu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgeu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgts2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgts4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgtu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgtu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetles2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetles4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetleu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetleu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetlts2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetlts4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetltu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetltu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsads2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsads4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsadu2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsadu4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_viaddmax_s16x2(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_viaddmax_s16x2_relu(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_viaddmax_s32(int x, int y, int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_viaddmax_s32_relu(int x, int y, int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_viaddmax_u16x2(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_viaddmax_u32(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_viaddmin_s16x2(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_viaddmin_s16x2_relu(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_viaddmin_s32(int x, int y, int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_viaddmin_s32_relu(int x, int y, int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_viaddmin_u16x2(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_viaddmin_u32(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vibmax_s16x2(unsigned int x,
                                                             unsigned int y,
                                                             bool *pred_hi,
                                                             bool *pred_lo);
extern __DPCPP_SYCL_EXTERNAL int __imf_vibmax_s32(int x, int y, bool *pred);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vibmax_u16x2(unsigned int x,
                                                             unsigned int y,
                                                             bool *pred_hi,
                                                             bool *pred_lo);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vibmax_u32(unsigned int x, unsigned int y, bool *pred);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vibmin_s16x2(unsigned int x,
                                                             unsigned int y,
                                                             bool *pred_hi,
                                                             bool *pred_lo);
extern __DPCPP_SYCL_EXTERNAL int __imf_vibmin_s32(int x, int y, bool *pred);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vibmin_u16x2(unsigned int x,
                                                             unsigned int y,
                                                             bool *pred_hi,
                                                             bool *pred_lo);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vibmin_u32(unsigned int x, unsigned int y, bool *pred);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimax3_s16x2(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimax3_s16x2_relu(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimin3_s16x2(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimin3_s16x2_relu(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_vimax3_s32(int x, int y, int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_vimax3_s32_relu(int x, int y, int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_vimin3_s32(int x, int y, int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_vimin3_s32_relu(int x, int y, int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimax3_u16x2(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimax3_u32(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimin3_u16x2(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimin3_u32(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimax_s16x2_relu(unsigned int x, unsigned int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_vimax_s32_relu(int x, int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimin_s16x2_relu(unsigned int x, unsigned int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_vimin_s32_relu(int x, int y);
}
#ifdef __GLIBC__
namespace std {
extern __DPCPP_SYCL_EXTERNAL_LIBC void
__glibcxx_assert_fail(const char *file, int line, const char *func,
                      const char *cond) noexcept;
} // namespace std
extern "C" {
extern __DPCPP_SYCL_EXTERNAL_LIBC void __assert_fail(const char *expr,
                                                     const char *file,
                                                     unsigned int line,
                                                     const char *func);
}
#elif defined(_WIN32)
extern "C" {
// TODO: documented C runtime library APIs must be recognized as
//       builtins by FE. This includes _dpcomp, _dsign, _dtest,
//       _fdpcomp, _fdsign, _fdtest, _hypotf, _wassert.
//       APIs used by STL, such as _Cosh, are undocumented, even though
//       they are open-sourced. Recognizing them as builtins is not
//       straightforward currently.
extern __DPCPP_SYCL_EXTERNAL_LIBC void
_wassert(const wchar_t *wexpr, const wchar_t *wfile, unsigned line);
}
#endif
#endif // __SYCL_DEVICE_ONLY__
