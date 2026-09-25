//==-------------- imf_simd.hpp - APIS for simd emulation ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// APIs for simd emulation
//===----------------------------------------------------------------------===//

#pragma once

#include <type_traits>

extern "C" {
unsigned int __imf_vabs2(unsigned int);
unsigned int __imf_vabs4(unsigned int);
unsigned int __imf_vneg2(unsigned int);
unsigned int __imf_vneg4(unsigned int);
unsigned int __imf_vnegss2(unsigned int);
unsigned int __imf_vnegss4(unsigned int);
unsigned int __imf_vabsdiffs2(unsigned int, unsigned int);
unsigned int __imf_vabsdiffs4(unsigned int, unsigned int);
unsigned int __imf_vabsdiffu2(unsigned int, unsigned int);
unsigned int __imf_vabsdiffu4(unsigned int, unsigned int);
unsigned int __imf_vabsss2(unsigned int);
unsigned int __imf_vabsss4(unsigned int);
unsigned int __imf_vadd2(unsigned int, unsigned int);
unsigned int __imf_vadd4(unsigned int, unsigned int);
unsigned int __imf_vaddss2(unsigned int, unsigned int);
unsigned int __imf_vaddss4(unsigned int, unsigned int);
unsigned int __imf_vaddus2(unsigned int, unsigned int);
unsigned int __imf_vaddus4(unsigned int, unsigned int);
unsigned int __imf_vsub2(unsigned int, unsigned int);
unsigned int __imf_vsub4(unsigned int, unsigned int);
unsigned int __imf_vsubss2(unsigned int, unsigned int);
unsigned int __imf_vsubss4(unsigned int, unsigned int);
unsigned int __imf_vsubus2(unsigned int, unsigned int);
unsigned int __imf_vsubus4(unsigned int, unsigned int);
unsigned int __imf_vhaddu2(unsigned int, unsigned int);
unsigned int __imf_vhaddu4(unsigned int, unsigned int);
unsigned int __imf_vavgs2(unsigned int, unsigned int);
unsigned int __imf_vavgs4(unsigned int, unsigned int);
unsigned int __imf_vavgu2(unsigned int, unsigned int);
unsigned int __imf_vavgu4(unsigned int, unsigned int);
unsigned int __imf_vcmpeq2(unsigned int, unsigned int);
unsigned int __imf_vcmpeq4(unsigned int, unsigned int);
unsigned int __imf_vcmpges2(unsigned int, unsigned int);
unsigned int __imf_vcmpges4(unsigned int, unsigned int);
unsigned int __imf_vcmpgeu2(unsigned int, unsigned int);
unsigned int __imf_vcmpgeu4(unsigned int, unsigned int);
unsigned int __imf_vcmpgts2(unsigned int, unsigned int);
unsigned int __imf_vcmpgts4(unsigned int, unsigned int);
unsigned int __imf_vcmpgtu2(unsigned int, unsigned int);
unsigned int __imf_vcmpgtu4(unsigned int, unsigned int);
unsigned int __imf_vcmples2(unsigned int, unsigned int);
unsigned int __imf_vcmples4(unsigned int, unsigned int);
unsigned int __imf_vcmpleu2(unsigned int, unsigned int);
unsigned int __imf_vcmpleu4(unsigned int, unsigned int);
unsigned int __imf_vcmplts2(unsigned int, unsigned int);
unsigned int __imf_vcmplts4(unsigned int, unsigned int);
unsigned int __imf_vcmpltu2(unsigned int, unsigned int);
unsigned int __imf_vcmpltu4(unsigned int, unsigned int);
unsigned int __imf_vcmpne2(unsigned int, unsigned int);
unsigned int __imf_vcmpne4(unsigned int, unsigned int);
unsigned int __imf_vmaxs2(unsigned int, unsigned int);
unsigned int __imf_vmaxs4(unsigned int, unsigned int);
unsigned int __imf_vmaxu2(unsigned int, unsigned int);
unsigned int __imf_vmaxu4(unsigned int, unsigned int);
unsigned int __imf_vmins2(unsigned int, unsigned int);
unsigned int __imf_vmins4(unsigned int, unsigned int);
unsigned int __imf_vminu2(unsigned int, unsigned int);
unsigned int __imf_vminu4(unsigned int, unsigned int);
unsigned int __imf_vseteq2(unsigned int, unsigned int);
unsigned int __imf_vseteq4(unsigned int, unsigned int);
unsigned int __imf_vsetne2(unsigned int, unsigned int);
unsigned int __imf_vsetne4(unsigned int, unsigned int);
unsigned int __imf_vsetges2(unsigned int, unsigned int);
unsigned int __imf_vsetges4(unsigned int, unsigned int);
unsigned int __imf_vsetgeu2(unsigned int, unsigned int);
unsigned int __imf_vsetgeu4(unsigned int, unsigned int);
unsigned int __imf_vsetgts2(unsigned int, unsigned int);
unsigned int __imf_vsetgts4(unsigned int, unsigned int);
unsigned int __imf_vsetgtu2(unsigned int, unsigned int);
unsigned int __imf_vsetgtu4(unsigned int, unsigned int);
unsigned int __imf_vsetles2(unsigned int, unsigned int);
unsigned int __imf_vsetles4(unsigned int, unsigned int);
unsigned int __imf_vsetleu2(unsigned int, unsigned int);
unsigned int __imf_vsetleu4(unsigned int, unsigned int);
unsigned int __imf_vsetlts2(unsigned int, unsigned int);
unsigned int __imf_vsetlts4(unsigned int, unsigned int);
unsigned int __imf_vsetltu2(unsigned int, unsigned int);
unsigned int __imf_vsetltu4(unsigned int, unsigned int);
unsigned int __imf_vsads2(unsigned int, unsigned int);
unsigned int __imf_vsads4(unsigned int, unsigned int);
unsigned int __imf_vsadu2(unsigned int, unsigned int);
unsigned int __imf_vsadu4(unsigned int, unsigned int);
unsigned int __imf_viaddmax_s16x2(unsigned int, unsigned int, unsigned int);
unsigned int __imf_viaddmax_s16x2_relu(unsigned int, unsigned int,
                                       unsigned int);
int __imf_viaddmax_s32(int, int, int);
int __imf_viaddmax_s32_relu(int, int, int);
unsigned int __imf_viaddmax_u16x2(unsigned int, unsigned int, unsigned int);
unsigned int __imf_viaddmax_u32(unsigned int, unsigned int, unsigned int);

unsigned int __imf_viaddmin_s16x2(unsigned int, unsigned int, unsigned int);
unsigned int __imf_viaddmin_s16x2_relu(unsigned int, unsigned int,
                                       unsigned int);
int __imf_viaddmin_s32(int, int, int);
int __imf_viaddmin_s32_relu(int, int, int);
unsigned int __imf_viaddmin_u16x2(unsigned int, unsigned int, unsigned int);
unsigned int __imf_viaddmin_u32(unsigned int, unsigned int, unsigned int);
unsigned int __imf_vibmax_s16x2(unsigned int, unsigned int, bool *, bool *);
int __imf_vibmax_s32(int, int, bool *);
unsigned int __imf_vibmax_u16x2(unsigned int, unsigned int, bool *, bool *);
unsigned int __imf_vibmax_u32(unsigned int, unsigned int, bool *);
unsigned int __imf_vibmin_s16x2(unsigned int, unsigned int, bool *, bool *);
int __imf_vibmin_s32(int, int, bool *);
unsigned int __imf_vibmin_u16x2(unsigned int, unsigned int, bool *, bool *);
unsigned int __imf_vibmin_u32(unsigned int, unsigned int, bool *);
unsigned int __imf_vimax3_s16x2(unsigned int, unsigned int, unsigned int);
unsigned int __imf_vimax3_s16x2_relu(unsigned int, unsigned int, unsigned int);
unsigned int __imf_vimin3_s16x2(unsigned int, unsigned int, unsigned int);
unsigned int __imf_vimin3_s16x2_relu(unsigned int, unsigned int, unsigned int);
int __imf_vimax3_s32(int, int, int);
int __imf_vimax3_s32_relu(int, int, int);
int __imf_vimin3_s32(int, int, int);
int __imf_vimin3_s32_relu(int, int, int);
unsigned int __imf_vimax3_u16x2(unsigned int, unsigned int, unsigned int);
unsigned int __imf_vimax3_u32(unsigned int, unsigned int, unsigned int);
unsigned int __imf_vimin3_u16x2(unsigned int, unsigned int, unsigned int);
unsigned int __imf_vimin3_u32(unsigned int, unsigned int, unsigned int);
unsigned int __imf_vimax_s16x2_relu(unsigned int, unsigned int);
int __imf_vimax_s32_relu(int, int);
unsigned int __imf_vimin_s16x2_relu(unsigned int, unsigned int);
int __imf_vimin_s32_relu(int, int);
};

namespace sycl {
inline namespace _V1 {
namespace ext::intel::math {

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vabs2(Tp x) {
  return __imf_vabs2(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vabs4(Tp x) {
  return __imf_vabs4(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vneg2(Tp x) {
  return __imf_vneg2(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vneg4(Tp x) {
  return __imf_vneg4(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vnegss2(Tp x) {
  return __imf_vnegss2(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vnegss4(Tp x) {
  return __imf_vnegss4(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vabsdiffs2(Tp x, Tp y) {
  return __imf_vabsdiffs2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vabsdiffs4(Tp x, Tp y) {
  return __imf_vabsdiffs4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vabsdiffu2(Tp x, Tp y) {
  return __imf_vabsdiffu2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vabsdiffu4(Tp x, Tp y) {
  return __imf_vabsdiffu4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vabsss2(Tp x) {
  return __imf_vabsss2(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vabsss4(Tp x) {
  return __imf_vabsss4(x);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vadd2(Tp x,
                                                                       Tp y) {
  return __imf_vadd2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vadd4(Tp x,
                                                                       Tp y) {
  return __imf_vadd4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vaddss2(Tp x,
                                                                         Tp y) {
  return __imf_vaddss2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vaddss4(Tp x,
                                                                         Tp y) {
  return __imf_vaddss4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vaddus2(Tp x,
                                                                         Tp y) {
  return __imf_vaddus2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vaddus4(Tp x,
                                                                         Tp y) {
  return __imf_vaddus4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vsub2(Tp x,
                                                                       Tp y) {
  return __imf_vsub2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vsub4(Tp x,
                                                                       Tp y) {
  return __imf_vsub4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vsubss2(Tp x,
                                                                         Tp y) {
  return __imf_vsubss2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vsubss4(Tp x,
                                                                         Tp y) {
  return __imf_vsubss4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vsubus2(Tp x,
                                                                         Tp y) {
  return __imf_vsubus2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vsubus4(Tp x,
                                                                         Tp y) {
  return __imf_vsubus4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vhaddu2(Tp x,
                                                                         Tp y) {
  return __imf_vhaddu2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vhaddu4(Tp x,
                                                                         Tp y) {
  return __imf_vhaddu4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vavgs2(Tp x,
                                                                        Tp y) {
  return __imf_vavgs2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vavgs4(Tp x,
                                                                        Tp y) {
  return __imf_vavgs4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vavgu2(Tp x,
                                                                        Tp y) {
  return __imf_vavgu2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vavgu4(Tp x,
                                                                        Tp y) {
  return __imf_vavgu4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vcmpeq2(Tp x,
                                                                         Tp y) {
  return __imf_vcmpeq2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vcmpeq4(Tp x,
                                                                         Tp y) {
  return __imf_vcmpeq4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vcmpges2(Tp x, Tp y) {
  return __imf_vcmpges2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vcmpges4(Tp x, Tp y) {
  return __imf_vcmpges4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vcmpgeu2(Tp x, Tp y) {
  return __imf_vcmpgeu2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vcmpgeu4(Tp x, Tp y) {
  return __imf_vcmpgeu4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vcmpgts2(Tp x, Tp y) {
  return __imf_vcmpgts2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vcmpgts4(Tp x, Tp y) {
  return __imf_vcmpgts4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vcmpgtu2(Tp x, Tp y) {
  return __imf_vcmpgtu2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vcmpgtu4(Tp x, Tp y) {
  return __imf_vcmpgtu4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vcmples2(Tp x, Tp y) {
  return __imf_vcmples2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vcmples4(Tp x, Tp y) {
  return __imf_vcmples4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vcmpleu2(Tp x, Tp y) {
  return __imf_vcmpleu2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vcmpleu4(Tp x, Tp y) {
  return __imf_vcmpleu4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vcmplts2(Tp x, Tp y) {
  return __imf_vcmplts2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vcmplts4(Tp x, Tp y) {
  return __imf_vcmplts4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vcmpltu2(Tp x, Tp y) {
  return __imf_vcmpltu2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vcmpltu4(Tp x, Tp y) {
  return __imf_vcmpltu4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vcmpne2(Tp x,
                                                                         Tp y) {
  return __imf_vcmpne2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vcmpne4(Tp x,
                                                                         Tp y) {
  return __imf_vcmpne4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vmaxs2(Tp x,
                                                                        Tp y) {
  return __imf_vmaxs2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vmaxs4(Tp x,
                                                                        Tp y) {
  return __imf_vmaxs4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vmaxu2(Tp x,
                                                                        Tp y) {
  return __imf_vmaxu2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vmaxu4(Tp x,
                                                                        Tp y) {
  return __imf_vmaxu4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vmins2(Tp x,
                                                                        Tp y) {
  return __imf_vmins2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vmins4(Tp x,
                                                                        Tp y) {
  return __imf_vmins4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vminu2(Tp x,
                                                                        Tp y) {
  return __imf_vminu2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vminu4(Tp x,
                                                                        Tp y) {
  return __imf_vminu4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vseteq2(Tp x,
                                                                         Tp y) {
  return __imf_vseteq2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vseteq4(Tp x,
                                                                         Tp y) {
  return __imf_vseteq4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vsetne2(Tp x,
                                                                         Tp y) {
  return __imf_vsetne2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vsetne4(Tp x,
                                                                         Tp y) {
  return __imf_vsetne4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vsetges2(Tp x, Tp y) {
  return __imf_vsetges2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vsetges4(Tp x, Tp y) {
  return __imf_vsetges4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vsetgeu2(Tp x, Tp y) {
  return __imf_vsetgeu2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vsetgeu4(Tp x, Tp y) {
  return __imf_vsetgeu4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vsetgts2(Tp x, Tp y) {
  return __imf_vsetgts2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vsetgts4(Tp x, Tp y) {
  return __imf_vsetgts4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vsetgtu2(Tp x, Tp y) {
  return __imf_vsetgtu2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vsetgtu4(Tp x, Tp y) {
  return __imf_vsetgtu4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vsetles2(Tp x, Tp y) {
  return __imf_vsetles2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vsetles4(Tp x, Tp y) {
  return __imf_vsetles4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vsetleu2(Tp x, Tp y) {
  return __imf_vsetleu2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vsetleu4(Tp x, Tp y) {
  return __imf_vsetleu4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vsetlts2(Tp x, Tp y) {
  return __imf_vsetlts2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vsetlts4(Tp x, Tp y) {
  return __imf_vsetlts4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vsetltu2(Tp x, Tp y) {
  return __imf_vsetltu2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vsetltu4(Tp x, Tp y) {
  return __imf_vsetltu4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vsads2(Tp x,
                                                                        Tp y) {
  return __imf_vsads2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vsads4(Tp x,
                                                                        Tp y) {
  return __imf_vsads4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vsadu2(Tp x,
                                                                        Tp y) {
  return __imf_vsadu2(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int> vsadu4(Tp x,
                                                                        Tp y) {
  return __imf_vsadu4(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
viaddmax_s16x2(Tp x, Tp y, Tp z) {
  return __imf_viaddmax_s16x2(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
viaddmax_s16x2_relu(Tp x, Tp y, Tp z) {
  return __imf_viaddmax_s16x2_relu(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, int>, int> viaddmax_s32(Tp x, Tp y, Tp z) {
  return __imf_viaddmax_s32(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, int>, int> viaddmax_s32_relu(Tp x, Tp y,
                                                                 Tp z) {
  return __imf_viaddmax_s32_relu(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
viaddmax_u16x2(Tp x, Tp y, Tp z) {
  return __imf_viaddmax_u16x2(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
viaddmax_u32(Tp x, Tp y, Tp z) {
  return __imf_viaddmax_u32(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
viaddmin_s16x2(Tp x, Tp y, Tp z) {
  return __imf_viaddmin_s16x2(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
viaddmin_s16x2_relu(Tp x, Tp y, Tp z) {
  return __imf_viaddmin_s16x2_relu(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, int>, int> viaddmin_s32(Tp x, Tp y, Tp z) {
  return __imf_viaddmin_s32(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, int>, int> viaddmin_s32_relu(Tp x, Tp y,
                                                                 Tp z) {
  return __imf_viaddmin_s32_relu(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
viaddmin_u16x2(Tp x, Tp y, Tp z) {
  return __imf_viaddmin_u16x2(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
viaddmin_u32(Tp x, Tp y, Tp z) {
  return __imf_viaddmin_u32(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vibmax_s16x2(Tp x, Tp y, bool *p_hi, bool *p_lo) {
  return __imf_vibmax_s16x2(x, y, p_hi, p_lo);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, int>, int> vibmax_s32(Tp x, Tp y, bool *p) {
  return __imf_vibmax_s32(x, y, p);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vibmax_u16x2(Tp x, Tp y, bool *p_hi, bool *p_lo) {
  return __imf_vibmax_u16x2(x, y, p_hi, p_lo);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vibmax_u32(Tp x, Tp y, bool *p) {
  return __imf_vibmax_u32(x, y, p);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vibmin_s16x2(Tp x, Tp y, bool *p_hi, bool *p_lo) {
  return __imf_vibmin_s16x2(x, y, p_hi, p_lo);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, int>, int> vibmin_s32(Tp x, Tp y, bool *p) {
  return __imf_vibmin_s32(x, y, p);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vibmin_u16x2(Tp x, Tp y, bool *p_hi, bool *p_lo) {
  return __imf_vibmin_u16x2(x, y, p_hi, p_lo);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vibmin_u32(Tp x, Tp y, bool *p) {
  return __imf_vibmin_u32(x, y, p);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vimax3_s16x2(Tp x, Tp y, Tp z) {
  return __imf_vimax3_s16x2(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vimin3_s16x2(Tp x, Tp y, Tp z) {
  return __imf_vimin3_s16x2(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vimax3_s16x2_relu(Tp x, Tp y, Tp z) {
  return __imf_vimax3_s16x2_relu(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vimin3_s16x2_relu(Tp x, Tp y, Tp z) {
  return __imf_vimin3_s16x2_relu(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, int>, int> vimax3_s32(Tp x, Tp y, Tp z) {
  return __imf_vimax3_s32(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, int>, int> vimin3_s32(Tp x, Tp y, Tp z) {
  return __imf_vimin3_s32(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, int>, int> vimax3_s32_relu(Tp x, Tp y,
                                                               Tp z) {
  return __imf_vimax3_s32_relu(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, int>, int> vimin3_s32_relu(Tp x, Tp y,
                                                               Tp z) {
  return __imf_vimin3_s32_relu(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vimax3_u16x2(Tp x, Tp y, Tp z) {
  return __imf_vimax3_u16x2(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vimin3_u16x2(Tp x, Tp y, Tp z) {
  return __imf_vimin3_u16x2(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vimax3_u32(Tp x, Tp y, Tp z) {
  return __imf_vimax3_u32(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vimin3_u32(Tp x, Tp y, Tp z) {
  return __imf_vimin3_u32(x, y, z);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, int>, int> vimax_s32_relu(Tp x, Tp y) {
  return __imf_vimax_s32_relu(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vimax_s16x2_relu(Tp x, Tp y) {
  return __imf_vimax_s16x2_relu(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, int>, int> vimin_s32_relu(Tp x, Tp y) {
  return __imf_vimin_s32_relu(x, y);
}

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, unsigned int>, unsigned int>
vimin_s16x2_relu(Tp x, Tp y) {
  return __imf_vimin_s16x2_relu(x, y);
}

} // namespace ext::intel::math
} // namespace _V1
} // namespace sycl
