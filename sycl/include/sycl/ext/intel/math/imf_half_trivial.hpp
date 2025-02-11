//==------------- imf_half_trivial.hpp - trivial half utils ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Trival half util functions.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/builtins.hpp>
#include <sycl/half_type.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::math {
inline sycl::half hadd(sycl::half x, sycl::half y) { return x + y; }

inline sycl::half hadd_sat(sycl::half x, sycl::half y) {
  return sycl::clamp((x + y), sycl::half(0.f), sycl::half(1.0f));
}

inline sycl::half hfma(sycl::half x, sycl::half y, sycl::half z) {
  return sycl::fma(x, y, z);
}

inline sycl::half hfma_sat(sycl::half x, sycl::half y, sycl::half z) {
  return sycl::clamp(sycl::fma(x, y, z), sycl::half(0.f), sycl::half(1.0f));
}

inline sycl::half hmul(sycl::half x, sycl::half y) { return x * y; }

inline sycl::half hmul_sat(sycl::half x, sycl::half y) {
  return sycl::clamp((x * y), sycl::half(0.f), sycl::half(1.0f));
}

inline sycl::half hneg(sycl::half x) { return -x; }

inline sycl::half hsub(sycl::half x, sycl::half y) { return x - y; }

inline sycl::half hsub_sat(sycl::half x, sycl::half y) {
  return sycl::clamp((x - y), sycl::half(0.f), sycl::half(1.0f));
}

inline sycl::half hdiv(sycl::half x, sycl::half y) { return x / y; }

inline bool heq(sycl::half x, sycl::half y) { return x == y; }

inline bool hequ(sycl::half x, sycl::half y) {
  if (sycl::isnan(x) || sycl::isnan(y))
    return true;
  else
    return x == y;
}

inline bool hge(sycl::half x, sycl::half y) { return x >= y; }

inline bool hgeu(sycl::half x, sycl::half y) {
  if (sycl::isnan(x) || sycl::isnan(y))
    return true;
  else
    return x >= y;
}

inline bool hgt(sycl::half x, sycl::half y) { return x > y; }

inline bool hgtu(sycl::half x, sycl::half y) {
  if (sycl::isnan(x) || sycl::isnan(y))
    return true;
  else
    return x > y;
}

inline bool hle(sycl::half x, sycl::half y) { return x <= y; }

inline bool hleu(sycl::half x, sycl::half y) {
  if (sycl::isnan(x) || sycl::isnan(y))
    return true;
  else
    return x <= y;
}

inline bool hlt(sycl::half x, sycl::half y) { return x < y; }

inline bool hltu(sycl::half x, sycl::half y) {
  if (sycl::isnan(x) || sycl::isnan(y))
    return true;
  return x < y;
}

inline bool hne(sycl::half x, sycl::half y) {
  if (sycl::isnan(x) || sycl::isnan(y))
    return false;
  return x != y;
}

inline bool hneu(sycl::half x, sycl::half y) {
  if (sycl::isnan(x) || sycl::isnan(y))
    return true;
  else
    return x != y;
}

inline bool hisinf(sycl::half x) { return sycl::isinf(x); }
inline bool hisnan(sycl::half y) { return sycl::isnan(y); }

inline sycl::half2 hadd2(sycl::half2 x, sycl::half2 y) { return x + y; }

inline sycl::half2 hadd2_sat(sycl::half2 x, sycl::half2 y) {
  return sycl::clamp((x + y), sycl::half2{0.f, 0.f}, sycl::half2{1.f, 1.f});
}

inline sycl::half2 hfma2(sycl::half2 x, sycl::half2 y, sycl::half2 z) {
  return sycl::fma(x, y, z);
}

inline sycl::half2 hfma2_sat(sycl::half2 x, sycl::half2 y, sycl::half2 z) {
  return sycl::clamp(sycl::fma(x, y, z), sycl::half2{0.f, 0.f},
                     sycl::half2{1.f, 1.f});
}

inline sycl::half2 hmul2(sycl::half2 x, sycl::half2 y) { return x * y; }

inline sycl::half2 hmul2_sat(sycl::half2 x, sycl::half2 y) {
  return sycl::clamp((x * y), sycl::half2{0.f, 0.f}, sycl::half2{1.f, 1.f});
}

inline sycl::half2 h2div(sycl::half2 x, sycl::half2 y) { return x / y; }

inline sycl::half2 hneg2(sycl::half2 x) { return -x; }

inline sycl::half2 hsub2(sycl::half2 x, sycl::half2 y) { return x - y; }

inline sycl::half2 hsub2_sat(sycl::half2 x, sycl::half2 y) {
  return sycl::clamp((x - y), sycl::half2{0.f, 0.f}, sycl::half2{1.f, 1.f});
}

inline bool hbeq2(sycl::half2 x, sycl::half2 y) {
  return heq(x.s0(), y.s0()) && heq(x.s1(), y.s1());
}

inline bool hbequ2(sycl::half2 x, sycl::half2 y) {
  return hequ(x.s0(), y.s0()) && hequ(x.s1(), y.s1());
}

inline bool hbge2(sycl::half2 x, sycl::half2 y) {
  return hge(x.s0(), y.s0()) && hge(x.s1(), y.s1());
}

inline bool hbgeu2(sycl::half2 x, sycl::half2 y) {
  return hgeu(x.s0(), y.s0()) && hgeu(x.s1(), y.s1());
}

inline bool hbgt2(sycl::half2 x, sycl::half2 y) {
  return hgt(x.s0(), y.s0()) && hgt(x.s1(), y.s1());
}

inline bool hbgtu2(sycl::half2 x, sycl::half2 y) {
  return hgtu(x.s0(), y.s0()) && hgtu(x.s1(), y.s1());
}

inline bool hble2(sycl::half2 x, sycl::half2 y) {
  return hle(x.s0(), y.s0()) && hle(x.s1(), y.s1());
}

inline bool hbleu2(sycl::half2 x, sycl::half2 y) {
  return hleu(x.s0(), y.s0()) && hleu(x.s1(), y.s1());
}

inline bool hblt2(sycl::half2 x, sycl::half2 y) {
  return hlt(x.s0(), y.s0()) && hlt(x.s1(), y.s1());
}

inline bool hbltu2(sycl::half2 x, sycl::half2 y) {
  return hltu(x.s0(), y.s0()) && hltu(x.s1(), y.s1());
}

inline bool hbne2(sycl::half2 x, sycl::half2 y) {
  return hne(x.s0(), y.s0()) && hne(x.s1(), y.s1());
}

inline bool hbneu2(sycl::half2 x, sycl::half2 y) {
  return hneu(x.s0(), y.s0()) && hneu(x.s1(), y.s1());
}

inline sycl::half2 heq2(sycl::half2 x, sycl::half2 y) {
  return sycl::half2{(heq(x.s0(), y.s0()) ? 1.0f : 0.f),
                     (heq(x.s1(), y.s1()) ? 1.0f : 0.f)};
}

inline sycl::half2 hequ2(sycl::half2 x, sycl::half2 y) {
  return sycl::half2{(hequ(x.s0(), y.s0()) ? 1.0f : 0.f),
                     (hequ(x.s1(), y.s1()) ? 1.0f : 0.f)};
}

inline sycl::half2 hge2(sycl::half2 x, sycl::half2 y) {
  return sycl::half2{(hge(x.s0(), y.s0()) ? 1.0f : 0.f),
                     (hge(x.s1(), y.s1()) ? 1.0f : 0.f)};
}

inline sycl::half2 hgeu2(sycl::half2 x, sycl::half2 y) {
  return sycl::half2{(hgeu(x.s0(), y.s0()) ? 1.0f : 0.f),
                     (hgeu(x.s1(), y.s1()) ? 1.0f : 0.f)};
}

inline sycl::half2 hgt2(sycl::half2 x, sycl::half2 y) {
  return sycl::half2{(hgt(x.s0(), y.s0()) ? 1.0f : 0.f),
                     (hgt(x.s1(), y.s1()) ? 1.0f : 0.f)};
}

inline sycl::half2 hgtu2(sycl::half2 x, sycl::half2 y) {
  return sycl::half2{(hgtu(x.s0(), y.s0()) ? 1.0f : 0.f),
                     (hgtu(x.s1(), y.s1()) ? 1.0f : 0.f)};
}

inline sycl::half2 hle2(sycl::half2 x, sycl::half2 y) {
  return sycl::half2{(hle(x.s0(), y.s0()) ? 1.0f : 0.f),
                     (hle(x.s1(), y.s1()) ? 1.0f : 0.f)};
}

inline sycl::half2 hleu2(sycl::half2 x, sycl::half2 y) {
  return sycl::half2{(hleu(x.s0(), y.s0()) ? 1.0f : 0.f),
                     (hleu(x.s1(), y.s1()) ? 1.0f : 0.f)};
}

inline sycl::half2 hlt2(sycl::half2 x, sycl::half2 y) {
  return sycl::half2{(hlt(x.s0(), y.s0()) ? 1.0f : 0.f),
                     (hlt(x.s1(), y.s1()) ? 1.0f : 0.f)};
}

inline sycl::half2 hltu2(sycl::half2 x, sycl::half2 y) {
  return sycl::half2{(hltu(x.s0(), y.s0()) ? 1.0f : 0.f),
                     (hltu(x.s1(), y.s1()) ? 1.0f : 0.f)};
}

inline sycl::half2 hisnan2(sycl::half2 x) {
  return sycl::half2{(hisnan(x.s0()) ? 1.0f : 0.f),
                     (hisnan(x.s1()) ? 1.0f : 0.f)};
}

inline sycl::half2 hne2(sycl::half2 x, sycl::half2 y) {
  return sycl::half2{(hne(x.s0(), y.s0()) ? 1.0f : 0.f),
                     (hne(x.s1(), y.s1()) ? 1.0f : 0.f)};
}

inline sycl::half2 hneu2(sycl::half2 x, sycl::half2 y) {
  return sycl::half2{(hneu(x.s0(), y.s0()) ? 1.0f : 0.f),
                     (hneu(x.s1(), y.s1()) ? 1.0f : 0.f)};
}

inline sycl::half hmax(sycl::half x, sycl::half y) { return sycl::fmax(x, y); }

inline sycl::half hmax_nan(sycl::half x, sycl::half y) {
  if (hisnan(x) || hisnan(y))
    return sycl::half(std::numeric_limits<float>::quiet_NaN());
  else
    return sycl::fmax(x, y);
}

inline sycl::half2 hmax2(sycl::half2 x, sycl::half2 y) {
  return sycl::half2{hmax(x.s0(), y.s0()), hmax(x.s1(), y.s1())};
}

inline sycl::half2 hmax2_nan(sycl::half2 x, sycl::half2 y) {
  return sycl::half2{hmax_nan(x.s0(), y.s0()), hmax_nan(x.s1(), y.s1())};
}

inline sycl::half hmin(sycl::half x, sycl::half y) { return sycl::fmin(x, y); }

inline sycl::half hmin_nan(sycl::half x, sycl::half y) {
  if (hisnan(x) || hisnan(y))
    return sycl::half(std::numeric_limits<float>::quiet_NaN());
  else
    return sycl::fmin(x, y);
}

inline sycl::half2 hmin2(sycl::half2 x, sycl::half2 y) {
  return sycl::half2{hmin(x.s0(), y.s0()), hmin(x.s1(), y.s1())};
}

inline sycl::half2 hmin2_nan(sycl::half2 x, sycl::half2 y) {
  return sycl::half2{hmin_nan(x.s0(), y.s0()), hmin_nan(x.s1(), y.s1())};
}

inline sycl::half2 hcmadd(sycl::half2 x, sycl::half2 y, sycl::half2 z) {
  return sycl::half2{x.s0() * y.s0() - x.s1() * y.s1() + z.s0(),
                     x.s0() * y.s1() + x.s1() * y.s0() + z.s1()};
}

inline sycl::half hfma_relu(sycl::half x, sycl::half y, sycl::half z) {
  sycl::half r = sycl::fma(x, y, z);
  if (!hisnan(r)) {
    if (r < 0.f)
      return sycl::half{0.f};
    else
      return r;
  }
  return r;
}

inline sycl::half2 hfma2_relu(sycl::half2 x, sycl::half2 y, sycl::half2 z) {
  sycl::half2 r = sycl::fma(x, y, z);
  if (!hisnan(r.s0()) && r.s0() < 0.f)
    r.s0() = 0.f;
  if (!hisnan(r.s1()) && r.s1() < 0.f)
    r.s1() = 0.f;
  return r;
}

inline sycl::half habs(sycl::half x) { return sycl::fabs(x); }

inline sycl::half2 habs2(sycl::half2 x) { return sycl::fabs(x); }
} // namespace ext::intel::math
} // namespace _V1
} // namespace sycl
