//==- ternary_bitwise.hpp --- SYCL extension for ternary bitwise functions -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/builtins.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/detail/type_traits/vec_marray_traits.hpp>

#include <stdint.h>
#include <type_traits>

#pragma once

namespace sycl {
inline namespace _V1 {
namespace detail {
#if !defined(__SYCL_DEVICE_ONLY__) || defined(__NVPTX__) || defined(__AMDGCN__)
// Host implementation of the ternary bitwise operation LUT.
template <uint8_t LUTIndex, typename T> T applyTernaryBitwise(T A, T B, T C) {
  switch (LUTIndex) {
  case 0:
    return T(0);
  case 1:
    return (~(~C & B) & (~C & ~A));
  case 2:
    return (~(~C & B) & (~C & A));
  case 3:
    return (~C & ~B);
  case 4:
    return ~(~(~C & B) | (~C & A));
  case 5:
    return (~C & ~A);
  case 6:
    return (~(~C & B) ^ ~(~C & A));
  case 7:
    return ((~C & ~B) | (~C & ~A));
  case 8:
    return ~(~(~C & B) | ~(~C & A));
  case 9:
    return ~(~(~C & B) ^ (~C & ~A));
  case 10:
    return (~C & A);
  case 11:
    return ((~C & ~B) | (~C & A));
  case 12:
    return (~C & B);
  case 13:
    return ~(~(~C & B) & ~(~C & ~A));
  case 14:
    return ~(~(~C & B) & ~(~C & A));
  case 15:
    return ~C;
  case 16:
    return (~(~C | B) & ~(~C | A));
  case 17:
    return (~B & ~A);
  case 18:
    return ~(~(~C | B) ^ ~(~B & A));
  case 19:
    return ((~C & ~B) | (~B & ~A));
  case 20:
    return ~(~(~C ^ ~B) | A);
  case 21:
    return ((~C | ~B) & ~A);
  case 22:
    return ~((~(~C ^ ~B) & ~(~C & A)) | ~(~(~C ^ ~B) ^ ~A));
  case 23:
    return ~(((~C & ~B) | ~(~C ^ ~A)) ^ ~(~(~C | B) | ~(~C | ~A)));
  case 24:
    return ~(~(~C ^ ~B) | ~(~C ^ ~A));
  case 25:
    return ((~C | ~B) & ~(~B ^ ~A));
  case 26:
    return ((~C | ~B) & (~C ^ ~A));
  case 27:
    return ~(~(~C & A) ^ (~B & ~A));
  case 28:
    return ~(~(~C ^ ~B) | (~B & A));
  case 29:
    return ~(~(~C & B) ^ (~B & ~A));
  case 30:
    return (~(~C ^ ~B) ^ ~(~B & A));
  case 31:
    return ((~C | ~B) & (~C | ~A));
  case 32:
    return (~(~C | B) & (~C | A));
  case 33:
    return (~(~C | B) ^ (~B & ~A));
  case 34:
    return (~B & A);
  case 35:
    return ((~C & ~B) | (~B & A));
  case 36:
    return ~(~(~C ^ ~B) | (~C ^ ~A));
  case 37:
    return ((~C | ~B) & ~(~C ^ ~A));
  case 38:
    return ((~C | ~B) & (~B ^ ~A));
  case 39:
    return (~(~C | A) ^ (~B | ~A));
  case 40:
    return ~(~(~C ^ ~B) | ~A);
  case 41:
    return (~(~(~C ^ ~B) & (~C | A)) ^ ~A);
  case 42:
    return ((~C | ~B) & A);
  case 43:
    return ~((~(~C ^ ~B) & ~(~C & A)) ^ ((~C | ~B) & ~A));
  case 44:
    return ~(~(~C ^ ~B) | ~(~C | A));
  case 45:
    return ~(~(~C ^ ~B) ^ (~B & ~A));
  case 46:
    return (~(~C & B) ^ ~(~B & A));
  case 47:
    return ((~C | ~B) & (~C | A));
  case 48:
    return ~(~C | B);
  case 49:
    return (~(~C | B) | (~B & ~A));
  case 50:
    return (~(~C | B) | (~B & A));
  case 51:
    return ~B;
  case 52:
    return ~(~(~C ^ ~B) | (~C & A));
  case 53:
    return (~(~C | B) ^ (~C & ~A));
  case 54:
    return (~(~C ^ ~B) ^ ~(~C & A));
  case 55:
    return ((~C | ~B) & (~B | ~A));
  case 56:
    return ~(~(~C ^ ~B) | ~(~B | A));
  case 57:
    return ~(~(~C ^ ~B) ^ (~C & ~A));
  case 58:
    return ~(~(~C | B) ^ ~(~C & A));
  case 59:
    return ((~C | ~B) & (~B | A));
  case 60:
    return (~C ^ ~B);
  case 61:
    return ~(~(~C ^ ~B) & ~(~C & ~A));
  case 62:
    return ~(~(~C ^ ~B) & ~(~C & A));
  case 63:
    return (~C | ~B);
  case 64:
    return (~(~C & B) & ~(~B | A));
  case 65:
    return (~(~C ^ ~B) & ~A);
  case 66:
    return (~(~C ^ ~B) & (~C ^ ~A));
  case 67:
    return (~(~C ^ ~B) & (~C | ~A));
  case 68:
    return ~(~B | A);
  case 69:
    return ~(~(~C | B) | A);
  case 70:
    return ~(~(~C | B) | ~(~B ^ ~A));
  case 71:
    return (~(~C | B) ^ (~B | ~A));
  case 72:
    return ~(~(~C & B) ^ ~(~B | A));
  case 73:
    return ~((~(~C ^ ~B) & ~(~C & A)) ^ (~B | ~A));
  case 74:
    return ~(~(~C | B) | ~(~C ^ ~A));
  case 75:
    return ~(~(~C ^ ~B) ^ (~B | ~A));
  case 76:
    return ~(~(~C & B) & (~B | A));
  case 77:
    return ~((~(~C ^ ~B) & ~(~C & A)) ^ (~(~C & B) & (~B | ~A)));
  case 78:
    return ~(~(~C & A) ^ ~(~B | A));
  case 79:
    return ~(~(~C | B) | ~(~C | ~A));
  case 80:
    return ~(~C | A);
  case 81:
    return (~(~C & B) & ~A);
  case 82:
    return (~(~C & B) & (~C ^ ~A));
  case 83:
    return ~(~(~C & B) ^ (~C | ~A));
  case 84:
    return ~((~C & ~B) | A);
  case 85:
    return ~A;
  case 86:
    return (~(~C & B) ^ ~(~C ^ ~A));
  case 87:
    return ((~C & ~B) | ~A);
  case 88:
    return ~((~C & ~B) | ~(~C ^ ~A));
  case 89:
    return ~(~(~C & B) ^ ~A);
  case 90:
    return (~C ^ ~A);
  case 91:
    return ((~C & ~B) | (~C ^ ~A));
  case 92:
    return ~(~(~C & B) ^ ~(~C | A));
  case 93:
    return ~(~(~C & B) & A);
  case 94:
    return ~(~(~C & B) & ~(~C ^ ~A));
  case 95:
    return (~C | ~A);
  case 96:
    return (~(~C | B) ^ ~(~C | A));
  case 97:
    return ~((~(~C ^ ~B) & ~(~C & A)) ^ (~C | ~A));
  case 98:
    return (~(~C & B) & (~B ^ ~A));
  case 99:
    return ~(~(~C ^ ~B) ^ (~C | ~A));
  case 100:
    return ~((~C & ~B) | ~(~B ^ ~A));
  case 101:
    return (~(~C | B) ^ ~A);
  case 102:
    return (~B ^ ~A);
  case 103:
    return ((~C & ~B) | (~B ^ ~A));
  case 104:
    return ((~(~C ^ ~B) & ~(~C & A)) ^ ~(~(~C & B) ^ ~(~C ^ ~A)));
  case 105:
    return ~(~(~C ^ ~B) ^ ~A);
  case 106:
    return (~(~C & B) ^ ~(~B ^ ~A));
  case 107:
    return ~((~(~C ^ ~B) & ~(~C & A)) ^ ~A);
  case 108:
    return ~(~(~C ^ ~B) ^ ~(~C | A));
  case 109:
    return ~((~(~C ^ ~B) & ~(~C & A)) ^ ~(~(~C & B) ^ (~C | ~A)));
  case 110:
    return ~(~(~C & B) & ~(~B ^ ~A));
  case 111:
    return (~(~C | B) ^ (~C | ~A));
  case 112:
    return (~(~C | B) | ~(~C | A));
  case 113:
    return ~((~(~C ^ ~B) & ~(~C & A)) ^ ~(~(~C | B) | ~(~C | ~A)));
  case 114:
    return ~(~(~C | A) ^ ~(~B & A));
  case 115:
    return (~(~C & B) & (~B | ~A));
  case 116:
    return (~(~C | B) ^ ~(~B | A));
  case 117:
    return (~(~C | B) | ~A);
  case 118:
    return (~(~C | B) | (~B ^ ~A));
  case 119:
    return (~B | ~A);
  case 120:
    return ~(~(~C ^ ~B) ^ ~(~B | A));
  case 121:
    return ((~(~C ^ ~B) & ~(~C & A)) ^ ~(~(~C | B) ^ (~B | ~A)));
  case 122:
    return (~(~C | B) | (~C ^ ~A));
  case 123:
    return ~(~(~C & B) ^ (~B | ~A));
  case 124:
    return ~(~(~C ^ ~B) & (~C | A));
  case 125:
    return ~(~(~C ^ ~B) & A);
  case 126:
    return ~(~(~C ^ ~B) & ~(~C ^ ~A));
  case 127:
    return ~(~(~C & B) & ~(~B | ~A));
  case 128:
    return (~(~C & B) & ~(~B | ~A));
  case 129:
    return (~(~C ^ ~B) & ~(~C ^ ~A));
  case 130:
    return (~(~C ^ ~B) & A);
  case 131:
    return (~(~C ^ ~B) & (~C | A));
  case 132:
    return (~(~C & B) ^ (~B | ~A));
  case 133:
    return ~(~(~C | B) | (~C ^ ~A));
  case 134:
    return ~((~(~C ^ ~B) & ~(~C & A)) ^ ~(~(~C | B) ^ (~B | ~A)));
  case 135:
    return (~(~C ^ ~B) ^ ~(~B | A));
  case 136:
    return ~(~B | ~A);
  case 137:
    return ~(~(~C | B) | (~B ^ ~A));
  case 138:
    return ~(~(~C | B) | ~A);
  case 139:
    return ~(~(~C | B) ^ ~(~B | A));
  case 140:
    return ~(~(~C & B) & (~B | ~A));
  case 141:
    return (~(~C | A) ^ ~(~B & A));
  case 142:
    return ((~(~C ^ ~B) & ~(~C & A)) ^ ~(~(~C | B) | ~(~C | ~A)));
  case 143:
    return ~(~(~C | B) | ~(~C | A));
  case 144:
    return ~(~(~C | B) ^ (~C | ~A));
  case 145:
    return (~(~C & B) & ~(~B ^ ~A));
  case 146:
    return ((~(~C ^ ~B) & ~(~C & A)) ^ ~(~(~C & B) ^ (~C | ~A)));
  case 147:
    return (~(~C ^ ~B) ^ ~(~C | A));
  case 148:
    return ((~(~C ^ ~B) & ~(~C & A)) ^ ~A);
  case 149:
    return ~(~(~C & B) ^ ~(~B ^ ~A));
  case 150:
    return (~(~C ^ ~B) ^ ~A);
  case 151:
    return ~((~(~C ^ ~B) & ~(~C & A)) ^ ~(~(~C & B) ^ ~(~C ^ ~A)));
  case 152:
    return ~((~C & ~B) | (~B ^ ~A));
  case 153:
    return ~(~B ^ ~A);
  case 154:
    return ~(~(~C | B) ^ ~A);
  case 155:
    return ((~C & ~B) | ~(~B ^ ~A));
  case 156:
    return (~(~C ^ ~B) ^ (~C | ~A));
  case 157:
    return ~(~(~C & B) & (~B ^ ~A));
  case 158:
    return ((~(~C ^ ~B) & ~(~C & A)) ^ (~C | ~A));
  case 159:
    return ~(~(~C | B) ^ ~(~C | A));
  case 160:
    return ~(~C | ~A);
  case 161:
    return (~(~C & B) & ~(~C ^ ~A));
  case 162:
    return (~(~C & B) & A);
  case 163:
    return (~(~C & B) ^ ~(~C | A));
  case 164:
    return ~((~C & ~B) | (~C ^ ~A));
  case 165:
    return ~(~C ^ ~A);
  case 166:
    return (~(~C & B) ^ ~A);
  case 167:
    return ((~C & ~B) | ~(~C ^ ~A));
  case 168:
    return ~((~C & ~B) | ~A);
  case 169:
    return ~(~(~C & B) ^ ~(~C ^ ~A));
  case 170:
    return A;
  case 171:
    return ((~C & ~B) | A);
  case 172:
    return (~(~C & B) ^ (~C | ~A));
  case 173:
    return ~(~(~C & B) & (~C ^ ~A));
  case 174:
    return ~(~(~C & B) & ~A);
  case 175:
    return (~C | A);
  case 176:
    return (~(~C | B) | ~(~C | ~A));
  case 177:
    return (~(~C & A) ^ ~(~B | A));
  case 178:
    return ((~(~C ^ ~B) & ~(~C & A)) ^ (~(~C & B) & (~B | ~A)));
  case 179:
    return (~(~C & B) & (~B | A));
  case 180:
    return (~(~C ^ ~B) ^ (~B | ~A));
  case 181:
    return (~(~C | B) | ~(~C ^ ~A));
  case 182:
    return ((~(~C ^ ~B) & ~(~C & A)) ^ (~B | ~A));
  case 183:
    return (~(~C & B) ^ ~(~B | A));
  case 184:
    return ~(~(~C | B) ^ (~B | ~A));
  case 185:
    return (~(~C | B) | ~(~B ^ ~A));
  case 186:
    return (~(~C | B) | A);
  case 187:
    return (~B | A);
  case 188:
    return ~(~(~C ^ ~B) & (~C | ~A));
  case 189:
    return ~(~(~C ^ ~B) & (~C ^ ~A));
  case 190:
    return ~(~(~C ^ ~B) & ~A);
  case 191:
    return ~(~(~C & B) & ~(~B | A));
  case 192:
    return ~(~C | ~B);
  case 193:
    return (~(~C ^ ~B) & ~(~C & A));
  case 194:
    return (~(~C ^ ~B) & ~(~C & ~A));
  case 195:
    return ~(~C ^ ~B);
  case 196:
    return ~((~C | ~B) & (~B | A));
  case 197:
    return (~(~C | B) ^ ~(~C & A));
  case 198:
    return (~(~C ^ ~B) ^ (~C & ~A));
  case 199:
    return (~(~C ^ ~B) | ~(~B | A));
  case 200:
    return ~((~C | ~B) & (~B | ~A));
  case 201:
    return ~(~(~C ^ ~B) ^ ~(~C & A));
  case 202:
    return ~(~(~C | B) ^ (~C & ~A));
  case 203:
    return (~(~C ^ ~B) | (~C & A));
  case 204:
    return B;
  case 205:
    return ~(~(~C | B) | (~B & A));
  case 206:
    return ~(~(~C | B) | (~B & ~A));
  case 207:
    return (~C | B);
  case 208:
    return ~((~C | ~B) & (~C | A));
  case 209:
    return ~(~(~C & B) ^ ~(~B & A));
  case 210:
    return (~(~C ^ ~B) ^ (~B & ~A));
  case 211:
    return (~(~C ^ ~B) | ~(~C | A));
  case 212:
    return ((~(~C ^ ~B) & ~(~C & A)) ^ ((~C | ~B) & ~A));
  case 213:
    return ~((~C | ~B) & A);
  case 214:
    return ~(~(~(~C ^ ~B) & (~C | A)) ^ ~A);
  case 215:
    return (~(~C ^ ~B) | ~A);
  case 216:
    return ~(~(~C | A) ^ (~B | ~A));
  case 217:
    return ~((~C | ~B) & (~B ^ ~A));
  case 218:
    return ~((~C | ~B) & ~(~C ^ ~A));
  case 219:
    return (~(~C ^ ~B) | (~C ^ ~A));
  case 220:
    return ~((~C & ~B) | (~B & A));
  case 221:
    return ~(~B & A);
  case 222:
    return ~(~(~C | B) ^ (~B & ~A));
  case 223:
    return ~(~(~C | B) & (~C | A));
  case 224:
    return ~((~C | ~B) & (~C | ~A));
  case 225:
    return ~(~(~C ^ ~B) ^ ~(~B & A));
  case 226:
    return (~(~C & B) ^ (~B & ~A));
  case 227:
    return (~(~C ^ ~B) | (~B & A));
  case 228:
    return (~(~C & A) ^ (~B & ~A));
  case 229:
    return ~((~C | ~B) & (~C ^ ~A));
  case 230:
    return ~((~C | ~B) & ~(~B ^ ~A));
  case 231:
    return (~(~C ^ ~B) | ~(~C ^ ~A));
  case 232:
    return (((~C & ~B) | ~(~C ^ ~A)) ^ ~(~(~C | B) | ~(~C | ~A)));
  case 233:
    return ((~(~C ^ ~B) & ~(~C & A)) | ~(~(~C ^ ~B) ^ ~A));
  case 234:
    return ~((~C | ~B) & ~A);
  case 235:
    return (~(~C ^ ~B) | A);
  case 236:
    return ~((~C & ~B) | (~B & ~A));
  case 237:
    return (~(~C | B) ^ ~(~B & A));
  case 238:
    return ~(~B & ~A);
  case 239:
    return ~(~(~C | B) & ~(~C | A));
  case 240:
    return C;
  case 241:
    return (~(~C & B) & ~(~C & A));
  case 242:
    return (~(~C & B) & ~(~C & ~A));
  case 243:
    return ~(~C & B);
  case 244:
    return ~((~C & ~B) | (~C & A));
  case 245:
    return ~(~C & A);
  case 246:
    return (~(~C & B) ^ (~C & ~A));
  case 247:
    return (~(~C & B) | ~(~C & A));
  case 248:
    return ~((~C & ~B) | (~C & ~A));
  case 249:
    return ~(~(~C & B) ^ ~(~C & A));
  case 250:
    return ~(~C & ~A);
  case 251:
    return (~(~C & B) | (~C & A));
  case 252:
    return ~(~C & ~B);
  case 253:
    return ~(~(~C & B) & (~C & A));
  case 254:
    return ~(~(~C & B) & (~C & ~A));
  case 255:
    return T(1);
  }
}
#endif
} // namespace detail

namespace ext::oneapi::experimental {

template <uint8_t LUTIndex, typename T>
sycl::detail::builtin_enable_integer_t<T> ternary_bitwise(T A, T B, T C) {
  if constexpr (sycl::detail::is_marray_v<T>) {
    return sycl::detail::builtin_marray_impl(
        [](auto SA, auto SB, auto SC) {
          return ternary_bitwise<LUTIndex>(SA, SB, SC);
        },
        A, B, C);
  } else {
#if defined(__SYCL_DEVICE_ONLY__) && !defined(__NVPTX__) && !defined(__AMDGCN__)
    // TODO: Implement __spirv_BitwiseFunctionINTEL for NVPTX and AMDGCN.
    return __spirv_BitwiseFunctionINTEL(
        sycl::detail::simplify_if_swizzle_t<T>{A},
        sycl::detail::simplify_if_swizzle_t<T>{B},
        sycl::detail::simplify_if_swizzle_t<T>{C},
        static_cast<uint32_t>(LUTIndex));
#else
    return sycl::detail::applyTernaryBitwise<LUTIndex>(
        sycl::detail::simplify_if_swizzle_t<T>{A},
        sycl::detail::simplify_if_swizzle_t<T>{B},
        sycl::detail::simplify_if_swizzle_t<T>{C});
#endif
  }
}
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
