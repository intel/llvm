//==----------------- stream.hpp - SYCL standard header file ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/builtins.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/handler.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

namespace detail {

using FmtFlags = unsigned int;

// Mapping from stream_manipulator to FmtFlags. Each manipulator corresponds
// to the bit in FmtFlags.
static constexpr FmtFlags Dec = 0x0001;
static constexpr FmtFlags Hex = 0x0002;
static constexpr FmtFlags Oct = 0x0004;
static constexpr FmtFlags ShowBase = 0x0008;
static constexpr FmtFlags ShowPos = 0x0010;
static constexpr FmtFlags Fixed = 0x0020;
static constexpr FmtFlags Scientific = 0x0040;

// Bitmask made of the combination of the base flags. Base flags are mutually
// exclusive, this mask is used to clean base field before setting the new
// base flag.
static constexpr FmtFlags BaseField = Dec | Hex | Oct;

// Bitmask made of the combination of the floating point value format flags.
// Thease flags are mutually exclusive, this mask is used to clean float field
// before setting the new float flag.
static constexpr FmtFlags FloatField = Scientific | Fixed;

constexpr size_t MAX_FLOATING_POINT_DIGITS = 24;
constexpr size_t MAX_INTEGRAL_DIGITS = 23;
constexpr const char *VEC_ELEMENT_DELIMITER = ", ";
constexpr char VEC_OPEN_BRACE = '{';
constexpr char VEC_CLOSE_BRACE = '}';

constexpr size_t MAX_DIMENSIONS = 3;

// Space for integrals (up to 3), comma and space between the
// integrals and enclosing braces.
constexpr size_t MAX_ARRAY_SIZE =
    MAX_INTEGRAL_DIGITS * MAX_DIMENSIONS + 2 * (MAX_DIMENSIONS - 1) + 2;

// First 2 bytes in each work item's flush buffer are reserved for saving
// statement offset.
constexpr unsigned FLUSH_BUF_OFFSET_SIZE = 2;

template <class F, class T = void>
using EnableIfFP =
    typename detail::enable_if_t<std::is_same<F, float>::value ||
                                     std::is_same<F, double>::value ||
                                     std::is_same<F, half>::value,
                                 T>;

using GlobalBufAccessorT = accessor<char, 1, cl::sycl::access::mode::read_write,
                                    cl::sycl::access::target::global_buffer,
                                    cl::sycl::access::placeholder::false_t>;

constexpr static access::address_space GlobalBufAS =
    TargetToAS<cl::sycl::access::target::global_buffer>::AS;
using GlobalBufPtrType =
    typename detail::DecoratedType<char, GlobalBufAS>::type *;
constexpr static int GlobalBufDim = 1;

using GlobalOffsetAccessorT =
    accessor<unsigned, 1, cl::sycl::access::mode::atomic,
             cl::sycl::access::target::global_buffer,
             cl::sycl::access::placeholder::false_t>;

constexpr static access::address_space GlobalOffsetAS =
    TargetToAS<cl::sycl::access::target::global_buffer>::AS;
using GlobalOffsetPtrType =
    typename detail::DecoratedType<unsigned, GlobalBufAS>::type *;
constexpr static int GlobalOffsetDim = 1;

// Read first 2 bytes of flush buffer to get buffer offset.
// TODO: Should be optimized to the following:
//   return *reinterpret_cast<uint16_t *>(&GlobalFlushBuf[WIOffset]);
// when an issue with device code compilation using this optimization is fixed.
inline unsigned GetFlushBufOffset(const GlobalBufAccessorT &GlobalFlushBuf,
                                  unsigned WIOffset) {
  return ((static_cast<unsigned>(static_cast<uint8_t>(GlobalFlushBuf[WIOffset]))
           << 8) +
          static_cast<uint8_t>(GlobalFlushBuf[WIOffset + 1]));
}

// Write flush buffer's offset into first 2 bytes of that buffer.
// TODO: Should be optimized to the following:
//   *reinterpret_cast<uint16_t *>(&GlobalFlushBuf[WIOffset]) =
//       static_cast<uint16_t>(Offset);
// when an issue with device code compilation using this optimization is fixed.
inline void SetFlushBufOffset(GlobalBufAccessorT &GlobalFlushBuf,
                              unsigned WIOffset, unsigned Offset) {
  GlobalFlushBuf[WIOffset] = static_cast<char>((Offset >> 8) & 0xff);
  GlobalFlushBuf[WIOffset + 1] = static_cast<char>(Offset & 0xff);
}

inline void write(GlobalBufAccessorT &GlobalFlushBuf, size_t FlushBufferSize,
                  unsigned WIOffset, const char *Str, unsigned Len,
                  unsigned Padding = 0) {
  unsigned Offset =
      GetFlushBufOffset(GlobalFlushBuf, WIOffset) + FLUSH_BUF_OFFSET_SIZE;

  if ((Offset + Len + Padding > FlushBufferSize) ||
      (WIOffset + Offset + Len + Padding > GlobalFlushBuf.size()))
    // TODO: flush here
    return;

  // Write padding
  for (size_t I = 0; I < Padding; ++I, ++Offset)
    GlobalFlushBuf[WIOffset + Offset] = ' ';

  for (size_t I = 0; I < Len; ++I, ++Offset) {
    GlobalFlushBuf[WIOffset + Offset] = Str[I];
  }

  SetFlushBufOffset(GlobalFlushBuf, WIOffset, Offset - FLUSH_BUF_OFFSET_SIZE);
}

inline void reverseBuf(char *Buf, unsigned Len) {
  int I = Len - 1;
  int J = 0;
  while (I > J) {
    int Temp = Buf[I];
    Buf[I] = Buf[J];
    Buf[J] = Temp;
    I--;
    J++;
  }
}

template <typename T>
inline typename std::make_unsigned<T>::type getAbsVal(const T Val,
                                                      const int Base) {
  return ((Base == 10) && (Val < 0)) ? -Val : Val;
}

inline char digitToChar(const int Digit) {
  if (Digit < 10) {
    return '0' + Digit;
  } else {
    return 'a' + Digit - 10;
  }
}

template <typename T>
inline typename detail::enable_if_t<std::is_integral<T>::value, unsigned>
integralToBase(T Val, int Base, char *Digits) {
  unsigned NumDigits = 0;

  do {
    Digits[NumDigits++] = digitToChar(Val % Base);
    Val /= Base;
  } while (Val);

  return NumDigits;
}

// Returns number of symbols written to the buffer
template <typename T>
inline typename detail::enable_if_t<std::is_integral<T>::value, unsigned>
ScalarToStr(const T &Val, char *Buf, unsigned Flags, int, int Precision = -1) {
  (void)Precision;
  int Base = 10;

  // append base manipulator
  switch (Flags & BaseField) {
  case Dec:
    Base = 10;
    break;
  case Hex:
    Base = 16;
    break;
  case Oct:
    Base = 8;
    break;
  default:
    // default value is 10
    break;
  }

  unsigned Offset = 0;

  // write '+' to the stream if the base is 10 and the value is non-negative
  // or write '-' to stream if base is 10 and the value is negative
  if (Base == 10) {
    if ((Flags & ShowPos) && Val >= 0)
      Buf[Offset++] = '+';
    else if (Val < 0)
      Buf[Offset++] = '-';
  }

  // write 0 or 0x to the stream if base is not 10 and the manipulator is set
  if (Base != 10 && (Flags & ShowBase)) {
    Buf[Offset++] = '0';
    if (Base == 16)
      Buf[Offset++] = 'x';
  }

  auto AbsVal = getAbsVal(Val, Base);

  const unsigned NumBuf = integralToBase(AbsVal, Base, Buf + Offset);

  reverseBuf(Buf + Offset, NumBuf);
  return Offset + NumBuf;
}

inline unsigned append(char *Dst, const char *Src) {
  unsigned Len = 0;
  for (; Src[Len] != '\0'; ++Len)
    ;

  for (unsigned I = 0; I < Len; ++I)
    Dst[I] = Src[I];
  return Len;
}

template <typename T>
inline typename detail::enable_if_t<
    std::is_same<T, float>::value || std::is_same<T, double>::value, unsigned>
checkForInfNan(char *Buf, T Val) {
  if (isnan(Val))
    return append(Buf, "nan");
  if (isinf(Val)) {
    if (signbit(Val))
      return append(Buf, "-inf");
    return append(Buf, "inf");
  }
  return 0;
}

template <typename T>
inline typename detail::enable_if_t<std::is_same<T, half>::value, unsigned>
checkForInfNan(char *Buf, T Val) {
  if (Val != Val)
    return append(Buf, "nan");

  // Extract the sign from the bits
  const uint16_t Sign = reinterpret_cast<uint16_t &>(Val) & 0x8000;
  // Extract the exponent from the bits
  const uint16_t Exp16 = (reinterpret_cast<uint16_t &>(Val) & 0x7c00) >> 10;

  if (Exp16 == 0x1f) {
    if (Sign)
      return append(Buf, "-inf");
    return append(Buf, "inf");
  }
  return 0;
}

template <typename T>
EnableIfFP<T, unsigned> floatingPointToDecStr(T AbsVal, char *Digits,
                                              int Precision, bool IsSci) {
  int Exp = 0;

  // For the case that the value is larger than 10.0
  while (AbsVal >= 10.0) {
    ++Exp;
    AbsVal /= 10.0;
  }
  // For the case that the value is less than 1.0
  while (AbsVal > 0.0 && AbsVal < 1.0) {
    --Exp;
    AbsVal *= 10.0;
  }

  auto IntegralPart = static_cast<int>(AbsVal);
  auto FractionPart = AbsVal - IntegralPart;

  int FractionDigits[MAX_FLOATING_POINT_DIGITS] = {0};

  // Exponent
  int P = Precision > 0 ? Precision : 4;
  size_t FractionLength = Exp + P;

  // After normalization integral part contains 1 symbol, also there could be
  // '.', 'e', sign of the exponent and sign of the number, overall 5 symbols.
  // So, clamp fraction length if required according to maximum size of the
  // buffer for floating point number.
  if (FractionLength > MAX_FLOATING_POINT_DIGITS - 5)
    FractionLength = MAX_FLOATING_POINT_DIGITS - 5;

  for (unsigned I = 0; I < FractionLength; ++I) {
    FractionPart *= 10.0;
    FractionDigits[I] = static_cast<int>(FractionPart);
    FractionPart -= static_cast<int>(FractionPart);
  }

  int Carry = FractionPart > static_cast<T>(0.5) ? 1 : 0;

  // Propagate the Carry
  for (int I = FractionLength - 1; I >= 0 && Carry; --I) {
    auto Digit = FractionDigits[I] + Carry;
    FractionDigits[I] = Digit % 10;
    Carry = Digit / 10;
  }

  // Carry from the fraction part is propagated to integral part
  IntegralPart += Carry;
  if (IntegralPart == 10) {
    IntegralPart = 1;
    ++Exp;
  }

  unsigned Offset = 0;

  // Assemble the final string correspondingly
  if (IsSci) { // scientific mode
    // Append the integral part
    Digits[Offset++] = digitToChar(IntegralPart);
    Digits[Offset++] = '.';

    // Append all fraction
    for (unsigned I = 0; I < FractionLength; ++I)
      Digits[Offset++] = digitToChar(FractionDigits[I]);

    // Exponent part
    Digits[Offset++] = 'e';
    Digits[Offset++] = Exp >= 0 ? '+' : '-';
    Digits[Offset++] = digitToChar(abs(Exp) / 10);
    Digits[Offset++] = digitToChar(abs(Exp) % 10);
  } else { // normal mode
    if (Exp < 0) {
      Digits[Offset++] = '0';
      Digits[Offset++] = '.';
      while (++Exp)
        Digits[Offset++] = '0';

      // Append the integral part
      Digits[Offset++] = digitToChar(IntegralPart);

      // Append all fraction
      for (unsigned I = 0; I < FractionLength; ++I)
        Digits[Offset++] = digitToChar(FractionDigits[I]);
    } else {
      // Append the integral part
      Digits[Offset++] = digitToChar(IntegralPart);
      unsigned I = 0;
      // Append the integral part first
      for (; I < FractionLength && Exp--; ++I)
        Digits[Offset++] = digitToChar(FractionDigits[I]);

      // Put the dot
      Digits[Offset++] = '.';

      // Append the rest of fraction part, or the real fraction part
      for (; I < FractionLength; ++I)
        Digits[Offset++] = digitToChar(FractionDigits[I]);
    }
    // The normal mode requires no tailing zero digit, then we need to first
    // find the first non-zero digit
    while (Digits[Offset - 1] == '0')
      Offset--;

    // If dot is the last digit, it should be stripped off as well
    if (Digits[Offset - 1] == '.')
      Offset--;
  }
  return Offset;
}

// Returns number of symbols written to the buffer
template <typename T>
inline EnableIfFP<T, unsigned>
ScalarToStr(const T &Val, char *Buf, unsigned Flags, int, int Precision = -1) {
  unsigned Offset = checkForInfNan(Buf, Val);
  if (Offset)
    return Offset;

  T Neg = -Val;
  auto AbsVal = Val < 0 ? Neg : Val;

  if (Val < 0) {
    Buf[Offset++] = '-';
  } else if (Flags & ShowPos) {
    Buf[Offset++] = '+';
  }

  bool IsSci = false;
  if (Flags & detail::Scientific)
    IsSci = true;

  // TODO: manipulators for floating-point output - hexfloat, fixed
  Offset += floatingPointToDecStr(AbsVal, Buf + Offset, Precision, IsSci);

  return Offset;
}

template <typename T>
inline typename detail::enable_if_t<std::is_integral<T>::value>
writeIntegral(GlobalBufAccessorT &GlobalFlushBuf, size_t FlushBufferSize,
              unsigned WIOffset, unsigned Flags, int Width, const T &Val) {
  char Digits[MAX_INTEGRAL_DIGITS] = {0};
  unsigned Len = ScalarToStr(Val, Digits, Flags, Width);
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, Digits, Len,
        (Width > 0 && static_cast<unsigned>(Width) > Len)
            ? static_cast<unsigned>(Width) - Len
            : 0);
}

template <typename T>
inline EnableIfFP<T>
writeFloatingPoint(GlobalBufAccessorT &GlobalFlushBuf, size_t FlushBufferSize,
                   unsigned WIOffset, unsigned Flags, int Width, int Precision,
                   const T &Val) {
  char Digits[MAX_FLOATING_POINT_DIGITS] = {0};
  unsigned Len = ScalarToStr(Val, Digits, Flags, Width, Precision);
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, Digits, Len,
        (Width > 0 && static_cast<unsigned>(Width) > Len)
            ? static_cast<unsigned>(Width) - Len
            : 0);
}

// Helper method to update offset in the global buffer atomically according to
// the provided size of the data in the flush buffer. Return true if offset is
// updated and false in case of overflow.
inline bool updateOffset(GlobalOffsetAccessorT &GlobalOffset,
                         GlobalBufAccessorT &GlobalBuf, unsigned Size,
                         unsigned &Cur) {
  unsigned New;
  Cur = GlobalOffset[0].load();
  do {
    if (GlobalBuf.get_range().size() - Cur < Size)
      // Overflow
      return false;
    New = Cur + Size;
  } while (!GlobalOffset[0].compare_exchange_strong(Cur, New));
  return true;
}

inline void flushBuffer(GlobalOffsetAccessorT &GlobalOffset,
                        GlobalBufAccessorT &GlobalBuf,
                        GlobalBufAccessorT &GlobalFlushBuf, unsigned WIOffset) {
  unsigned Offset = GetFlushBufOffset(GlobalFlushBuf, WIOffset);
  if (Offset == 0)
    return;

  unsigned Cur = 0;
  if (!updateOffset(GlobalOffset, GlobalBuf, Offset, Cur))
    return;

  unsigned StmtOffset = WIOffset + FLUSH_BUF_OFFSET_SIZE;
  for (unsigned I = StmtOffset; I < StmtOffset + Offset; I++) {
    GlobalBuf[Cur++] = GlobalFlushBuf[I];
  }
  // Reset the offset in the flush buffer
  SetFlushBufOffset(GlobalFlushBuf, WIOffset, 0);
}

template <typename T, int VecLength>
typename detail::enable_if_t<(VecLength == 1), unsigned>
VecToStr(const vec<T, VecLength> &Vec, char *VecStr, unsigned Flags, int Width,
         int Precision) {
  return ScalarToStr(static_cast<T>(Vec.x()), VecStr, Flags, Width, Precision);
}

template <typename T, int VecLength>
typename detail::enable_if_t<(VecLength == 2 || VecLength == 4 ||
                              VecLength == 8 || VecLength == 16),
                             unsigned>
VecToStr(const vec<T, VecLength> &Vec, char *VecStr, unsigned Flags, int Width,
         int Precision) {
  unsigned Len =
      VecToStr<T, VecLength / 2>(Vec.lo(), VecStr, Flags, Width, Precision);
  Len += append(VecStr + Len, VEC_ELEMENT_DELIMITER);
  Len += VecToStr<T, VecLength / 2>(Vec.hi(), VecStr + Len, Flags, Width,
                                    Precision);
  return Len;
}

template <typename T, int VecLength>
typename detail::enable_if_t<(VecLength == 3), unsigned>
VecToStr(const vec<T, VecLength> &Vec, char *VecStr, unsigned Flags, int Width,
         int Precision) {
  unsigned Len = VecToStr<T, 2>(Vec.lo(), VecStr, Flags, Width, Precision);
  Len += append(VecStr + Len, VEC_ELEMENT_DELIMITER);
  Len += VecToStr<T, 1>(Vec.z(), VecStr + Len, Flags, Width, Precision);
  return Len;
}

template <typename T, int VecLength>
inline void writeVec(GlobalBufAccessorT &GlobalFlushBuf, size_t FlushBufferSize,
                     unsigned WIOffset, unsigned Flags, int Width,
                     int Precision, const vec<T, VecLength> &Vec) {
  // Reserve space for vector elements and delimiters
  constexpr size_t MAX_VEC_SIZE =
      MAX_FLOATING_POINT_DIGITS * VecLength + (VecLength - 1) * 2;
  char VecStr[MAX_VEC_SIZE] = {0};
  unsigned Len = VecToStr<T, VecLength>(Vec, VecStr, Flags, Width, Precision);
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, VecStr, Len,
        (Width > 0 && Width > Len) ? Width - Len : 0);
}

template <int ArrayLength>
inline unsigned ArrayToStr(char *Buf, const array<ArrayLength> &Arr) {
  unsigned Len = 0;
  Buf[Len++] = VEC_OPEN_BRACE;

  for (int I = 0; I < ArrayLength; ++I) {
    Len += ScalarToStr(Arr[I], Buf + Len, 0 /* No flags */, -1, -1);
    if (I != ArrayLength - 1)
      Len += append(Buf + Len, VEC_ELEMENT_DELIMITER);
  }

  Buf[Len++] = VEC_CLOSE_BRACE;

  return Len;
}

template <int ArrayLength>
inline void writeArray(GlobalBufAccessorT &GlobalFlushBuf,
                       size_t FlushBufferSize, unsigned WIOffset,
                       const array<ArrayLength> &Arr) {
  char Buf[MAX_ARRAY_SIZE];
  unsigned Len = ArrayToStr(Buf, Arr);
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, Buf, Len);
}

template <int Dimensions>
inline void writeItem(GlobalBufAccessorT &GlobalFlushBuf,
                      size_t FlushBufferSize, unsigned WIOffset,
                      const item<Dimensions> &Item) {
  // Reserve space for 3 arrays and additional place (40 symbols) for printing
  // the text
  char Buf[3 * MAX_ARRAY_SIZE + 40];
  unsigned Len = 0;
  Len += append(Buf, "item(");
  Len += append(Buf + Len, "range: ");
  Len += ArrayToStr(Buf + Len, Item.get_range());
  Len += append(Buf + Len, ", id: ");
  Len += ArrayToStr(Buf + Len, Item.get_id());
  Len += append(Buf + Len, ", offset: ");
  Len += ArrayToStr(Buf + Len, Item.get_offset());
  Buf[Len++] = ')';
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, Buf, Len);
}

template <int Dimensions>
inline void writeNDRange(GlobalBufAccessorT &GlobalFlushBuf,
                         size_t FlushBufferSize, unsigned WIOffset,
                         const nd_range<Dimensions> &ND_Range) {
  // Reserve space for 3 arrays and additional place (50 symbols) for printing
  // the text
  char Buf[3 * MAX_ARRAY_SIZE + 50];
  unsigned Len = 0;
  Len += append(Buf, "nd_range(");
  Len += append(Buf + Len, "global_range: ");
  Len += ArrayToStr(Buf + Len, ND_Range.get_global_range());
  Len += append(Buf + Len, ", local_range: ");
  Len += ArrayToStr(Buf + Len, ND_Range.get_local_range());
  Len += append(Buf + Len, ", offset: ");
  Len += ArrayToStr(Buf + Len, ND_Range.get_offset());
  Buf[Len++] = ')';
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, Buf, Len);
}

template <int Dimensions>
inline void writeNDItem(GlobalBufAccessorT &GlobalFlushBuf,
                        size_t FlushBufferSize, unsigned WIOffset,
                        const nd_item<Dimensions> &ND_Item) {
  // Reserve space for 2 arrays and additional place (40 symbols) for printing
  // the text
  char Buf[2 * MAX_ARRAY_SIZE + 40];
  unsigned Len = 0;
  Len += append(Buf, "nd_item(");
  Len += append(Buf + Len, "global_id: ");
  Len += ArrayToStr(Buf + Len, ND_Item.get_global_id());
  Len += append(Buf + Len, ", local_id: ");
  Len += ArrayToStr(Buf + Len, ND_Item.get_local_id());
  Buf[Len++] = ')';
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, Buf, Len);
}

template <int Dimensions>
inline void writeGroup(GlobalBufAccessorT &GlobalFlushBuf,
                       size_t FlushBufferSize, unsigned WIOffset,
                       const group<Dimensions> &Group) {
  // Reserve space for 4 arrays and additional place (60 symbols) for printing
  // the text
  char Buf[4 * MAX_ARRAY_SIZE + 60];
  unsigned Len = 0;
  Len += append(Buf, "group(");
  Len += append(Buf + Len, "id: ");
  Len += ArrayToStr(Buf + Len, Group.get_id());
  Len += append(Buf + Len, ", global_range: ");
  Len += ArrayToStr(Buf + Len, Group.get_global_range());
  Len += append(Buf + Len, ", local_range: ");
  Len += ArrayToStr(Buf + Len, Group.get_local_range());
  Len += append(Buf + Len, ", group_range: ");
  Len += ArrayToStr(Buf + Len, Group.get_group_range());
  Buf[Len++] = ')';
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, Buf, Len);
}

// Space for 2 arrays and additional place (20 symbols) for printing
// the text
constexpr size_t MAX_ITEM_SIZE = 2 * MAX_ARRAY_SIZE + 20;

template <int Dimensions>
inline unsigned ItemToStr(char *Buf, const item<Dimensions, false> &Item) {
  unsigned Len = 0;
  Len += append(Buf, "item(");
  for (int I = 0; I < 2; ++I) {
    Len += append(Buf + Len, I == 0 ? "range: " : ", id: ");
    Len += ArrayToStr(Buf + Len, I == 0 ? Item.get_range() : Item.get_id());
  }
  Buf[Len++] = ')';
  return Len;
}

template <int Dimensions>
inline void writeHItem(GlobalBufAccessorT &GlobalFlushBuf,
                       size_t FlushBufferSize, unsigned WIOffset,
                       const h_item<Dimensions> &HItem) {
  // Reserve space for 3 items and additional place (60 symbols) for printing
  // the text
  char Buf[3 * MAX_ITEM_SIZE + 60];
  unsigned Len = 0;
  Len += append(Buf, "h_item(");
  for (int I = 0; I < 3; ++I) {
    Len += append(Buf + Len, I == 0 ? "\n  global "
                                    : I == 1 ? "\n  logical local "
                                             : "\n  physical local ");
    Len += ItemToStr(Buf + Len, I == 0 ? HItem.get_global()
                                       : I == 1 ? HItem.get_logical_local()
                                                : HItem.get_physical_local());
  }
  Len += append(Buf + Len, "\n)");
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, Buf, Len);
}

template <typename> struct IsSwizzleOp : std::false_type {};

template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
struct IsSwizzleOp<cl::sycl::detail::SwizzleOp<
    VecT, OperationLeftT, OperationRightT, OperationCurrentT, Indexes...>>
    : std::true_type {
  using T = typename VecT::element_type;
  using Type = typename cl::sycl::vec<T, (sizeof...(Indexes))>;
};

template <typename T>
using EnableIfSwizzleVec =
    typename detail::enable_if_t<IsSwizzleOp<T>::value,
                                 typename IsSwizzleOp<T>::Type>;

} // namespace detail

enum class stream_manipulator {
  dec = 0,
  hex = 1,
  oct = 2,
  noshowbase = 3,
  showbase = 4,
  noshowpos = 5,
  showpos = 6,
  endl = 7,
  flush = 8,
  fixed = 9,
  scientific = 10,
  hexfloat = 11,
  defaultfloat = 12
};

constexpr stream_manipulator dec = stream_manipulator::dec;

constexpr stream_manipulator hex = stream_manipulator::hex;

constexpr stream_manipulator oct = stream_manipulator::oct;

constexpr stream_manipulator noshowbase = stream_manipulator::noshowbase;

constexpr stream_manipulator showbase = stream_manipulator::showbase;

constexpr stream_manipulator noshowpos = stream_manipulator::noshowpos;

constexpr stream_manipulator showpos = stream_manipulator::showpos;

constexpr stream_manipulator endl = stream_manipulator::endl;

constexpr stream_manipulator flush = stream_manipulator::flush;

constexpr stream_manipulator fixed = stream_manipulator::fixed;

constexpr stream_manipulator scientific = stream_manipulator::scientific;

constexpr stream_manipulator hexfloat = stream_manipulator::hexfloat;

constexpr stream_manipulator defaultfloat = stream_manipulator::defaultfloat;

class stream;

class __precision_manipulator__ {
  int Precision_;

public:
  __precision_manipulator__(int Precision) : Precision_(Precision) {}

  int precision() const { return Precision_; }

  friend const stream &operator<<(const stream &,
                                  const __precision_manipulator__ &);
};

class __width_manipulator__ {
  int Width_;

public:
  __width_manipulator__(int Width) : Width_(Width) {}

  int width() const { return Width_; }

  friend const stream &operator<<(const stream &,
                                  const __width_manipulator__ &);
};

inline __precision_manipulator__ setprecision(int Precision) {
  return __precision_manipulator__(Precision);
}

inline __width_manipulator__ setw(int Width) {
  return __width_manipulator__(Width);
}

/// A buffered output stream that allows outputting the values of built-in,
/// vector and SYCL types to the console.
///
/// \ingroup sycl_api
class __SYCL_EXPORT stream {
public:
#ifdef __SYCL_DEVICE_ONLY__
  // Default constructor for objects later initialized with __init member.
  stream() = default;
#endif

  // Throws exception in case of invalid input parameters
  stream(size_t BufferSize, size_t MaxStatementSize, handler &CGH);

  size_t get_size() const;

  size_t get_max_statement_size() const;

  size_t get_precision() const { return Precision; }

  size_t get_width() const { return Width; }

  stream_manipulator get_stream_mode() const { return Manipulator; }

  bool operator==(const stream &RHS) const;

  bool operator!=(const stream &LHS) const;

private:
#ifdef __SYCL_DEVICE_ONLY__
  char padding[sizeof(std::shared_ptr<detail::stream_impl>)];
#else
  std::shared_ptr<detail::stream_impl> impl;
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
#endif

  // Accessor to the global stream buffer. Global buffer contains all output
  // from the kernel.
  mutable detail::GlobalBufAccessorT GlobalBuf;

  // Atomic accessor to the global offset variable. It represents an offset in
  // the global stream buffer. Since work items will flush data to global buffer
  // in parallel we need atomic access to this offset.
  mutable detail::GlobalOffsetAccessorT GlobalOffset;

  // Accessor to the flush buffer. Each work item writes its
  // output to a designated section of the flush buffer.
  mutable detail::GlobalBufAccessorT GlobalFlushBuf;

  // Offset of the WI's flush buffer in the pool.
  mutable unsigned WIOffset = 0;

  // Offset in the flush buffer
  // TODO: This field is not used anymore.
  // To be removed when API/ABI changes are allowed.
  mutable unsigned Offset = 0;

  mutable size_t FlushBufferSize;

  // Fields and methods to work with manipulators
  mutable stream_manipulator Manipulator = defaultfloat;

  // Type used for format flags
  using FmtFlags = unsigned int;

  mutable int Precision = -1;
  mutable int Width = -1;
  mutable FmtFlags Flags = 0x0;

  void set_flag(FmtFlags FormatFlag) const { Flags |= FormatFlag; }

  void unset_flag(FmtFlags FormatFlag) const { Flags &= ~FormatFlag; }

  FmtFlags get_flags() const { return Flags; }

  // This method is used to set the flag for base and float manipulators. These
  // flags are mutually exclusive and base/float field needs to be cleared
  // before the setting new flag.
  void set_flag(FmtFlags FormatFlag, FmtFlags Mask) const {
    unset_flag(Mask);
    Flags |= FormatFlag & Mask;
  }

  // Set the flags which correspond to the input stream manipulator.
  void set_manipulator(const stream_manipulator SM) const {
    switch (SM) {
    case stream_manipulator::dec:
      set_flag(detail::Dec, detail::BaseField);
      break;
    case stream_manipulator::hex:
      set_flag(detail::Hex, detail::BaseField);
      break;
    case stream_manipulator::oct:
      set_flag(detail::Oct, detail::BaseField);
      break;
    case stream_manipulator::noshowbase:
      unset_flag(detail::ShowBase);
      break;
    case stream_manipulator::showbase:
      set_flag(detail::ShowBase);
      break;
    case stream_manipulator::noshowpos:
      unset_flag(detail::ShowPos);
      break;
    case stream_manipulator::showpos:
      set_flag(detail::ShowPos);
      break;
    case stream_manipulator::fixed:
      set_flag(detail::Fixed, detail::FloatField);
      break;
    case stream_manipulator::scientific:
      set_flag(detail::Scientific, detail::FloatField);
      break;
    case stream_manipulator::hexfloat:
      set_flag(detail::Fixed | detail::Scientific, detail::FloatField);
      break;
    case stream_manipulator::defaultfloat:
      unset_flag(detail::FloatField);
      break;
    default:
      // Unknown manipulator
      break;
    }
  }

#ifdef __SYCL_DEVICE_ONLY__
  void __init(detail::GlobalBufPtrType GlobalBufPtr,
              range<detail::GlobalBufDim> GlobalBufAccRange,
              range<detail::GlobalBufDim> GlobalBufMemRange,
              id<detail::GlobalBufDim> GlobalBufId,
              detail::GlobalOffsetPtrType GlobalOffsetPtr,
              range<detail::GlobalOffsetDim> GlobalOffsetAccRange,
              range<detail::GlobalOffsetDim> GlobalOffsetMemRange,
              id<detail::GlobalOffsetDim> GlobalOffsetId,
              detail::GlobalBufPtrType GlobalFlushPtr,
              range<detail::GlobalBufDim> GlobalFlushAccRange,
              range<detail::GlobalBufDim> GlobalFlushMemRange,
              id<detail::GlobalBufDim> GlobalFlushId, size_t _FlushBufferSize) {
    GlobalBuf.__init(GlobalBufPtr, GlobalBufAccRange, GlobalBufMemRange,
                     GlobalBufId);
    GlobalOffset.__init(GlobalOffsetPtr, GlobalOffsetAccRange,
                        GlobalOffsetMemRange, GlobalOffsetId);
    GlobalFlushBuf.__init(GlobalFlushPtr, GlobalFlushAccRange,
                          GlobalFlushMemRange, GlobalFlushId);
    FlushBufferSize = _FlushBufferSize;
    // Calculate offset in the flush buffer for each work item in the global
    // work space. We need to avoid calling intrinsics to get global id because
    // when stream is used in a single_task kernel this could cause some
    // overhead on FPGA target. That is why use global atomic variable to
    // calculate offsets.
    WIOffset = GlobalOffset[1].fetch_add(FlushBufferSize);

    // Initialize flush subbuffer's offset for each work item on device.
    // Initialization on host device is performed via submition of additional
    // host task.
    SetFlushBufOffset(GlobalFlushBuf, WIOffset, 0);
  }

  void __finalize() {
    // Flush data to global buffer if flush buffer is not empty. This could be
    // necessary if user hasn't yet flushed data on its own and kernel execution
    // is finished
    // NOTE: A call to this function will be generated by compiler
    // NOTE: In the current implementation user should explicitly flush data on
    // the host device. Data is not flushed automatically after kernel execution
    // because of the missing feature in scheduler.
    flushBuffer(GlobalOffset, GlobalBuf, GlobalFlushBuf, WIOffset);
  }
#endif

  friend class handler;

  friend const stream &operator<<(const stream &, const char);
  friend const stream &operator<<(const stream &, const char *);
  template <typename ValueType>
  friend typename detail::enable_if_t<std::is_integral<ValueType>::value,
                                      const stream &>
  operator<<(const stream &, const ValueType &);
  friend const stream &operator<<(const stream &, const float &);
  friend const stream &operator<<(const stream &, const double &);
  friend const stream &operator<<(const stream &, const half &);

  friend const stream &operator<<(const stream &, const stream_manipulator);

  friend const stream &operator<<(const stream &Out,
                                  const __precision_manipulator__ &RHS);

  friend const stream &operator<<(const stream &Out,
                                  const __width_manipulator__ &RHS);
  template <typename T, int Dimensions>
  friend const stream &operator<<(const stream &Out,
                                  const vec<T, Dimensions> &RHS);
  template <typename T>
  friend const stream &operator<<(const stream &Out, const T *RHS);
  template <int Dimensions>
  friend const stream &operator<<(const stream &Out, const id<Dimensions> &RHS);

  template <int Dimensions>
  friend const stream &operator<<(const stream &Out,
                                  const range<Dimensions> &RHS);

  template <int Dimensions>
  friend const stream &operator<<(const stream &Out,
                                  const item<Dimensions> &RHS);

  template <int Dimensions>
  friend const stream &operator<<(const stream &Out,
                                  const nd_range<Dimensions> &RHS);

  template <int Dimensions>
  friend const stream &operator<<(const stream &Out,
                                  const nd_item<Dimensions> &RHS);

  template <int Dimensions>
  friend const stream &operator<<(const stream &Out,
                                  const group<Dimensions> &RHS);

  template <int Dimensions>
  friend const stream &operator<<(const stream &Out,
                                  const h_item<Dimensions> &RHS);
};

// Character
inline const stream &operator<<(const stream &Out, const char C) {
  detail::write(Out.GlobalFlushBuf, Out.FlushBufferSize, Out.WIOffset, &C, 1);
  return Out;
}

// String
inline const stream &operator<<(const stream &Out, const char *Str) {
  unsigned Len = 0;
  for (; Str[Len] != '\0'; Len++)
    ;

  detail::write(Out.GlobalFlushBuf, Out.FlushBufferSize, Out.WIOffset, Str,
                Len);
  return Out;
}

// Boolean
inline const stream &operator<<(const stream &Out, const bool &RHS) {
  Out << (RHS ? "true" : "false");
  return Out;
}

// Integral
template <typename ValueType>
inline typename detail::enable_if_t<std::is_integral<ValueType>::value,
                                    const stream &>
operator<<(const stream &Out, const ValueType &RHS) {
  detail::writeIntegral(Out.GlobalFlushBuf, Out.FlushBufferSize, Out.WIOffset,
                        Out.get_flags(), Out.get_width(), RHS);
  return Out;
}

// Floating points

inline const stream &operator<<(const stream &Out, const float &RHS) {
  detail::writeFloatingPoint<float>(Out.GlobalFlushBuf, Out.FlushBufferSize,
                                    Out.WIOffset, Out.get_flags(),
                                    Out.get_width(), Out.get_precision(), RHS);
  return Out;
}

inline const stream &operator<<(const stream &Out, const double &RHS) {
  detail::writeFloatingPoint<double>(Out.GlobalFlushBuf, Out.FlushBufferSize,
                                     Out.WIOffset, Out.get_flags(),
                                     Out.get_width(), Out.get_precision(), RHS);
  return Out;
}

inline const stream &operator<<(const stream &Out, const half &RHS) {
  detail::writeFloatingPoint<half>(Out.GlobalFlushBuf, Out.FlushBufferSize,
                                   Out.WIOffset, Out.get_flags(),
                                   Out.get_width(), Out.get_precision(), RHS);
  return Out;
}

// Pointer

template <typename ElementType, access::address_space Space>
inline const stream &operator<<(const stream &Out,
                                const multi_ptr<ElementType, Space> &RHS) {
  Out << RHS.get();
  return Out;
}

template <typename T>
const stream &operator<<(const stream &Out, const T *RHS) {
  detail::FmtFlags Flags = Out.get_flags();
  Flags &= ~detail::BaseField;
  Flags |= detail::Hex | detail::ShowBase;
  detail::writeIntegral(Out.GlobalFlushBuf, Out.FlushBufferSize, Out.WIOffset,
                        Flags, Out.get_width(), reinterpret_cast<size_t>(RHS));
  return Out;
}

// Manipulators

inline const stream &operator<<(const stream &Out,
                                const __precision_manipulator__ &RHS) {
  Out.Precision = RHS.precision();
  return Out;
}

inline const stream &operator<<(const stream &Out,
                                const __width_manipulator__ &RHS) {
  Out.Width = RHS.width();
  return Out;
}

inline const stream &operator<<(const stream &Out,
                                const stream_manipulator RHS) {
  switch (RHS) {
  case stream_manipulator::endl:
    Out << '\n';
    flushBuffer(Out.GlobalOffset, Out.GlobalBuf, Out.GlobalFlushBuf,
                Out.WIOffset);
    break;
  case stream_manipulator::flush:
    flushBuffer(Out.GlobalOffset, Out.GlobalBuf, Out.GlobalFlushBuf,
                Out.WIOffset);
    break;
  default:
    Out.set_manipulator(RHS);
    break;
  }
  return Out;
}

// Vec

template <typename T, int VectorLength>
const stream &operator<<(const stream &Out, const vec<T, VectorLength> &RHS) {
  detail::writeVec<T, VectorLength>(Out.GlobalFlushBuf, Out.FlushBufferSize,
                                    Out.WIOffset, Out.get_flags(),
                                    Out.get_width(), Out.get_precision(), RHS);
  return Out;
}

// SYCL types

template <int Dimensions>
inline const stream &operator<<(const stream &Out, const id<Dimensions> &RHS) {
  detail::writeArray<Dimensions>(Out.GlobalFlushBuf, Out.FlushBufferSize,
                                 Out.WIOffset, RHS);
  return Out;
}

template <int Dimensions>
inline const stream &operator<<(const stream &Out,
                                const range<Dimensions> &RHS) {
  detail::writeArray<Dimensions>(Out.GlobalFlushBuf, Out.FlushBufferSize,
                                 Out.WIOffset, RHS);
  return Out;
}

template <int Dimensions>
inline const stream &operator<<(const stream &Out,
                                const item<Dimensions> &RHS) {
  detail::writeItem<Dimensions>(Out.GlobalFlushBuf, Out.FlushBufferSize,
                                Out.WIOffset, RHS);
  return Out;
}

template <int Dimensions>
inline const stream &operator<<(const stream &Out,
                                const nd_range<Dimensions> &RHS) {
  detail::writeNDRange<Dimensions>(Out.GlobalFlushBuf, Out.FlushBufferSize,
                                   Out.WIOffset, RHS);
  return Out;
}

template <int Dimensions>
inline const stream &operator<<(const stream &Out,
                                const nd_item<Dimensions> &RHS) {
  detail::writeNDItem<Dimensions>(Out.GlobalFlushBuf, Out.FlushBufferSize,
                                  Out.WIOffset, RHS);
  return Out;
}

template <int Dimensions>
inline const stream &operator<<(const stream &Out,
                                const group<Dimensions> &RHS) {
  detail::writeGroup<Dimensions>(Out.GlobalFlushBuf, Out.FlushBufferSize,
                                 Out.WIOffset, RHS);
  return Out;
}

template <int Dimensions>
inline const stream &operator<<(const stream &Out,
                                const h_item<Dimensions> &RHS) {
  detail::writeHItem<Dimensions>(Out.GlobalFlushBuf, Out.FlushBufferSize,
                                 Out.WIOffset, RHS);
  return Out;
}

template <typename T, typename RT = detail::EnableIfSwizzleVec<T>>
inline const stream &operator<<(const stream &Out, const T &RHS) {
  RT V = RHS;
  Out << V;
  return Out;
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
namespace std {
template <> struct hash<cl::sycl::stream> {
  size_t operator()(const cl::sycl::stream &S) const {
#ifdef __SYCL_DEVICE_ONLY__
    (void)S;
    return 0;
#else
    return hash<std::shared_ptr<cl::sycl::detail::stream_impl>>()(
        cl::sycl::detail::getSyclObjImpl(S));
#endif
  }
};
} // namespace std
