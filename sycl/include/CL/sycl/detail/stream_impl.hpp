//==----------------- stream_impl.hpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/builtins.hpp>
#include <CL/sycl/detail/array.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/queue.hpp>

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

template <class F, class T = void>
using EnableIfFP = typename std::enable_if<std::is_same<F, float>::value ||
                                               std::is_same<F, double>::value ||
                                               std::is_same<F, half>::value,
                                           T>::type;

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
    typename std::enable_if<IsSwizzleOp<T>::value,
                            typename IsSwizzleOp<T>::Type>::type;

class __SYCL_EXPORT stream_impl {
public:
  using GlobalBufAccessorT =
      accessor<char, 1, cl::sycl::access::mode::read_write,
               cl::sycl::access::target::global_buffer,
               cl::sycl::access::placeholder::false_t>;

  using GlobalOffsetAccessorT =
      accessor<unsigned, 1, cl::sycl::access::mode::atomic,
               cl::sycl::access::target::global_buffer,
               cl::sycl::access::placeholder::false_t>;

  stream_impl(size_t BufferSize, size_t MaxStatementSize, handler &CGH);

  // Method to provide an access to the global stream buffer
  GlobalBufAccessorT accessGlobalBuf(handler &CGH) {
    return Buf.get_access<cl::sycl::access::mode::read_write>(
        CGH, range<1>(BufferSize_), id<1>(OffsetSize));
  }

  // Method to provide an accessor to the global flush buffer
  GlobalBufAccessorT accessGlobalFlushBuf(handler &CGH) {
    return FlushBuf.get_access<cl::sycl::access::mode::read_write>(
        CGH, range<1>(MaxStatementSize_), id<1>(0));
  }

  // Method to provide an atomic access to the offset in the global stream
  // buffer
  GlobalOffsetAccessorT accessGlobalOffset(handler &CGH) {
    auto OffsetSubBuf = buffer<char, 1>(Buf, id<1>(0), range<1>(OffsetSize));
    auto ReinterpretedBuf = OffsetSubBuf.reinterpret<unsigned, 1>(range<1>(1));
    ReinterpretedBuf.set_write_back(false); // Buf handles write back.
    return ReinterpretedBuf.get_access<cl::sycl::access::mode::atomic>(
        CGH, range<1>(1), id<1>(0));
  }

  // Copy stream buffer to the host and print the contents
  void flush();

  size_t get_size() const;

  size_t get_max_statement_size() const;

private:
  // Size of the stream buffer
  size_t BufferSize_;

  // Maximum number of symbols which could be streamed from the beginning of a
  // statement till the semicolon
  unsigned MaxStatementSize_;

  // Size of the variable which is used as an offset in the stream buffer.
  // Additinonal memory is allocated in the beginning of the stream buffer for
  // this variable.
  static const size_t OffsetSize = sizeof(unsigned);

  // Vector on the host side which is used to initialize the stream buffer
  std::vector<char> Data;

  // Stream buffer
  buffer<char, 1> Buf;

  // Global flush buffer
  buffer<char, 1> FlushBuf;
};

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
inline typename std::enable_if<std::is_integral<T>::value, unsigned>::type
integralToBase(T Val, int Base, char *Digits) {
  unsigned NumDigits = 0;

  do {
    Digits[NumDigits++] = digitToChar(Val % Base);
    Val /= Base;
  } while (Val);

  return NumDigits;
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

// Helper method to update offset in the global buffer atomically according to
// the provided size of the data in the flush buffer. Return true if offset is
// updated and false in case of overflow.
inline bool updateOffset(stream_impl::GlobalOffsetAccessorT &GlobalOffset,
                         stream_impl::GlobalBufAccessorT &GlobalBuf,
                         unsigned Size, unsigned &Cur) {
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

inline void flushBuffer(stream_impl::GlobalOffsetAccessorT &GlobalOffset,
                        stream_impl::GlobalBufAccessorT &GlobalBuf,
                        stream_impl::GlobalBufAccessorT &GlobalFlushBuf,
                        unsigned &WIOffset, unsigned &Offset) {

  unsigned Cur = 0;
  if (!updateOffset(GlobalOffset, GlobalBuf, Offset, Cur))
    return;

  for (unsigned I = WIOffset; I < WIOffset + Offset; I++) {
    GlobalBuf[Cur++] = GlobalFlushBuf[I];
  }
  // Reset the offset in the flush buffer
  Offset = 0;
}

inline void write(stream_impl::GlobalBufAccessorT &GlobalFlushBuf,
                  size_t FlushBufferSize, unsigned WIOffset, unsigned &Offset,
                  const char *Str, unsigned Len, unsigned Padding = 0) {
  if ((FlushBufferSize - Offset < Len + Padding) ||
      (WIOffset + Offset + Len + Padding > GlobalFlushBuf.get_count()))
    // TODO: flush here
    return;

  // Write padding
  for (size_t I = 0; I < Padding; ++I, ++Offset)
    GlobalFlushBuf[WIOffset + Offset] = ' ';

  for (size_t I = 0; I < Len; ++I, ++Offset) {
    GlobalFlushBuf[WIOffset + Offset] = Str[I];
  }
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

inline unsigned append(char *Dst, const char *Src) {
  unsigned Len = 0;
  for (; Src[Len] != '\0'; ++Len)
    ;

  for (unsigned I = 0; I < Len; ++I)
    Dst[I] = Src[I];
  return Len;
}

template <typename T>
inline typename std::enable_if<std::is_same<T, half>::value, unsigned>::type
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
inline typename std::enable_if<std::is_same<T, float>::value ||
                                   std::is_same<T, double>::value,
                               unsigned>::type
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

// Returns number of symbols written to the buffer
template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, unsigned>::type
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

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value>::type
writeIntegral(stream_impl::GlobalBufAccessorT &GlobalFlushBuf,
              size_t FlushBufferSize, unsigned WIOffset, unsigned &Offset,
              unsigned Flags, int Width, const T &Val) {
  char Digits[MAX_INTEGRAL_DIGITS] = {0};
  unsigned Len = ScalarToStr(Val, Digits, Flags, Width);
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, Offset, Digits, Len,
        (Width > 0 && static_cast<unsigned>(Width) > Len)
            ? static_cast<unsigned>(Width) - Len
            : 0);
}

template <typename T>
inline EnableIfFP<T>
writeFloatingPoint(stream_impl::GlobalBufAccessorT &GlobalFlushBuf,
                   size_t FlushBufferSize, unsigned WIOffset, unsigned &Offset,
                   unsigned Flags, int Width, int Precision, const T &Val) {
  char Digits[MAX_FLOATING_POINT_DIGITS] = {0};
  unsigned Len = ScalarToStr(Val, Digits, Flags, Width, Precision);
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, Offset, Digits, Len,
        (Width > 0 && static_cast<unsigned>(Width) > Len)
            ? static_cast<unsigned>(Width) - Len
            : 0);
}

template <typename T, int VecLength>
typename std::enable_if<(VecLength == 1), unsigned>::type
VecToStr(const vec<T, VecLength> &Vec, char *VecStr, unsigned Flags, int Width,
         int Precision) {
  return ScalarToStr(static_cast<T>(Vec.x()), VecStr, Flags, Width, Precision);
}

template <typename T, int VecLength>
typename std::enable_if<(VecLength == 2 || VecLength == 4 || VecLength == 8 ||
                         VecLength == 16),
                        unsigned>::type
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
typename std::enable_if<(VecLength == 3), unsigned>::type
VecToStr(const vec<T, VecLength> &Vec, char *VecStr, unsigned Flags, int Width,
         int Precision) {
  unsigned Len = VecToStr<T, 2>(Vec.lo(), VecStr, Flags, Width, Precision);
  Len += append(VecStr + Len, VEC_ELEMENT_DELIMITER);
  Len += VecToStr<T, 1>(Vec.z(), VecStr + Len, Flags, Width, Precision);
  return Len;
}

template <typename T, int VecLength>
inline void writeVec(stream_impl::GlobalBufAccessorT &GlobalFlushBuf,
                     size_t FlushBufferSize, unsigned WIOffset,
                     unsigned &Offset, unsigned Flags, int Width, int Precision,
                     const vec<T, VecLength> &Vec) {
  // Reserve space for vector elements and delimiters
  constexpr size_t MAX_VEC_SIZE =
      MAX_FLOATING_POINT_DIGITS * VecLength + (VecLength - 1) * 2;
  char VecStr[MAX_VEC_SIZE] = {0};
  unsigned Len = VecToStr<T, VecLength>(Vec, VecStr, Flags, Width, Precision);
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, Offset, VecStr, Len,
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
inline void writeArray(stream_impl::GlobalBufAccessorT &GlobalFlushBuf,
                       size_t FlushBufferSize, unsigned WIOffset,
                       unsigned &Offset, const array<ArrayLength> &Arr) {
  char Buf[MAX_ARRAY_SIZE];
  unsigned Len = ArrayToStr(Buf, Arr);
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, Offset, Buf, Len);
}

template <int Dimensions>
inline void writeItem(stream_impl::GlobalBufAccessorT &GlobalFlushBuf,
                      size_t FlushBufferSize, unsigned WIOffset,
                      unsigned &Offset, const item<Dimensions> &Item) {
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
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, Offset, Buf, Len);
}

template <int Dimensions>
inline void writeNDRange(stream_impl::GlobalBufAccessorT &GlobalFlushBuf,
                         size_t FlushBufferSize, unsigned WIOffset,
                         unsigned &Offset,
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
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, Offset, Buf, Len);
}

template <int Dimensions>
inline void writeNDItem(stream_impl::GlobalBufAccessorT &GlobalFlushBuf,
                        size_t FlushBufferSize, unsigned WIOffset,
                        unsigned &Offset, const nd_item<Dimensions> &ND_Item) {
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
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, Offset, Buf, Len);
}

template <int Dimensions>
inline void writeGroup(stream_impl::GlobalBufAccessorT &GlobalFlushBuf,
                       size_t FlushBufferSize, unsigned WIOffset,
                       unsigned &Offset, const group<Dimensions> &Group) {
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
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, Offset, Buf, Len);
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
inline void writeHItem(stream_impl::GlobalBufAccessorT &GlobalFlushBuf,
                       size_t FlushBufferSize, unsigned WIOffset,
                       unsigned &Offset, const h_item<Dimensions> &HItem) {
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
  write(GlobalFlushBuf, FlushBufferSize, WIOffset, Offset, Buf, Len);
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
