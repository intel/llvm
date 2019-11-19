//==----------------- stream.hpp - SYCL standard header file ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/stream_impl.hpp>

namespace cl {
namespace sycl {

enum class stream_manipulator {
  dec,
  hex,
  oct,
  noshowbase,
  showbase,
  noshowpos,
  showpos,
  endl,
  flush,
  fixed,
  scientific,
  hexfloat,
  defaultfloat
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

class stream {
public:
  stream(size_t BufferSize, size_t MaxStatementSize, handler &CGH);

  size_t get_size() const;

  size_t get_max_statement_size() const;

  size_t get_precision() const { return Precision; }

  size_t get_width() const { return Width; }

  stream_manipulator get_stream_mode() const { return Manipulator; }

  bool operator==(const stream &RHS) const;

  bool operator!=(const stream &LHS) const;

  ~stream() {
    // Flush data to global buffer in stream destruction if flush buffer is not
    // empty. This could be necessary if user hasn't flushed data himself and
    // kernel execution is finished
    // NOTE: In the current implementation user should explicitly flush data on
    // the host device. Data is not flushed automatically after kernel execution
    // because of the missing feature in scheduler.
    if (Offset) {
      flushBuffer(GlobalOffset, GlobalBuf, FlushBufs, WIOffset, Offset);
    }
  }

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
  mutable detail::stream_impl::GlobalBufAccessorT GlobalBuf;

  // Atomic accessor to the global offset variable. It represents an offset in
  // the global stream buffer. Since work items will flush data to global buffer
  // in parallel we need atomic access to this offset.
  mutable detail::stream_impl::GlobalOffsetAccessorT GlobalOffset;

  // Accessor to the pool of flush buffers. Flush buffer contains output from
  // work item in the work group. One flush buffer per work item in the work
  // group.
  mutable detail::stream_impl::FlushBufAccessorT FlushBufs;

  // Each work item in the work group writes to its own flush buffer in the
  // pool. This accessor is used to atomically get offset of the flush buffer in
  // the pool for each work item in the work group. This approach is used
  // because currently it is not possible to get work item id in the work group
  // without id object, which is passed to the kernel.
  mutable detail::stream_impl::LocalOffsetAccessorT WIOffsetAcc;

  mutable detail::stream_impl::GlobalOffsetAccessorT FlushSize;

  // Offset of the WI's flush buffer in the pool.
  mutable unsigned WIOffset = 0;

  // Offset in the flush buffer
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
  void __init() {
    // Calculate work item id inside work group, this should be done once, that
    // is why this is done in _init method, call to __init method is generated
    // by frontend. As a result each work item will write to its own flush
    // buffer.
    FlushBufferSize = FlushSize[0].load();
    WIOffsetAcc[0].store(0);
    detail::workGroupBarrier();
    WIOffset = WIOffsetAcc[0].fetch_add(FlushBufferSize);
  }
#endif

  friend const stream &operator<<(const stream &, const char);
  friend const stream &operator<<(const stream &, const char *);
  template <typename ValueType>
  friend typename std::enable_if<std::is_integral<ValueType>::value,
                                 const stream &>::type
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
  if (Out.Offset >= Out.FlushBufferSize ||
      Out.WIOffset + Out.Offset + 1 > Out.FlushBufs.get_count())
    return Out;
  Out.FlushBufs[Out.WIOffset + Out.Offset] = C;
  ++Out.Offset;
  return Out;
}

// String
inline const stream &operator<<(const stream &Out, const char *Str) {
  unsigned Len = 0;
  for (; Str[Len] != '\0'; Len++)
    ;

  detail::write(Out.FlushBufs, Out.FlushBufferSize, Out.WIOffset, Out.Offset,
                Str, Len);
  return Out;
}

// Boolean
inline const stream &operator<<(const stream &Out, const bool &RHS) {
  Out << (RHS ? "true" : "false");
  return Out;
}

// Integral
template <typename ValueType>
inline typename std::enable_if<std::is_integral<ValueType>::value,
                               const stream &>::type
operator<<(const stream &Out, const ValueType &RHS) {
  detail::writeIntegral(Out.FlushBufs, Out.FlushBufferSize, Out.WIOffset,
                        Out.Offset, Out.get_flags(), Out.get_width(), RHS);
  return Out;
}

// Floating points

inline const stream &operator<<(const stream &Out, const float &RHS) {
  detail::writeFloatingPoint<float>(Out.FlushBufs, Out.FlushBufferSize,
                                    Out.WIOffset, Out.Offset, Out.get_flags(),
                                    Out.get_width(), Out.get_precision(), RHS);
  return Out;
}

inline const stream &operator<<(const stream &Out, const double &RHS) {
  detail::writeFloatingPoint<double>(Out.FlushBufs, Out.FlushBufferSize,
                                     Out.WIOffset, Out.Offset, Out.get_flags(),
                                     Out.get_width(), Out.get_precision(), RHS);
  return Out;
}

inline const stream &operator<<(const stream &Out, const half &RHS) {
  detail::writeFloatingPoint<half>(Out.FlushBufs, Out.FlushBufferSize,
                                   Out.WIOffset, Out.Offset, Out.get_flags(),
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
  detail::writeIntegral(Out.FlushBufs, Out.FlushBufferSize, Out.WIOffset,
                        Out.Offset, Flags, Out.get_width(),
                        reinterpret_cast<size_t>(RHS));
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
    flushBuffer(Out.GlobalOffset, Out.GlobalBuf, Out.FlushBufs, Out.WIOffset,
                Out.Offset);
    break;
  case stream_manipulator::flush:
    flushBuffer(Out.GlobalOffset, Out.GlobalBuf, Out.FlushBufs, Out.WIOffset,
                Out.Offset);
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
  detail::writeVec<T, VectorLength>(Out.FlushBufs, Out.FlushBufferSize,
                                    Out.WIOffset, Out.Offset, Out.get_flags(),
                                    Out.get_width(), Out.get_precision(), RHS);
  return Out;
}

// SYCL types

template <int Dimensions>
inline const stream &operator<<(const stream &Out, const id<Dimensions> &RHS) {
  detail::writeArray<Dimensions>(Out.FlushBufs, Out.FlushBufferSize,
                                 Out.WIOffset, Out.Offset, RHS);
  return Out;
}

template <int Dimensions>
inline const stream &operator<<(const stream &Out,
                                const range<Dimensions> &RHS) {
  detail::writeArray<Dimensions>(Out.FlushBufs, Out.FlushBufferSize,
                                 Out.WIOffset, Out.Offset, RHS);
  return Out;
}

template <int Dimensions>
inline const stream &operator<<(const stream &Out,
                                const item<Dimensions> &RHS) {
  detail::writeItem<Dimensions>(Out.FlushBufs, Out.FlushBufferSize,
                                Out.WIOffset, Out.Offset, RHS);
  return Out;
}

template <int Dimensions>
inline const stream &operator<<(const stream &Out,
                                const nd_range<Dimensions> &RHS) {
  detail::writeNDRange<Dimensions>(Out.FlushBufs, Out.FlushBufferSize,
                                   Out.WIOffset, Out.Offset, RHS);
  return Out;
}

template <int Dimensions>
inline const stream &operator<<(const stream &Out,
                                const nd_item<Dimensions> &RHS) {
  detail::writeNDItem<Dimensions>(Out.FlushBufs, Out.FlushBufferSize,
                                  Out.WIOffset, Out.Offset, RHS);
  return Out;
}

template <int Dimensions>
inline const stream &operator<<(const stream &Out,
                                const group<Dimensions> &RHS) {
  detail::writeGroup<Dimensions>(Out.FlushBufs, Out.FlushBufferSize,
                                 Out.WIOffset, Out.Offset, RHS);
  return Out;
}

template <int Dimensions>
inline const stream &operator<<(const stream &Out,
                                const h_item<Dimensions> &RHS) {
  detail::writeHItem<Dimensions>(Out.FlushBufs, Out.FlushBufferSize,
                                 Out.WIOffset, Out.Offset, RHS);
  return Out;
}

template <typename T, typename RT = detail::EnableIfSwizzleVec<T>>
inline const stream &operator<<(const stream &Out, const T &RHS) {
  RT V = RHS;
  Out << V;
  return Out;
}

} // namespace sycl
} // namespace cl
namespace std {
template <> struct hash<cl::sycl::stream> {
  size_t operator()(const cl::sycl::stream &S) const {
#ifdef __SYCL_DEVICE_ONLY__
    return 0;
#else
    return hash<std::shared_ptr<cl::sycl::detail::stream_impl>>()(
        cl::sycl::detail::getSyclObjImpl(S));
#endif
  }
};
} // namespace std

