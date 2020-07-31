//==----------------- stream_impl.hpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/buffer.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/range.hpp>
#include <CL/sycl/stream.hpp>

#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

class __SYCL_EXPORT stream_impl {
public:
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

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
