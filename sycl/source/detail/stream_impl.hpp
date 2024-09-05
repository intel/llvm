//==----------------- stream_impl.hpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/accessor.hpp>
#include <sycl/buffer.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/handler.hpp>
#include <sycl/property_list.hpp>
#include <sycl/range.hpp>
#include <sycl/stream.hpp>

#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

class stream_impl {
public:
  stream_impl(size_t BufferSize, size_t MaxStatementSize,
              const property_list &PropList);

  // Method to provide an access to the global stream buffer
  GlobalBufAccessorT accessGlobalBuf(handler &CGH);

  // Method to provide an accessor to the global flush buffer
  GlobalBufAccessorT accessGlobalFlushBuf(handler &CGH);

  // Method to provide an atomic access to the offset in the global stream
  // buffer and offset in the flush buffer
  GlobalOffsetAccessorT accessGlobalOffset(handler &CGH);

  size_t size() const noexcept;

  size_t get_work_item_buffer_size() const;

  void generateFlushCommand(handler &cgh);

  const property_list &getPropList() const { return PropList_; }

private:
  // Size of the stream buffer
  size_t BufferSize_;

  // Maximum number of symbols which could be streamed from the beginning of a
  // statement till the semicolon
  unsigned MaxStatementSize_;

  // Property list
  property_list PropList_;

  // It's fine to store the buffers in the stream_impl itself since the
  // underlying buffer_impls are relased in a deferred manner by scheduler.
  // Stream buffer
  buffer<char, 1> Buf_;

  // Global flush buffer
  buffer<char, 1> FlushBuf_;

  // Additinonal memory is allocated in the beginning of the stream buffer for
  // 2 variables: offset in the stream buffer and offset in the flush buffer.
  static const size_t OffsetSize = 2 * sizeof(unsigned);
};

} // namespace detail
} // namespace _V1
} // namespace sycl
