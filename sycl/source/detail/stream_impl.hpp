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
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/range.hpp>
#include <CL/sycl/stream.hpp>

#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

class __SYCL_EXPORT stream_impl {
public:
  // TODO: This constructor is unused.
  // To be removed when API/ABI changes are allowed.
  stream_impl(size_t BufferSize, size_t MaxStatementSize, handler &CGH);

  stream_impl(size_t BufferSize, size_t MaxStatementSize,
              const property_list &PropList);

  // Method to provide an access to the global stream buffer
  GlobalBufAccessorT accessGlobalBuf(handler &CGH);

  // Method to provide an accessor to the global flush buffer
  GlobalBufAccessorT accessGlobalFlushBuf(handler &CGH);

  // Method to provide an atomic access to the offset in the global stream
  // buffer and offset in the flush buffer
  GlobalOffsetAccessorT accessGlobalOffset(handler &CGH);

  // Enqueue task to copy stream buffer to the host and print the contents
  void flush();

  size_t get_size() const;

  size_t get_max_statement_size() const;

  template <typename propertyT> bool has_property() const noexcept {
    return PropList_.has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return PropList_.get_property<propertyT>();
  }

private:
  // Size of the stream buffer
  size_t BufferSize_;

  // Maximum number of symbols which could be streamed from the beginning of a
  // statement till the semicolon
  unsigned MaxStatementSize_;

  // Property list
  property_list PropList_;

  // Additinonal memory is allocated in the beginning of the stream buffer for
  // 2 variables: offset in the stream buffer and offset in the flush buffer.
  static const size_t OffsetSize = 2 * sizeof(unsigned);

};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
