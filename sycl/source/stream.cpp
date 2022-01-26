//==------------------- stream.cpp - SYCL standard header file ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/properties/all_properties.hpp>
#include <CL/sycl/stream.hpp>
#include <detail/queue_impl.hpp>
#include <detail/stream_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Maximum possible size of a flush buffer statement in bytes
static constexpr size_t MAX_STATEMENT_SIZE =
    (1 << (CHAR_BIT * detail::FLUSH_BUF_OFFSET_SIZE)) - 1;

// Checks the MaxStatementSize argument of the sycl::stream class. This is
// called on MaxStatementSize as it is passed to the constructor of the
// underlying stream_impl to make it throw before the stream buffers are
// allocated, avoiding memory leaks.
static size_t CheckMaxStatementSize(const size_t &MaxStatementSize) {
  if (MaxStatementSize > MAX_STATEMENT_SIZE) {
    throw sycl::invalid_parameter_error(
        "Maximum statement size exceeds limit of " +
            std::to_string(MAX_STATEMENT_SIZE) + " bytes.",
        PI_INVALID_VALUE);
  }
  return MaxStatementSize;
}

stream::stream(size_t BufferSize, size_t MaxStatementSize, handler &CGH)
    : stream(BufferSize, MaxStatementSize, CGH, {}) {}

stream::stream(size_t BufferSize, size_t MaxStatementSize, handler &CGH,
               const property_list &PropList)
    : impl(std::make_shared<detail::stream_impl>(
          BufferSize, CheckMaxStatementSize(MaxStatementSize), PropList)),
      GlobalBuf(impl->accessGlobalBuf(CGH)),
      GlobalOffset(impl->accessGlobalOffset(CGH)),
      // Allocate the flush buffer, which contains space for each work item
      GlobalFlushBuf(impl->accessGlobalFlushBuf(CGH)),
      FlushBufferSize(MaxStatementSize + detail::FLUSH_BUF_OFFSET_SIZE) {
  // Save stream implementation in the handler so that stream will be alive
  // during kernel execution
  CGH.addStream(impl);

  // Set flag identifying that created accessor has perWI size. Accessor
  // will be resized in SYCL RT when number of work items will be
  // available.
  detail::getSyclObjImpl(GlobalFlushBuf)->PerWI = true;
}

size_t stream::get_size() const { return impl->get_size(); }

size_t stream::get_max_statement_size() const {
  return impl->get_max_statement_size();
}

bool stream::operator==(const stream &RHS) const { return (impl == RHS.impl); }

bool stream::operator!=(const stream &RHS) const { return !(impl == RHS.impl); }

#define __SYCL_PARAM_TRAITS_SPEC(param_type)                                   \
  template <>                                                                  \
  __SYCL_EXPORT bool stream::has_property<param_type>() const noexcept {       \
    return impl->has_property<param_type>();                                   \
  }
#include <CL/sycl/detail/properties_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

#define __SYCL_PARAM_TRAITS_SPEC(param_type)                                   \
  template <>                                                                  \
  __SYCL_EXPORT param_type stream::get_property<param_type>() const {          \
    return impl->get_property<param_type>();                                   \
  }
#include <CL/sycl/detail/properties_traits.def>

#undef __SYCL_PARAM_TRAITS_SPEC

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

