//==------------ sycl_mem_obj_t.hpp - SYCL standard header file ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>

#include <deque>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

class handler;
namespace detail {

// allows us to represent whether a user has set (or not) a boolean value, and,
// if so, to what.
enum class settable_bool { set_false = -1, not_set, set_true };

// acts as return value of whenCopyBack(buff*).  which returns dtor, immediate
// or never acts as the 'now' parameter for copyBackSubBuffer(now, buff*),
//  because that function will be called from two places (dtor and immedaite)
// acts as the 'when' parameter for shouldCopyBack(when, buff*) (dtor, immediate
// and undertermined)
//  where 'undetermined' is simply "will this buffer copy back at any time?"
enum class when_copyback {
  dtor,      // addCopyBack during sub-buffer dtor
  immediate, // copy-back achieved by enqueued map operation
  never
};

// need to track information about a sub/buffer,
// even after its destruction, we may need to know about it.
struct buffer_info {
  const size_t SizeInBytes;
  const size_t OffsetInBytes;
  const bool IsSubBuffer;

  buffer_info(const size_t sz, const size_t offset, const bool IsSub)
      : SizeInBytes(sz), OffsetInBytes(offset), IsSubBuffer(IsSub) {}
};

// given a sub/buffer, this tracks how it was used (were there write accessors,
// etc.)
struct buffer_usage {
  // the address of a sub/buffer is used to uniquely identify it, but is never
  // dereferenced.
  const void *const buffAddr;

  // basic info about the buffer (range, offset, isSub)
  buffer_info BufferInfo;

  // did the user set the writeback?
  settable_bool MWriteBackSet;

  // History of accessor modes and devices.
  std::deque<
      std::tuple<bool, access::mode, std::shared_ptr<detail::context_impl>>>
      MHistory;

  // ctor
  buffer_usage(const void *const BuffPtr, const size_t Sz, const size_t Offset,
               const bool IsSub)
      : buffAddr(BuffPtr), BufferInfo(Sz, Offset, IsSub),
        MWriteBackSet(settable_bool::not_set){};
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)