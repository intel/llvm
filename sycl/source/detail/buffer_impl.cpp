//==----------------- buffer_impl.cpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/buffer_impl.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/scheduler/scheduler.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
void *buffer_impl::allocateMem(ContextImplPtr Context, bool InitFromUserData,
                  void *HostPtr, RT::PiEvent &OutEventToWait) {

  assert(!(InitFromUserData && HostPtr) &&
          "Cannot init from user data and reuse host ptr provided "
          "simultaneously");

  void *UserPtr = InitFromUserData ? BaseT::getUserPtr() : HostPtr;

  assert(!(nullptr == UserPtr && BaseT::useHostPtr() && Context->is_host()) &&
          "Internal error. Allocating memory on the host "
          "while having use_host_ptr property");

  return MemoryManager::allocateMemBuffer(
      std::move(Context), this, UserPtr, BaseT::MHostPtrReadOnly,
      BaseT::getSize(), BaseT::MInteropEvent, BaseT::MInteropContext,
      OutEventToWait);
}

using ContextImplPtr = std::shared_ptr<detail::context_impl>;

void buffer_impl::recordBufferUsage(const void *const BuffPtr, const size_t Sz,
                                    const size_t Offset, const bool IsSub) {
  MBufferUsageDQ.emplace_back(BuffPtr, Sz, Offset, IsSub);
}

static bool isAReadMode(access::mode Mode) {
  if (Mode == access::mode::write || Mode == access::mode::discard_write)
    return false;
  else
    return true;
}

static bool isAWriteMode(access::mode Mode) {
  if (Mode == access::mode::read || Mode == access::mode::discard_write ||
      Mode == access::mode::discard_read_write)
    return false;
  else
    return true;
}

static detail::when_copyback whenDataResolves(buffer_usage &BU) {
  using hentry = std::tuple<bool, access::mode, ContextImplPtr>;

  detail::when_copyback when = when_copyback::never;
  find_if(BU.MHistory.begin(), BU.MHistory.end(), [&when](hentry HEntry) {
    // returns at first consequential entry.

    // writing on device
    if (!std::get<0>(HEntry) && isAWriteMode(std::get<1>(HEntry))) {
      when = when_copyback::dtor;
      return true;
    }
    // blocking host read (was updated via map op)
    if (std::get<0>(HEntry) && isAReadMode(std::get<1>(HEntry))) {
      when = when_copyback::immediate;
      return true;
    }
    // continue
    return false;
  });

  // immediate map op doesn't use writeback flag, return before flag check.
  if (when == when_copyback::immediate)
    return when;

  if (BU.MWriteBackSet == settable_bool::set_false)
    return when_copyback::never;

  return when;
}

static bool needDtorCopyBack(buffer_usage &BU) {
  return (whenDataResolves(BU) == when_copyback::dtor);
}

static ContextImplPtr getDtorCopyBackCtxImpl(buffer_usage &BU) {
  using hentry = std::tuple<bool, access::mode, ContextImplPtr>;

  ContextImplPtr theCtx = nullptr;
  find_if(BU.MHistory.begin(), BU.MHistory.end(), [&theCtx](hentry HEntry) {
    // same logic as needDtorCopyBack
    if (!std::get<0>(HEntry) && isAWriteMode(std::get<1>(HEntry))) {
      theCtx = std::get<2>(HEntry);
      return true;
    }
    return false;
  });
  return theCtx;
}

bool buffer_impl::hasSubBuffers() { return MBufferUsageDQ.size() > 1; }

void buffer_impl::set_write_back(bool flag) {
  // called for normal buffers.
  SYCLMemObjT::set_write_back(flag);
}

void buffer_impl::set_write_back(bool flag, const void *const BuffPtr) {
  // only called for subbuffers, we need to know if WB was set, and if so, what
  // to.
  std::deque<buffer_usage>::iterator it =
      find_if(MBufferUsageDQ.begin(), MBufferUsageDQ.end(),
              [BuffPtr](buffer_usage &BU) { return (BU.buffAddr == BuffPtr); });
  assert(it != MBufferUsageDQ.end() && "no record of subbuffer");
  buffer_usage &BU = it[0];
  BU.MWriteBackSet = flag ? settable_bool::set_true : settable_bool::set_false;
}

void buffer_impl::recordAccessorUsage(const void *const BuffPtr,
                                      access::mode Mode, handler &CGH) {
  std::deque<buffer_usage>::iterator it =
      find_if(MBufferUsageDQ.begin(), MBufferUsageDQ.end(),
              [BuffPtr](buffer_usage &BU) { return (BU.buffAddr == BuffPtr); });
  assert(it != MBufferUsageDQ.end() && "no record of (sub)buffer");
  buffer_usage &BU = it[0];

  bool v = detail::getDeviceFromHandler(CGH).is_host();
  ContextImplPtr ctx = detail::getContextFromHandler(CGH).impl;
  BU.MHistory.emplace_front(v, Mode, ctx);
}

void buffer_impl::recordAccessorUsage(const void *const BuffPtr,
                                      access::mode Mode) {
  std::deque<buffer_usage>::iterator it =
      find_if(MBufferUsageDQ.begin(), MBufferUsageDQ.end(),
              [BuffPtr](buffer_usage &BU) { return (BU.buffAddr == BuffPtr); });
  assert(it != MBufferUsageDQ.end() && "no record of (sub)buffer");
  buffer_usage &BU = it[0];

  BU.MHistory.emplace_front(true, Mode, nullptr);
}

static EventImplPtr scheduleSubCopyBack(buffer_impl *impl, buffer_info Info) {
  const id<3> Offset{Info.OffsetInBytes, 0, 0};
  const range<3> AccessRange{Info.SizeInBytes, 1, 1};
  const range<3> MemoryRange{Info.SizeInBytes, 1, 1};
  const access::mode AccessMode = access::mode::read;
  SYCLMemObjI *SYCLMemObject = impl;
  const int Dims = 1;
  const int ElemSize = 1;

  Requirement Req(Offset, AccessRange, MemoryRange, AccessMode, SYCLMemObject,
                  Dims, ElemSize, Info.OffsetInBytes, Info.IsSubBuffer);

  void *DataPtr = impl->getUserPtr(); // TODO - interface with set_final_data
  if (DataPtr != nullptr) {
    Req.MData = DataPtr;
    return Scheduler::getInstance().addCopyBack(&Req);
  }
  return nullptr;
}

// This detects the weird situation where the base buffer AND sub-buffer have
// all been performing write operations,
//  possibly in different contexts.  The sub-buffers take care of themselves.
//  Here we see if the base buffer actually needs to do anything, and if so,
//  schedule the copy-back.
void buffer_impl::copyBackAnyRemainingData() {
  buffer_usage &baseBU = MBufferUsageDQ.front();
  assert(!baseBU.BufferInfo.IsSubBuffer &&
         "first BU should be base buffer, not subbuffer");
  if (needDtorCopyBack(baseBU)) { // in unusual case base buffer ALSO used a
                                  // write acc on device.
    // NOTE: it's possible to measure how much of the base buffer "remains"
    //       and/or to count the number of discrete write-backs would be needed
    //       to resolve it without overlapping the sub-buffers. Then decide if
    //       it should be copied back in small pieces, rather than at once.  But
    //       this also requires changes to the Schedule.addCopyBack() interface.
    //       At this moment, in this unusual case, we simply copy back the
    //       entire base buffer.

    // the sub-buffers may have been used in different contexts than the base.
    // which means whatever context the base write accessor used might have been
    // changed since then.
    ContextImplPtr newCtx = getDtorCopyBackCtxImpl(baseBU);
    assert((newCtx != nullptr) && "missing base buffer context");
    auto theCtx = MRecord->MCurContext;
    MRecord->MCurContext = newCtx;

    // schedule
    EventImplPtr Event = scheduleSubCopyBack(this, baseBU.BufferInfo);
    if (Event)
      Event->wait(Event);

    // restore context
    MRecord->MCurContext = theCtx;
  }
}

void buffer_impl::copyBackSubBuffer(detail::when_copyback now,
                                    const void *const BuffPtr) {
  // find record of buffer_usage
  std::deque<buffer_usage>::iterator it =
      find_if(MBufferUsageDQ.begin(), MBufferUsageDQ.end(),
              [BuffPtr](buffer_usage &BU) { return (BU.buffAddr == BuffPtr); });
  assert(it != MBufferUsageDQ.end() && "no record of subbuffer");
  buffer_usage &BU = it[0];

  if (needDtorCopyBack(BU)) {
    // last context for the sub-buffer might not be the same as any last access
    // to greater buffer_impl.
    ContextImplPtr newCtx = getDtorCopyBackCtxImpl(BU);
    assert((newCtx != nullptr) &&
           "sub-buffer copyback context null (or host?)");
    auto theCtx = MRecord->MCurContext;
    MRecord->MCurContext = newCtx;

    EventImplPtr Event = scheduleSubCopyBack(this, BU.BufferInfo);
    if (Event)
      Event->wait(Event);

    // restore context
    MRecord->MCurContext = theCtx;
  }
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
