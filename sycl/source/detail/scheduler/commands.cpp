//===----------- commands.cpp - SYCL commands -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CL/sycl/access/access.hpp"
#include <CL/cl.h>
#include <CL/sycl/detail/clusm.hpp>
#include <CL/sycl/detail/event_impl.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/kernel_info.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/detail/program_manager/program_manager.hpp>
#include <CL/sycl/detail/queue_impl.hpp>
#include <CL/sycl/detail/scheduler/commands.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/detail/stream_impl.hpp>
#include <CL/sycl/sampler.hpp>

#include <vector>

#ifdef __GNUG__
#include <cstdlib>
#include <cxxabi.h>
#include <memory>
#endif

namespace cl {
namespace sycl {
namespace detail {

#ifdef __GNUG__
struct DemangleHandle {
  char *p;
  DemangleHandle(char *ptr) : p(ptr) {}
  ~DemangleHandle() { std::free(p); }
};
static std::string demangleKernelName(std::string Name) {
  int Status = -1; // some arbitrary value to eliminate the compiler warning
  DemangleHandle result(abi::__cxa_demangle(Name.c_str(), NULL, NULL, &Status));
  return (Status == 0) ? result.p : Name;
}
#else
static std::string demangleKernelName(std::string Name) { return Name; }
#endif

static std::string deviceToString(device Device) {
  if (Device.is_host())
    return "HOST";
  else if (Device.is_cpu())
    return "CPU";
  else if (Device.is_gpu())
    return "GPU";
  else if (Device.is_accelerator())
    return "ACCELERATOR";
  else
    return "UNKNOWN";
}

static std::string accessModeToString(access::mode Mode) {
  switch (Mode) {
  case access::mode::read:
    return "read";
  case access::mode::write:
    return "write";
  case access::mode::read_write:
    return "read_write";
  case access::mode::discard_write:
    return "discard_write";
  case access::mode::discard_read_write:
    return "discard_read_write";
  default:
    return "unknown";
  }
}

void EventCompletionClbk(RT::PiEvent, pi_int32, void *data) {
  // TODO: Handle return values. Store errors to async handler.
  PI_CALL(RT::piEventSetStatus(pi::cast<RT::PiEvent>(data), CL_COMPLETE));
}

// Method prepares PI event's from list sycl::event's
std::vector<RT::PiEvent> Command::prepareEvents(ContextImplPtr Context) {
  std::vector<RT::PiEvent> Result;
  std::vector<EventImplPtr> GlueEvents;
  for (EventImplPtr &Event : MDepsEvents) {
    // Async work is not supported for host device.
    if (Event->is_host()) {
      Event->waitInternal();
      continue;
    }
    // The event handle can be null in case of, for example, alloca command,
    // which is currently synchrounious, so don't generate OpenCL event.
    if (Event->getHandleRef() == nullptr) {
      continue;
    }
    ContextImplPtr EventContext = Event->getContextImpl();

    // If contexts don't match - connect them using user event
    if (EventContext != Context && !Context->is_host()) {
      RT::PiResult Error = PI_SUCCESS;

      EventImplPtr GlueEvent(new detail::event_impl());
      GlueEvent->setContextImpl(Context);

      RT::PiEvent &GlueEventHandle = GlueEvent->getHandleRef();
      PI_CALL((GlueEventHandle = RT::piEventCreate(
          Context->getHandleRef(), &Error), Error));
      PI_CALL(RT::piEventSetCallback(
          Event->getHandleRef(), CL_COMPLETE, EventCompletionClbk,
          /*data=*/GlueEventHandle));
      GlueEvents.push_back(std::move(GlueEvent));
      Result.push_back(GlueEventHandle);
      continue;
    }
    Result.push_back(Event->getHandleRef());
  }
  MDepsEvents.insert(MDepsEvents.end(), GlueEvents.begin(), GlueEvents.end());
  return Result;
}

Command::Command(CommandType Type, QueueImplPtr Queue, bool UseExclusiveQueue)
    : MQueue(std::move(Queue)), MUseExclusiveQueue(UseExclusiveQueue),
      MType(Type), MEnqueued(false) {
  MEvent.reset(new detail::event_impl());
  MEvent->setCommand(this);
  MEvent->setContextImpl(detail::getSyclObjImpl(MQueue->get_context()));
}

cl_int Command::enqueue() {
  bool Expected = false;
  if (MEnqueued.compare_exchange_strong(Expected, true))
    return enqueueImp();
  return CL_SUCCESS;
}

cl_int AllocaCommand::enqueueImp() {
  std::vector<RT::PiEvent> RawEvents =
      Command::prepareEvents(detail::getSyclObjImpl(MQueue->get_context()));

  RT::PiEvent &Event = MEvent->getHandleRef();
  MMemAllocation = MemoryManager::allocate(
      detail::getSyclObjImpl(MQueue->get_context()), getSYCLMemObj(),
      MInitFromUserData, std::move(RawEvents), Event);
  return CL_SUCCESS;
}

void AllocaCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#FFD28A\", label=\"";

  Stream << "ID = " << this << "\\n";
  Stream << "ALLOCA ON " << deviceToString(MQueue->get_device()) << "\\n";
  Stream << " MemObj : " << this->MReq.MSYCLMemObj << "\\n";
  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    if (Dep.MDepCommand == nullptr)
      continue;
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MReq->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MReq->MSYCLMemObj << " \" ]" << std::endl;
  }
}

cl_int AllocaSubBufCommand::enqueueImp() {
  std::vector<RT::PiEvent> RawEvents =
      Command::prepareEvents(detail::getSyclObjImpl(MQueue->get_context()));
  RT::PiEvent &Event = MEvent->getHandleRef();
  MMemAllocation = MemoryManager::createSubBuffer(
      pi::cast<RT::PiMem>(MParentAlloca->getMemAllocation()), MReq.MElemSize,
      MReq.MOffset, MReq.MAccessRange, std::move(RawEvents), Event);
  return CL_SUCCESS;
}

void AllocaSubBufCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#FFD28A\", label=\"";

  Stream << "ID = " << this << "\\n";
  Stream << "ALLOCA SUB BUF ON " << deviceToString(MQueue->get_device())
         << "\\n";
  Stream << " MemObj : " << this->MReq.MSYCLMemObj << "\\n";
  Stream << " Offset : " << this->MReq.MOffset[0] << "\\n";
  Stream << " Access range : " << this->MReq.MAccessRange[0] << "\\n";
  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    if (Dep.MDepCommand == nullptr)
      continue;
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MReq->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MReq->MSYCLMemObj << " \" ]" << std::endl;
  }
}

cl_int ReleaseCommand::enqueueImp() {
  std::vector<RT::PiEvent> RawEvents =
      Command::prepareEvents(detail::getSyclObjImpl(MQueue->get_context()));

  RT::PiEvent &Event = MEvent->getHandleRef();

  // On host side we only allocate memory for full buffers.
  // Thus, deallocating sub buffers leads to double memory freeing.
  if (!(MQueue->is_host() && MAllocaCmd->getType() == ALLOCA_SUB_BUF))
    MemoryManager::release(detail::getSyclObjImpl(MQueue->get_context()),
                           MAllocaCmd->getSYCLMemObj(),
                           MAllocaCmd->getMemAllocation(), std::move(RawEvents),
                           Event);
  else
    PI_CALL(RT::piEnqueueEventsWait(MQueue->getHandleRef(), RawEvents.size(),
                                    &RawEvents[0], &Event));

  return CL_SUCCESS;
}

void ReleaseCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#FF827A\", label=\"";

  Stream << "ID = " << this << " ; ";
  Stream << "RELEASE ON " << deviceToString(MQueue->get_device()) << "\\n";
  Stream << " Alloca : " << MAllocaCmd << "\\n";
  Stream << " MemObj : " << MAllocaCmd->getSYCLMemObj() << "\\n";
  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MReq->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MReq->MSYCLMemObj << " \" ]" << std::endl;
  }
}

MapMemObject::MapMemObject(Requirement SrcReq, AllocaCommandBase *SrcAlloca,
                           Requirement *DstAcc, QueueImplPtr Queue)
    : Command(CommandType::MAP_MEM_OBJ, std::move(Queue)),
      MSrcReq(std::move(SrcReq)), MSrcAlloca(SrcAlloca), MDstAcc(DstAcc),
      MDstReq(*DstAcc) {}

cl_int MapMemObject::enqueueImp() {
  std::vector<RT::PiEvent> RawEvents =
      Command::prepareEvents(detail::getSyclObjImpl(MQueue->get_context()));
  assert(MDstReq.MDims == 1);

  RT::PiEvent &Event = MEvent->getHandleRef();
  void *MappedPtr = MemoryManager::map(
      MSrcAlloca->getSYCLMemObj(), MSrcAlloca->getMemAllocation(), MQueue,
      MDstReq.MAccessMode, MDstReq.MDims, MDstReq.MMemoryRange,
      MDstReq.MAccessRange, MDstReq.MOffset, MDstReq.MElemSize,
      std::move(RawEvents), Event);
  MDstAcc->MData = MappedPtr;
  return CL_SUCCESS;
}

void MapMemObject::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#77AFFF\", label=\"";

  Stream << "ID = " << this << " ; ";
  Stream << "MAP ON " << deviceToString(MQueue->get_device()) << "\\n";

  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MReq->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MReq->MSYCLMemObj << " \" ]" << std::endl;
  }
}

UnMapMemObject::UnMapMemObject(Requirement SrcReq, AllocaCommandBase *SrcAlloca,
                               Requirement *DstAcc, QueueImplPtr Queue,
                               bool UseExclusiveQueue)
    : Command(CommandType::UNMAP_MEM_OBJ, std::move(Queue), UseExclusiveQueue),
      MSrcReq(std::move(SrcReq)), MSrcAlloca(SrcAlloca), MDstAcc(DstAcc) {}

cl_int UnMapMemObject::enqueueImp() {
  std::vector<RT::PiEvent> RawEvents =
      Command::prepareEvents(detail::getSyclObjImpl(MQueue->get_context()));

  RT::PiEvent &Event = MEvent->getHandleRef();
  MemoryManager::unmap(MSrcAlloca->getSYCLMemObj(),
                       MSrcAlloca->getMemAllocation(), MQueue, MDstAcc->MData,
                       std::move(RawEvents), MUseExclusiveQueue, Event);
  return CL_SUCCESS;
}

void UnMapMemObject::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#EBC40F\", label=\"";

  Stream << "ID = " << this << " ; ";
  Stream << "UNMAP ON " << deviceToString(MQueue->get_device()) << "\\n";

  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MReq->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MReq->MSYCLMemObj << " \" ]" << std::endl;
  }
}

MemCpyCommand::MemCpyCommand(Requirement SrcReq, AllocaCommandBase *SrcAlloca,
                             Requirement DstReq, AllocaCommandBase *DstAlloca,
                             QueueImplPtr SrcQueue, QueueImplPtr DstQueue,
                             bool UseExclusiveQueue)
    : Command(CommandType::COPY_MEMORY, std::move(DstQueue), UseExclusiveQueue),
      MSrcQueue(SrcQueue), MSrcReq(std::move(SrcReq)), MSrcAlloca(SrcAlloca),
      MDstReq(std::move(DstReq)), MDstAlloca(DstAlloca) {
  if (!MSrcQueue->is_host())
    MEvent->setContextImpl(detail::getSyclObjImpl(MSrcQueue->get_context()));
}

cl_int MemCpyCommand::enqueueImp() {
  std::vector<RT::PiEvent> RawEvents;
  QueueImplPtr Queue = MQueue->is_host() ? MSrcQueue : MQueue;
  RawEvents =
      Command::prepareEvents(detail::getSyclObjImpl(Queue->get_context()));

  RT::PiEvent &Event = MEvent->getHandleRef();

  // Omit copying if mode is discard one.
  // TODO: Handle this at the graph building time by, for example, creating
  // empty node instead of memcpy.
  if (MDstReq.MAccessMode == access::mode::discard_read_write ||
      MDstReq.MAccessMode == access::mode::discard_write ||
      MSrcAlloca->getMemAllocation() == MDstAlloca->getMemAllocation()) {

    if (!RawEvents.empty()) {
      if (Queue->is_host()) {
        PI_CALL(RT::piEventsWait(RawEvents.size(), &RawEvents[0]));
      } else {
        PI_CALL(RT::piEnqueueEventsWait(
            Queue->getHandleRef(), RawEvents.size(), &RawEvents[0], &Event));
      }
    }
  } else {
    MemoryManager::copy(
        MSrcAlloca->getSYCLMemObj(), MSrcAlloca->getMemAllocation(), MSrcQueue,
        MSrcReq.MDims, MSrcReq.MMemoryRange, MSrcReq.MAccessRange,
        MSrcReq.MOffset, MSrcReq.MElemSize, MDstAlloca->getMemAllocation(),
        MQueue, MDstReq.MDims, MDstReq.MMemoryRange, MDstReq.MAccessRange,
        MDstReq.MOffset, MDstReq.MElemSize, std::move(RawEvents),
        MUseExclusiveQueue, Event);
  }

  if (MAccToUpdate)
    MAccToUpdate->MData = MDstAlloca->getMemAllocation();
  return CL_SUCCESS;
}

void MemCpyCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#C7EB15\" label=\"";

  Stream << "ID = " << this << " ; ";
  Stream << "MEMCPY ON " << deviceToString(MQueue->get_device()) << "\\n";
  Stream << "From: " << MSrcAlloca << " is host: " << MSrcQueue->is_host()
         << "\\n";
  Stream << "To: " << MDstAlloca << " is host: " << MQueue->is_host() << "\\n";

  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MReq->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MReq->MSYCLMemObj << " \" ]" << std::endl;
  }
}

AllocaCommandBase *ExecCGCommand::getAllocaForReq(Requirement *Req) {
  for (const DepDesc &Dep : MDeps) {
    if (Dep.MReq == Req)
      return Dep.MAllocaCmd;
  }
  throw runtime_error("Alloca for command not found");
}

void ExecCGCommand::flushStreams() {
  assert(MCommandGroup->getType() == CG::KERNEL && "Expected kernel");
  for (auto StreamImplPtr :
       ((CGExecKernel *)MCommandGroup.get())->getStreams()) {
    StreamImplPtr->flush();
  }
}

MemCpyCommandHost::MemCpyCommandHost(Requirement SrcReq,
                                     AllocaCommandBase *SrcAlloca,
                                     Requirement *DstAcc, QueueImplPtr SrcQueue,
                                     QueueImplPtr DstQueue)
    : Command(CommandType::COPY_MEMORY, std::move(DstQueue)),
      MSrcQueue(SrcQueue), MSrcReq(std::move(SrcReq)), MSrcAlloca(SrcAlloca),
      MDstReq(*DstAcc), MDstAcc(DstAcc) {
  if (!MSrcQueue->is_host())
    MEvent->setContextImpl(detail::getSyclObjImpl(MSrcQueue->get_context()));
}

cl_int MemCpyCommandHost::enqueueImp() {
  QueueImplPtr Queue = MQueue->is_host() ? MSrcQueue : MQueue;
  std::vector<RT::PiEvent> RawEvents =
      Command::prepareEvents(detail::getSyclObjImpl(Queue->get_context()));

  RT::PiEvent &Event = MEvent->getHandleRef();
  // Omit copying if mode is discard one.
  // TODO: Handle this at the graph building time by, for example, creating
  // empty node instead of memcpy.
  if (MDstReq.MAccessMode == access::mode::discard_read_write ||
      MDstReq.MAccessMode == access::mode::discard_write) {

    if (!RawEvents.empty()) {
      if (Queue->is_host()) {
        PI_CALL(RT::piEventsWait(RawEvents.size(), &RawEvents[0]));
      } else {
        PI_CALL(RT::piEnqueueEventsWait(
            Queue->getHandleRef(), RawEvents.size(), &RawEvents[0], &Event));
      }
    }
    return CL_SUCCESS;
  }

  MemoryManager::copy(
      MSrcAlloca->getSYCLMemObj(), MSrcAlloca->getMemAllocation(), MSrcQueue,
      MSrcReq.MDims, MSrcReq.MMemoryRange, MSrcReq.MAccessRange,
      MSrcReq.MOffset, MSrcReq.MElemSize, MDstAcc->MData, MQueue, MDstReq.MDims,
      MDstReq.MMemoryRange, MDstReq.MAccessRange, MDstReq.MOffset,
      MDstReq.MElemSize, std::move(RawEvents), MUseExclusiveQueue, Event);
  return CL_SUCCESS;
}

void MemCpyCommandHost::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#B6A2EB\", label=\"";

  Stream << "ID = " << this << "\n";
  Stream << "MEMCPY HOST ON " << deviceToString(MQueue->get_device()) << "\\n";

  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MReq->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MReq->MSYCLMemObj << " \" ]" << std::endl;
  }
}

void ExecCGCommand::printDot(std::ostream &Stream) const {
  Stream << "\"" << this << "\" [style=filled, fillcolor=\"#AFFF82\", label=\"";

  Stream << "ID = " << this << "\n";
  Stream << "EXEC CG ON " << deviceToString(MQueue->get_device()) << "\\n";

  switch (MCommandGroup->getType()) {
  case detail::CG::KERNEL: {
    auto *KernelCG =
        reinterpret_cast<detail::CGExecKernel *>(MCommandGroup.get());
    Stream << "Kernel name: ";
    if (KernelCG->MSyclKernel && KernelCG->MSyclKernel->isCreatedFromSource())
      Stream << "created from source";
    else
      Stream << demangleKernelName(KernelCG->getKernelName());
    Stream << "\\n";
    break;
  }
  case detail::CG::UPDATE_HOST:
    Stream << "CG type: update_host\\n";
    break;
  case detail::CG::FILL:
    Stream << "CG type: fill\\n";
    break;
  case detail::CG::COPY_ACC_TO_ACC:
    Stream << "CG type: copy acc to acc\\n";
    break;
  case detail::CG::COPY_ACC_TO_PTR:
    Stream << "CG type: copy acc to ptr\\n";
    break;
  case detail::CG::COPY_PTR_TO_ACC:
    Stream << "CG type: copy ptr to acc\\n";
    break;
  }

  Stream << "\"];" << std::endl;

  for (const auto &Dep : MDeps) {
    Stream << "  \"" << this << "\" -> \"" << Dep.MDepCommand << "\""
           << " [ label = \"Access mode: "
           << accessModeToString(Dep.MReq->MAccessMode) << "\\n"
           << "MemObj: " << Dep.MReq->MSYCLMemObj << " \" ]" << std::endl;
  }
}

// SYCL has a parallel_for_work_group variant where the only NDRange
// characteristics set by a user is the number of work groups. This does not map
// to the OpenCL clEnqueueNDRangeAPI, which requires global work size to be set
// as well. This function determines local work size based on the device
// characteristics and the number of work groups requested by the user, then
// calculates the global work size.
// SYCL specification (from 4.8.5.3):
// The member function handler::parallel_for_work_group is parameterized by the
// number of work - groups, such that the size of each group is chosen by the
// runtime, or by the number of work - groups and number of work - items for
// users who need more control.
static void adjustNDRangePerKernel(NDRDescT &NDR, RT::PiKernel Kernel,
                                   RT::PiDevice Device) {
  if (NDR.GlobalSize[0] != 0)
    return; // GlobalSize is set - no need to adjust
  // check the prerequisites:
  assert(NDR.NumWorkGroups[0] != 0 && NDR.LocalSize[0] == 0);
  // TODO might be good to cache this info together with the kernel info to
  // avoid get_kernel_work_group_info on every kernel run
  range<3> WGSize = get_kernel_work_group_info<
      range<3>,
      cl::sycl::info::kernel_work_group::compile_work_group_size>::_(Kernel,
                                                                     Device);

  if (WGSize[0] == 0) {
    // kernel does not request specific workgroup shape - set one
    // TODO maximum work group size as the local size might not be the best
    //      choice for CPU or FPGA devices
    size_t WGSize1D = get_kernel_work_group_info<
        size_t, cl::sycl::info::kernel_work_group::work_group_size>::_(Kernel,
                                                                       Device);
    assert(WGSize1D != 0);
    // TODO implement better default for 2D/3D case:
    WGSize = {WGSize1D, 1, 1};
  }
  NDR.set(NDR.Dims, nd_range<3>(NDR.NumWorkGroups * WGSize, WGSize));
}

cl_int ExecCGCommand::enqueueImp() {
  std::vector<RT::PiEvent> RawEvents =
      Command::prepareEvents(detail::getSyclObjImpl(MQueue->get_context()));

  RT::PiEvent &Event = MEvent->getHandleRef();

  switch (MCommandGroup->getType()) {

  case CG::CGTYPE::UPDATE_HOST: {
    assert(!"Update host should be handled by the Scheduler.");
    throw runtime_error("Update host should be handled by the Scheduler.");
  }
  case CG::CGTYPE::COPY_ACC_TO_PTR: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *Req = (Requirement *)Copy->getSrc();
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    MemoryManager::copy(
        AllocaCmd->getSYCLMemObj(), AllocaCmd->getMemAllocation(), MQueue,
        Req->MDims, Req->MMemoryRange, Req->MAccessRange, Req->MOffset,
        Req->MElemSize, Copy->getDst(),
        Scheduler::getInstance().getDefaultHostQueue(), Req->MDims,
        Req->MAccessRange, Req->MAccessRange, /*DstOffset=*/{0, 0, 0},
        Req->MElemSize, std::move(RawEvents), MUseExclusiveQueue, Event);
    return CL_SUCCESS;
  }
  case CG::CGTYPE::COPY_PTR_TO_ACC: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *Req = (Requirement *)(Copy->getDst());
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    Scheduler::getInstance().getDefaultHostQueue();

    MemoryManager::copy(
        AllocaCmd->getSYCLMemObj(), Copy->getSrc(),
        Scheduler::getInstance().getDefaultHostQueue(), Req->MDims,
        Req->MAccessRange, Req->MAccessRange, /*SrcOffset*/ {0, 0, 0},
        Req->MElemSize, AllocaCmd->getMemAllocation(), MQueue, Req->MDims,
        Req->MMemoryRange, Req->MAccessRange, Req->MOffset, Req->MElemSize,
        std::move(RawEvents), MUseExclusiveQueue, Event);

    return CL_SUCCESS;
  }
  case CG::CGTYPE::COPY_ACC_TO_ACC: {
    CGCopy *Copy = (CGCopy *)MCommandGroup.get();
    Requirement *ReqSrc = (Requirement *)(Copy->getSrc());
    Requirement *ReqDst = (Requirement *)(Copy->getDst());

    AllocaCommandBase *AllocaCmdSrc = getAllocaForReq(ReqSrc);
    AllocaCommandBase *AllocaCmdDst = getAllocaForReq(ReqDst);

    MemoryManager::copy(
        AllocaCmdSrc->getSYCLMemObj(), AllocaCmdSrc->getMemAllocation(), MQueue,
        ReqSrc->MDims, ReqSrc->MMemoryRange, ReqSrc->MAccessRange,
        ReqSrc->MOffset, ReqSrc->MElemSize, AllocaCmdDst->getMemAllocation(),
        MQueue, ReqDst->MDims, ReqDst->MMemoryRange, ReqDst->MAccessRange,
        ReqDst->MOffset, ReqDst->MElemSize, std::move(RawEvents),
        MUseExclusiveQueue, Event);
    return CL_SUCCESS;
  }
  case CG::CGTYPE::FILL: {
    CGFill *Fill = (CGFill *)MCommandGroup.get();
    Requirement *Req = (Requirement *)(Fill->getReqToFill());
    AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);

    MemoryManager::fill(AllocaCmd->getSYCLMemObj(),
                        AllocaCmd->getMemAllocation(), MQueue,
                        Fill->MPattern.size(), Fill->MPattern.data(),
                        Req->MDims, Req->MMemoryRange, Req->MAccessRange,
                        Req->MOffset, Req->MElemSize, std::move(RawEvents),
                        Event);
    return CL_SUCCESS;
  }
  case CG::CGTYPE::KERNEL: {
    CGExecKernel *ExecKernel = (CGExecKernel *)MCommandGroup.get();

    NDRDescT &NDRDesc = ExecKernel->MNDRDesc;

    if (MQueue->is_host()) {
      for (ArgDesc &Arg : ExecKernel->MArgs)
        if (kernel_param_kind_t::kind_accessor == Arg.MType) {
          Requirement *Req = (Requirement *)(Arg.MPtr);
          AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);
          Req->MData = AllocaCmd->getMemAllocation();
        }
      if (!RawEvents.empty())
        PI_CALL(RT::piEventsWait(RawEvents.size(), &RawEvents[0]));
      ExecKernel->MHostKernel->call(NDRDesc);
      return CL_SUCCESS;
    }

    // Run OpenCL kernel
    sycl::context Context = MQueue->get_context();
    RT::PiKernel Kernel = nullptr;

    if (nullptr != ExecKernel->MSyclKernel) {
      assert(ExecKernel->MSyclKernel->get_context() == Context);
      Kernel = ExecKernel->MSyclKernel->getHandleRef();
    } else
      Kernel = detail::ProgramManager::getInstance().getOrCreateKernel(
          ExecKernel->MOSModuleHandle, Context, ExecKernel->MKernelName);

    bool usesUSM = false;
    for (ArgDesc &Arg : ExecKernel->MArgs) {
      switch (Arg.MType) {
      case kernel_param_kind_t::kind_accessor: {
        Requirement *Req = (Requirement *)(Arg.MPtr);
        AllocaCommandBase *AllocaCmd = getAllocaForReq(Req);
        cl_mem MemArg = (cl_mem)AllocaCmd->getMemAllocation();

        PI_CALL(RT::piKernelSetArg(
            Kernel, Arg.MIndex, sizeof(cl_mem), &MemArg));
        break;
      }
      case kernel_param_kind_t::kind_std_layout: {
        PI_CALL(RT::piKernelSetArg(
            Kernel, Arg.MIndex, Arg.MSize, Arg.MPtr));
        break;
      }
      case kernel_param_kind_t::kind_sampler: {
        sampler *SamplerPtr = (sampler *)Arg.MPtr;
        RT::PiSampler Sampler =
            detail::getSyclObjImpl(*SamplerPtr)->getOrCreateSampler(Context);
        PI_CALL(RT::piKernelSetArg(
            Kernel, Arg.MIndex, sizeof(cl_sampler), &Sampler));
        break;
      }
      case kernel_param_kind_t::kind_pointer:  {
        // TODO: Change to PI
        usesUSM = true;
        auto PtrToPtr = reinterpret_cast<intptr_t*>(Arg.MPtr);
        auto DerefPtr = reinterpret_cast<void*>(*PtrToPtr);
        auto theKernel = pi::cast<cl_kernel>(Kernel);
        CHECK_OCL_CODE(clSetKernelArgMemPointerINTEL(theKernel, Arg.MIndex, DerefPtr));
        break;
      }
      default:
        assert(!"Unhandled");
      }
    }

    adjustNDRangePerKernel(NDRDesc, Kernel,
                           detail::getSyclObjImpl(
                               MQueue->get_device())->getHandleRef());

    // TODO: Replace CL with PI
    auto clusm = GetCLUSM();
    if (usesUSM && clusm) {
      cl_bool t = CL_TRUE;
      auto theKernel = pi::cast<cl_kernel>(Kernel);
      // Enable USM Indirect Access for Kernels
      if (clusm->useCLUSM()) {
        CHECK_OCL_CODE(clusm->setKernelExecInfo(
            theKernel, CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL,
            sizeof(cl_bool), &t));
        CHECK_OCL_CODE(clusm->setKernelExecInfo(
            theKernel, CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
            sizeof(cl_bool), &t));
        CHECK_OCL_CODE(clusm->setKernelExecInfo(
            theKernel, CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL,
            sizeof(cl_bool), &t));

        // This passes all the allocations we've tracked as SVM Pointers
        CHECK_OCL_CODE(clusm->setKernelIndirectUSMExecInfo(
            pi::cast<cl_command_queue>(MQueue->getHandleRef()), theKernel));
      } else if (clusm->isInitialized()) {
        // Sanity check that nothing went wrong setting up clusm
        CHECK_OCL_CODE(clSetKernelExecInfo(
            theKernel, CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL,
            sizeof(cl_bool), &t));
        CHECK_OCL_CODE(clSetKernelExecInfo(
            theKernel, CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
            sizeof(cl_bool), &t));
        CHECK_OCL_CODE(clSetKernelExecInfo(
            theKernel, CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL,
            sizeof(cl_bool), &t));
      }
    }

    PI_CALL(RT::piEnqueueKernelLaunch(
        MQueue->getHandleRef(), Kernel, NDRDesc.Dims, &NDRDesc.GlobalOffset[0],
        &NDRDesc.GlobalSize[0],
        NDRDesc.LocalSize[0] ? &NDRDesc.LocalSize[0] : nullptr,
        RawEvents.size(),
        RawEvents.empty() ? nullptr : &RawEvents[0], &Event));

    return PI_SUCCESS;
  }
  }

  assert(!"CG type not implemented");
  throw runtime_error("CG type not implemented.");
}

} // namespace detail
} // namespace sycl
} // namespace cl
