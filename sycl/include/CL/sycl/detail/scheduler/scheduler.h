//==----------- scheduler.h ------------------------------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/scheduler/commands.h>
#include <CL/sycl/detail/scheduler/requirements.h>
#include <CL/sycl/device.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/nd_range.hpp>

#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>

namespace cl {
namespace sycl {
// Forward declaration
template <typename dataT, int dimensions, access::mode accessMode,
          access::target accessTarget, access::placeholder isPlaceholder>
class accessor;

namespace detail {
class queue_impl;
}
using QueueImplPtr = std::shared_ptr<detail::queue_impl>;

namespace simple_scheduler {

class Node {
public:
  Node(QueueImplPtr Queue) : m_Queue(std::move(Queue)) {}

  Node(Node &&RHS)
      : m_Bufs(std::move(RHS.m_Bufs)),
        m_InteropArgs(std::move(RHS.m_InteropArgs)),
        m_Kernel(std::move(RHS.m_Kernel)), m_Queue(std::move(RHS.m_Queue)),
        m_NextOCLIndex(RHS.m_NextOCLIndex) {}

  // Adds a buffer requirement for this node.
  template <access::mode Mode, access::target Target, typename T,
            int Dimensions, typename AllocatorT>
  void addBufRequirement(detail::buffer_impl<T, Dimensions, AllocatorT> &Buf);

  // Adds an accessor requirement for this node.
  template <typename dataT, int dimensions, access::mode accessMode,
            access::target accessTarget = access::target::global_buffer,
            access::placeholder isPlaceholder = access::placeholder::false_t>
  void addAccRequirement(accessor<dataT, dimensions, accessMode, accessTarget,
                                  isPlaceholder> &&Acc,
                         int argIndex);

  // Adds a kernel to this node, maps to single task.
  template <typename KernelType>
  void addKernel(const std::string &KernelName, const int KernelArgsNum,
                 const detail::kernel_param_desc_t *KernelArgs,
                 KernelType KernelFunc, cl_kernel ClKernel = nullptr);

  // Adds kernel to this node, maps on range parallel for.
  template <typename KernelType, int Dimensions, typename KernelArgType>
  void addKernel(const std::string &KernelName, const int KernelArgsNum,
                 const detail::kernel_param_desc_t *KernelArgs,
                 KernelType KernelFunc, range<Dimensions> NumWorkItems,
                 cl_kernel ClKernel = nullptr);

  // Adds kernel to this node, maps on range parallel for with offset.
  template <typename KernelType, int Dimensions, typename KernelArgType>
  void addKernel(const std::string &KernelName, const int KernelArgsNum,
                 const detail::kernel_param_desc_t *KernelArgs,
                 KernelType KernelFunc, range<Dimensions> NumWorkItems,
                 id<Dimensions> WorkItemOffset, cl_kernel ClKernel = nullptr);

  // Adds kernel to this node, maps on nd_range parallel for.
  template <typename KernelType, int Dimensions>
  void addKernel(const std::string &KernelName, const int KernelArgsNum,
                 const detail::kernel_param_desc_t *KernelArgs,
                 KernelType KernelFunc, nd_range<Dimensions> ExecutionRange,
                 cl_kernel ClKernel = nullptr);

  // Adds explicit memory operation to this node, maps on handler fill method
  template <typename T, int Dimensions, access::mode mode, access::target tgt,
            access::placeholder isPlaceholder = access::placeholder::false_t>
  void addExplicitMemOp(accessor<T, Dimensions, mode, tgt, isPlaceholder> &Dest,
                        T Src);

  // Adds explicit memory operation to this node, maps on handler copy method
  template <
      typename T_src, int dim_src, access::mode mode_src,
      access::target tgt_src, typename T_dest, int dim_dest,
      access::mode mode_dest, access::target tgt_dest,
      access::placeholder isPlaceholder_src = access::placeholder::false_t,
      access::placeholder isPlaceholder_dest = access::placeholder::false_t>
  void addExplicitMemOp(
      accessor<T_src, dim_src, mode_src, tgt_src, isPlaceholder_src> Src,
      accessor<T_dest, dim_dest, mode_dest, tgt_dest, isPlaceholder_dest> Dest);

  std::set<BufferReqPtr, classcomp> &getRequirements() { return m_Bufs; }

  void addInteropArg(shared_ptr_class<void> Ptr, size_t Size, int ArgIndex,
                     BufferReqPtr BufReq = nullptr);

  std::vector<InteropArg> &getInteropArgs() { return m_InteropArgs; }

  CommandPtr getKernel() { return m_Kernel; }

  QueueImplPtr getQueue() { return m_Queue; }

private:
  // Contains buffer requirements for this node.
  std::set<BufferReqPtr, classcomp> m_Bufs;
  // Contains arguments set via interoperability methods
  std::vector<InteropArg> m_InteropArgs;
  // Represent execute kernel command.
  CommandPtr m_Kernel;

  // SYCL queue for current command group.
  QueueImplPtr m_Queue;

  // WORKAROUND. Id for mapping OpenCL buffer to OpenCL kernel argument.
  size_t m_NextOCLIndex = 0;
};

class Scheduler {
public:
  // Adds copying of the specified buffer_impl and waits for completion.
  template <access::mode Mode, access::target Target, typename T,
            int Dimensions, typename AllocatorT>
  void copyBack(detail::buffer_impl<T, Dimensions, AllocatorT> &Buf);

  // Updates host data of the specified buffer_impl
  template <access::mode Mode, access::target Target, typename T,
            int Dimensions, typename AllocatorT>
  void updateHost(detail::buffer_impl<T, Dimensions, AllocatorT> &Buf,
                  cl::sycl::event &Event);

  // Updates host data of the specified accessor
  template <typename T, int Dimensions, access::mode mode, access::target tgt,
            access::placeholder isPlaceholder>
  void updateHost(accessor<T, Dimensions, mode, tgt, isPlaceholder> &Acc,
                  cl::sycl::event &Event);

  // Frees the specified buffer_impl.
  template <typename T, int Dimensions, typename AllocatorT>
  void removeBuffer(detail::buffer_impl<T, Dimensions, AllocatorT> &Buf);

  // Waits for the event passed.
  void waitForEvent(EventImplPtr Event);

  // Calls asynchronous handler for the passed event Event
  // and for those other events that Event depends on.
  void throwForEventRecursive(EventImplPtr Event);

  // Adds new node to graph, creating an Alloca and MemMove commands if
  // needed.
  cl::sycl::event addNode(Node NewNode);

  void print(std::ostream &Stream) const;
  void printDot(std::ostream &Stream) const;
  void dump() const { print(std::cout); }

  void dumpGraph() const {
    std::fstream GraphDot("graph.dot", std::ios::out);
    printDot(GraphDot);
  }

  void dumpGraphForCommand(CommandPtr Cmd) const;

  void optimize() { parallelReadOpt(); }

  // Converts the following:
  //
  //  =========    =========     =========
  // | kernel1 |<-| kernel2 |<--| kernel3 |
  // | write A |  | read A  |   | read A  |
  //  =========    =========     =========
  //
  // to: ---------------------------
  //     \/                        |
  //  =========    =========     =========
  // | kernel1 |<-| kernel2 |   | kernel3 |
  // | write A |  | read A  |   | read A  |
  //  =========    =========     =========
  //
  void parallelReadOpt();

  static Scheduler &getInstance();

  enum DumpOptions { Text = 0, WholeGraph = 1, RunGraph = 2 };
  bool getDumpFlagValue(DumpOptions DumpOption);

  // Recursively walks through the dependencies and initializes
  // the given EventsSet with the events that the Event
  // waits for. The unordered_set is used to collect unuque events,
  // and the unordered_set is convenient as it does not need operator<().
  void getDepEventsRecursive(std::unordered_set<cl::sycl::event> &EventsSet,
                             EventImplPtr Event);
protected:
  // TODO: Add releasing of OpenCL buffers.

  void enqueueAndWaitForCommand(CommandPtr Cmd);

  // Enqueues Cmd command and all its dependencies.
  cl::sycl::event EnqueueCommand(CommandPtr Cmd);

  cl::sycl::event dispatch(CommandPtr Cmd);

  // Recursively generates dot records for the command passed and all that the
  // command depends on.
  void printGraphForCommand(CommandPtr Cmd, std::ostream &Stream) const;
private:
  Scheduler();
  ~Scheduler();
  std::array<unsigned char, 3> m_DumpOptions;
  // Buffer that represents evolution of buffers - actions that is added
  // for each buffer.
  std::map<BufferReqPtr, std::vector<CommandPtr>, classcomp> m_BuffersEvolution;
  // Events for tracking execution of kernels without requirements
  std::vector<EventImplPtr> m_EventsWithoutRequirements;
  // TODO: At some point of time we should remove already processed commands.
  // But we have to be sure that nobody will references them(thru events).

  Scheduler(Scheduler const &) = delete;
  Scheduler &operator=(Scheduler const &) = delete;

  // Returns the pointer to the command associated with the given event,
  // or nullptr if none is found.
  CommandPtr getCmdForEvent(EventImplPtr Event);

  // Basically it is the helper method for throwForEventRecursive() now.
  // It calls async handler for the command Cmd and those other
  // commands that Cmd depends on.
  void throwForCmdRecursive(std::shared_ptr<Command> Cmd);
};

} // namespace simple_scheduler
} // namespace sycl
} // namespace cl
