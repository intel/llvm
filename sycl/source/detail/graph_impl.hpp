//==--------- graph_impl.hpp --- SYCL graph extension ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/cg_types.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/handler.hpp>

#include <detail/accessor_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/kernel_impl.hpp>

#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <list>
#include <set>
#include <shared_mutex>

namespace sycl {
inline namespace _V1 {

namespace detail {
class SYCLMemObjT;
}

namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {

/// Implementation of node class from SYCL_EXT_ONEAPI_GRAPH.
class node_impl {
public:
  /// List of successors to this node.
  std::vector<std::weak_ptr<node_impl>> MSuccessors;
  /// List of predecessors to this node.
  ///
  /// Using weak_ptr here to prevent circular references between nodes.
  std::vector<std::weak_ptr<node_impl>> MPredecessors;
  /// Type of the command-group for the node.
  sycl::detail::CG::CGTYPE MCGType = sycl::detail::CG::None;
  /// Command group object which stores all args etc needed to enqueue the node
  std::unique_ptr<sycl::detail::CG> MCommandGroup;

  /// Used for tracking visited status during cycle checks.
  bool MVisited = false;

  /// Partition number needed to assign a Node to a a partition.
  /// Note : This number is only used during the partitionning process and
  /// cannot be used to find out the partion of a node outside of this process.
  int MPartitionNum = -1;

  /// Add successor to the node.
  /// @param Node Node to add as a successor.
  /// @param Prev Predecessor to \p node being added as successor.
  ///
  /// \p Prev should be a shared_ptr to an instance of this object, but can't
  /// use a raw \p this pointer, so the extra \Prev parameter is passed.
  void registerSuccessor(const std::shared_ptr<node_impl> &Node,
                         const std::shared_ptr<node_impl> &Prev) {
    if (std::find_if(MSuccessors.begin(), MSuccessors.end(),
                     [Node](const std::weak_ptr<node_impl> &Ptr) {
                       return Ptr.lock() == Node;
                     }) != MSuccessors.end()) {
      return;
    }
    MSuccessors.push_back(Node);
    Node->registerPredecessor(Prev);
  }

  /// Add predecessor to the node.
  /// @param Node Node to add as a predecessor.
  void registerPredecessor(const std::shared_ptr<node_impl> &Node) {
    if (std::find_if(MPredecessors.begin(), MPredecessors.end(),
                     [&Node](const std::weak_ptr<node_impl> &Ptr) {
                       return Ptr.lock() == Node;
                     }) != MPredecessors.end()) {
      return;
    }
    MPredecessors.push_back(Node);
  }

  /// Construct an empty node.
  node_impl() {}

  /// Construct a node representing a command-group.
  /// @param CGType Type of the command-group.
  /// @param CommandGroup The CG which stores the command information for this
  /// node.
  node_impl(sycl::detail::CG::CGTYPE CGType,
            std::unique_ptr<sycl::detail::CG> &&CommandGroup)
      : MCGType(CGType), MCommandGroup(std::move(CommandGroup)) {}

  /// Checks if this node has a given requirement.
  /// @param Requirement Requirement to lookup.
  /// @return True if \p Requirement is present in node, false otherwise.
  bool hasRequirement(sycl::detail::AccessorImplHost *IncomingReq) {
    for (sycl::detail::AccessorImplHost *CurrentReq :
         MCommandGroup->getRequirements()) {
      if (IncomingReq->MSYCLMemObj == CurrentReq->MSYCLMemObj) {
        return true;
      }
    }
    return false;
  }

  /// Query if this is an empty node.
  /// Barrier nodes are also considered empty nodes since they do not embed any
  /// workload but only dependencies
  /// @return True if this is an empty node, false otherwise.
  bool isEmpty() const {
    return ((MCGType == sycl::detail::CG::None) ||
            (MCGType == sycl::detail::CG::Barrier));
  }

  /// Get a deep copy of this node's command group
  /// @return A unique ptr to the new command group object.
  std::unique_ptr<sycl::detail::CG> getCGCopy() const {
    switch (MCGType) {
    case sycl::detail::CG::Kernel:
      return createCGCopy<sycl::detail::CGExecKernel>();
    case sycl::detail::CG::CopyAccToPtr:
    case sycl::detail::CG::CopyPtrToAcc:
    case sycl::detail::CG::CopyAccToAcc:
      return createCGCopy<sycl::detail::CGCopy>();
    case sycl::detail::CG::Fill:
      return createCGCopy<sycl::detail::CGFill>();
    case sycl::detail::CG::UpdateHost:
      return createCGCopy<sycl::detail::CGUpdateHost>();
    case sycl::detail::CG::CopyUSM:
      return createCGCopy<sycl::detail::CGCopyUSM>();
    case sycl::detail::CG::FillUSM:
      return createCGCopy<sycl::detail::CGFillUSM>();
    case sycl::detail::CG::PrefetchUSM:
      return createCGCopy<sycl::detail::CGPrefetchUSM>();
    case sycl::detail::CG::AdviseUSM:
      return createCGCopy<sycl::detail::CGAdviseUSM>();
    case sycl::detail::CG::Copy2DUSM:
      return createCGCopy<sycl::detail::CGCopy2DUSM>();
    case sycl::detail::CG::Fill2DUSM:
      return createCGCopy<sycl::detail::CGFill2DUSM>();
    case sycl::detail::CG::Memset2DUSM:
      return createCGCopy<sycl::detail::CGMemset2DUSM>();
    case sycl::detail::CG::CodeplayHostTask: {
      // The unique_ptr to the `sycl::detail::HostTask` in the HostTask CG
      // prevents from copying the CG.
      // We overcome this restriction by creating a new CG with the same data.
      auto CommandGroupPtr =
          static_cast<sycl::detail::CGHostTask *>(MCommandGroup.get());
      sycl::detail::HostTask HostTask = *CommandGroupPtr->MHostTask.get();
      auto HostTaskUPtr = std::make_unique<sycl::detail::HostTask>(HostTask);

      sycl::detail::CG::StorageInitHelper Data(
          CommandGroupPtr->getArgsStorage(), CommandGroupPtr->getAccStorage(),
          CommandGroupPtr->getSharedPtrStorage(),
          CommandGroupPtr->getRequirements(), CommandGroupPtr->getEvents());

      sycl::detail::code_location Loc(CommandGroupPtr->MFileName.data(),
                                      CommandGroupPtr->MFunctionName.data(),
                                      CommandGroupPtr->MLine,
                                      CommandGroupPtr->MColumn);

      return std::make_unique<sycl::detail::CGHostTask>(
          sycl::detail::CGHostTask(
              std::move(HostTaskUPtr), CommandGroupPtr->MQueue,
              CommandGroupPtr->MContext, CommandGroupPtr->MArgs, Data,
              CommandGroupPtr->getType(), Loc));
    }
    case sycl::detail::CG::Barrier:
    case sycl::detail::CG::BarrierWaitlist:
      return createCGCopy<sycl::detail::CGBarrier>();
    case sycl::detail::CG::CopyToDeviceGlobal:
      return createCGCopy<sycl::detail::CGCopyToDeviceGlobal>();
    case sycl::detail::CG::CopyFromDeviceGlobal:
      return createCGCopy<sycl::detail::CGCopyFromDeviceGlobal>();
    case sycl::detail::CG::ReadWriteHostPipe:
      return createCGCopy<sycl::detail::CGReadWriteHostPipe>();
    case sycl::detail::CG::CopyImage:
      return createCGCopy<sycl::detail::CGCopyImage>();
    case sycl::detail::CG::SemaphoreSignal:
      return createCGCopy<sycl::detail::CGSemaphoreSignal>();
    case sycl::detail::CG::SemaphoreWait:
      return createCGCopy<sycl::detail::CGSemaphoreWait>();
    case sycl::detail::CG::ExecCommandBuffer:
      assert(false &&
             "Error: Command graph submission should not be a node in a graph");
      break;
    case sycl::detail::CG::None:
      assert(false &&
             "Error: Empty nodes should not be enqueue to a command buffer");
      break;
    }
    return nullptr;
  }

  /// Tests if the caller is similar to Node, this is only used for testing.
  /// @param Node The node to check for similarity.
  /// @param CompareContentOnly Skip comparisons related to graph structure,
  /// compare only the type and command groups of the nodes
  /// @return True if the two nodes are similar
  bool isSimilar(const std::shared_ptr<node_impl> &Node,
                 bool CompareContentOnly = false) const {
    if (!CompareContentOnly) {
      if (MSuccessors.size() != Node->MSuccessors.size())
        return false;

      if (MPredecessors.size() != Node->MPredecessors.size())
        return false;
    }
    if (MCGType != Node->MCGType)
      return false;

    switch (MCGType) {
    case sycl::detail::CG::CGTYPE::Kernel: {
      sycl::detail::CGExecKernel *ExecKernelA =
          static_cast<sycl::detail::CGExecKernel *>(MCommandGroup.get());
      sycl::detail::CGExecKernel *ExecKernelB =
          static_cast<sycl::detail::CGExecKernel *>(Node->MCommandGroup.get());
      return ExecKernelA->MKernelName.compare(ExecKernelB->MKernelName) == 0;
    }
    case sycl::detail::CG::CGTYPE::CopyUSM: {
      sycl::detail::CGCopyUSM *CopyA =
          static_cast<sycl::detail::CGCopyUSM *>(MCommandGroup.get());
      sycl::detail::CGCopyUSM *CopyB =
          static_cast<sycl::detail::CGCopyUSM *>(Node->MCommandGroup.get());
      return (CopyA->getSrc() == CopyB->getSrc()) &&
             (CopyA->getDst() == CopyB->getDst()) &&
             (CopyA->getLength() == CopyB->getLength());
    }
    case sycl::detail::CG::CGTYPE::CopyAccToAcc:
    case sycl::detail::CG::CGTYPE::CopyAccToPtr:
    case sycl::detail::CG::CGTYPE::CopyPtrToAcc: {
      sycl::detail::CGCopy *CopyA =
          static_cast<sycl::detail::CGCopy *>(MCommandGroup.get());
      sycl::detail::CGCopy *CopyB =
          static_cast<sycl::detail::CGCopy *>(Node->MCommandGroup.get());
      return (CopyA->getSrc() == CopyB->getSrc()) &&
             (CopyA->getDst() == CopyB->getDst());
    }
    default:
      assert(false && "Unexpected command group type!");
      return false;
    }
  }

  /// Recursive Depth first traversal of linked nodes.
  /// to print node information and connection to Stream.
  /// @param Stream Where to print node information.
  /// @param Visited Vector of the already visited nodes.
  /// @param Verbose If true, print additional information about the nodes such
  /// as kernel args or memory access where applicable.
  void printDotRecursive(std::fstream &Stream,
                         std::vector<node_impl *> &Visited, bool Verbose) {
    // if Node has been already visited, we skip it
    if (std::find(Visited.begin(), Visited.end(), this) != Visited.end())
      return;

    Visited.push_back(this);

    printDotCG(Stream, Verbose);
    for (const auto &Dep : MPredecessors) {
      auto NodeDep = Dep.lock();
      Stream << "  \"" << NodeDep.get() << "\" -> \"" << this << "\""
             << std::endl;
    }

    for (std::weak_ptr<node_impl> Succ : MSuccessors) {
      if (MPartitionNum == Succ.lock()->MPartitionNum)
        Succ.lock()->printDotRecursive(Stream, Visited, Verbose);
    }
  }

private:
  /// Prints Node information to Stream.
  /// @param Stream Where to print the Node information
  /// @param Verbose If true, print additional information about the nodes such
  /// as kernel args or memory access where applicable.
  void printDotCG(std::ostream &Stream, bool Verbose) {
    Stream << "\"" << this << "\" [style=bold, label=\"";

    Stream << "ID = " << this << "\\n";
    Stream << "TYPE = ";

    switch (MCGType) {
    case sycl::detail::CG::CGTYPE::None:
      Stream << "None \\n";
      break;
    case sycl::detail::CG::CGTYPE::Kernel: {
      Stream << "CGExecKernel \\n";
      sycl::detail::CGExecKernel *Kernel =
          static_cast<sycl::detail::CGExecKernel *>(MCommandGroup.get());
      Stream << "NAME = " << Kernel->MKernelName << "\\n";
      if (Verbose) {
        Stream << "ARGS = \\n";
        for (size_t i = 0; i < Kernel->MArgs.size(); i++) {
          auto Arg = Kernel->MArgs[i];
          std::string Type = "Undefined";
          if (Arg.MType == sycl::detail::kernel_param_kind_t::kind_accessor) {
            Type = "Accessor";
          } else if (Arg.MType ==
                     sycl::detail::kernel_param_kind_t::kind_std_layout) {
            Type = "STD_Layout";
          } else if (Arg.MType ==
                     sycl::detail::kernel_param_kind_t::kind_sampler) {
            Type = "Sampler";
          } else if (Arg.MType ==
                     sycl::detail::kernel_param_kind_t::kind_pointer) {
            Type = "Pointer";
          } else if (Arg.MType == sycl::detail::kernel_param_kind_t::
                                      kind_specialization_constants_buffer) {
            Type = "Specialization Constants Buffer";
          } else if (Arg.MType ==
                     sycl::detail::kernel_param_kind_t::kind_stream) {
            Type = "Stream";
          } else if (Arg.MType ==
                     sycl::detail::kernel_param_kind_t::kind_invalid) {
            Type = "Invalid";
          }
          Stream << i << ") Type: " << Type << " Ptr: " << Arg.MPtr << "\\n";
        }
      }
      break;
    }
    case sycl::detail::CG::CGTYPE::CopyAccToPtr:
      Stream << "CGCopy Device-to-Host \\n";
      if (Verbose) {
        sycl::detail::CGCopy *Copy =
            static_cast<sycl::detail::CGCopy *>(MCommandGroup.get());
        Stream << "Src: " << Copy->getSrc() << " Dst: " << Copy->getDst()
               << "\\n";
      }
      break;
    case sycl::detail::CG::CGTYPE::CopyPtrToAcc:
      Stream << "CGCopy Host-to-Device \\n";
      if (Verbose) {
        sycl::detail::CGCopy *Copy =
            static_cast<sycl::detail::CGCopy *>(MCommandGroup.get());
        Stream << "Src: " << Copy->getSrc() << " Dst: " << Copy->getDst()
               << "\\n";
      }
      break;
    case sycl::detail::CG::CGTYPE::CopyAccToAcc:
      Stream << "CGCopy Device-to-Device \\n";
      if (Verbose) {
        sycl::detail::CGCopy *Copy =
            static_cast<sycl::detail::CGCopy *>(MCommandGroup.get());
        Stream << "Src: " << Copy->getSrc() << " Dst: " << Copy->getDst()
               << "\\n";
      }
      break;
    case sycl::detail::CG::CGTYPE::Fill:
      Stream << "CGFill \\n";
      if (Verbose) {
        sycl::detail::CGFill *Fill =
            static_cast<sycl::detail::CGFill *>(MCommandGroup.get());
        Stream << "Ptr: " << Fill->MPtr << "\\n";
      }
      break;
    case sycl::detail::CG::CGTYPE::UpdateHost:
      Stream << "CGCUpdateHost \\n";
      if (Verbose) {
        sycl::detail::CGUpdateHost *Host =
            static_cast<sycl::detail::CGUpdateHost *>(MCommandGroup.get());
        Stream << "Ptr: " << Host->getReqToUpdate() << "\\n";
      }
      break;
    case sycl::detail::CG::CGTYPE::CopyUSM:
      Stream << "CGCopyUSM \\n";
      if (Verbose) {
        sycl::detail::CGCopyUSM *CopyUSM =
            static_cast<sycl::detail::CGCopyUSM *>(MCommandGroup.get());
        Stream << "Src: " << CopyUSM->getSrc() << " Dst: " << CopyUSM->getDst()
               << " Length: " << CopyUSM->getLength() << "\\n";
      }
      break;
    case sycl::detail::CG::CGTYPE::FillUSM:
      Stream << "CGFillUSM \\n";
      if (Verbose) {
        sycl::detail::CGFillUSM *FillUSM =
            static_cast<sycl::detail::CGFillUSM *>(MCommandGroup.get());
        Stream << "Dst: " << FillUSM->getDst()
               << " Length: " << FillUSM->getLength()
               << " Pattern: " << FillUSM->getFill() << "\\n";
      }
      break;
    case sycl::detail::CG::CGTYPE::PrefetchUSM:
      Stream << "CGPrefetchUSM \\n";
      if (Verbose) {
        sycl::detail::CGPrefetchUSM *Prefetch =
            static_cast<sycl::detail::CGPrefetchUSM *>(MCommandGroup.get());
        Stream << "Dst: " << Prefetch->getDst()
               << " Length: " << Prefetch->getLength() << "\\n";
      }
      break;
    case sycl::detail::CG::CGTYPE::AdviseUSM:
      Stream << "CGAdviseUSM \\n";
      if (Verbose) {
        sycl::detail::CGAdviseUSM *AdviseUSM =
            static_cast<sycl::detail::CGAdviseUSM *>(MCommandGroup.get());
        Stream << "Dst: " << AdviseUSM->getDst()
               << " Length: " << AdviseUSM->getLength() << "\\n";
      }
      break;
    case sycl::detail::CG::CGTYPE::CodeplayHostTask:
      Stream << "CGHostTask \\n";
      break;
    case sycl::detail::CG::CGTYPE::Barrier:
      Stream << "CGBarrier \\n";
      break;
    case sycl::detail::CG::CGTYPE::Copy2DUSM:
      Stream << "CGCopy2DUSM \\n";
      if (Verbose) {
        sycl::detail::CGCopy2DUSM *Copy2DUSM =
            static_cast<sycl::detail::CGCopy2DUSM *>(MCommandGroup.get());
        Stream << "Src:" << Copy2DUSM->getSrc()
               << " Dst: " << Copy2DUSM->getDst() << "\\n";
      }
      break;
    case sycl::detail::CG::CGTYPE::Fill2DUSM:
      Stream << "CGFill2DUSM \\n";
      if (Verbose) {
        sycl::detail::CGFill2DUSM *Fill2DUSM =
            static_cast<sycl::detail::CGFill2DUSM *>(MCommandGroup.get());
        Stream << "Dst: " << Fill2DUSM->getDst() << "\\n";
      }
      break;
    case sycl::detail::CG::CGTYPE::Memset2DUSM:
      Stream << "CGMemset2DUSM \\n";
      if (Verbose) {
        sycl::detail::CGMemset2DUSM *Memset2DUSM =
            static_cast<sycl::detail::CGMemset2DUSM *>(MCommandGroup.get());
        Stream << "Dst: " << Memset2DUSM->getDst() << "\\n";
      }
      break;
    case sycl::detail::CG::CGTYPE::ReadWriteHostPipe:
      Stream << "CGReadWriteHostPipe \\n";
      break;
    case sycl::detail::CG::CGTYPE::CopyToDeviceGlobal:
      Stream << "CGCopyToDeviceGlobal \\n";
      if (Verbose) {
        sycl::detail::CGCopyToDeviceGlobal *CopyToDeviceGlobal =
            static_cast<sycl::detail::CGCopyToDeviceGlobal *>(
                MCommandGroup.get());
        Stream << "Src: " << CopyToDeviceGlobal->getSrc()
               << " Dst: " << CopyToDeviceGlobal->getDeviceGlobalPtr() << "\\n";
      }
      break;
    case sycl::detail::CG::CGTYPE::CopyFromDeviceGlobal:
      Stream << "CGCopyFromDeviceGlobal \\n";
      if (Verbose) {
        sycl::detail::CGCopyFromDeviceGlobal *CopyFromDeviceGlobal =
            static_cast<sycl::detail::CGCopyFromDeviceGlobal *>(
                MCommandGroup.get());
        Stream << "Src: " << CopyFromDeviceGlobal->getDeviceGlobalPtr()
               << " Dst: " << CopyFromDeviceGlobal->getDest() << "\\n";
      }
      break;
    case sycl::detail::CG::CGTYPE::ExecCommandBuffer:
      Stream << "CGExecCommandBuffer \\n";
      break;
    default:
      Stream << "Other \\n";
      break;
    }
    Stream << "\"];" << std::endl;
  }

  /// Creates a copy of the node's CG by casting to it's actual type, then using
  /// that to copy construct and create a new unique ptr from that copy.
  /// @tparam CGT The derived type of the CG.
  /// @return A new unique ptr to the copied CG.
  template <typename CGT> std::unique_ptr<CGT> createCGCopy() const {
    return std::make_unique<CGT>(*static_cast<CGT *>(MCommandGroup.get()));
  }
};

class partition {
public:
  /// Constructor.
  partition() : MSchedule(), MPiCommandBuffers() {}

  /// List of root nodes.
  std::set<std::weak_ptr<node_impl>, std::owner_less<std::weak_ptr<node_impl>>>
      MRoots;
  /// Execution schedule of nodes in the graph.
  std::list<std::shared_ptr<node_impl>> MSchedule;
  /// Map of devices to command buffers.
  std::unordered_map<sycl::device, sycl::detail::pi::PiExtCommandBuffer>
      MPiCommandBuffers;

  /// @return True if the partition contains a host task
  bool isHostTask() const {
    return (MRoots.size() && ((*MRoots.begin()).lock()->MCGType ==
                              sycl::detail::CG::CGTYPE::CodeplayHostTask));
  }

  /// Add nodes to MSchedule.
  void schedule();
};

/// Implementation details of command_graph<modifiable>.
class graph_impl {
public:
  using ReadLock = std::shared_lock<std::shared_mutex>;
  using WriteLock = std::unique_lock<std::shared_mutex>;

  /// Protects all the fields that can be changed by class' methods.
  mutable std::shared_mutex MMutex;

  /// Constructor.
  /// @param SyclContext Context to use for graph.
  /// @param SyclDevice Device to create nodes with.
  /// @param PropList Optional list of properties.
  graph_impl(const sycl::context &SyclContext, const sycl::device &SyclDevice,
             const sycl::property_list &PropList = {})
      : MContext(SyclContext), MDevice(SyclDevice), MRecordingQueues(),
        MEventsMap(), MInorderQueueMap() {
    if (PropList.has_property<property::graph::no_cycle_check>()) {
      MSkipCycleChecks = true;
    }
    if (PropList
            .has_property<property::graph::assume_buffer_outlives_graph>()) {
      MAllowBuffers = true;
    }

    if (SyclDevice.get_info<
            ext::oneapi::experimental::info::device::graph_support>() ==
        graph_support_level::unsupported) {
      std::stringstream Stream;
      Stream << SyclDevice.get_backend();
      std::string BackendString = Stream.str();
      throw sycl::exception(
          sycl::make_error_code(errc::invalid),
          BackendString + " backend is not supported by SYCL Graph extension.");
    }
  }

  ~graph_impl();

  /// Remove node from list of root nodes.
  /// @param Root Node to remove from list of root nodes.
  void removeRoot(const std::shared_ptr<node_impl> &Root);

  /// Create a kernel node in the graph.
  /// @param CGType Type of the command-group.
  /// @param CommandGroup The CG which stores all information for this node.
  /// @param Dep Dependencies of the created node.
  /// @return Created node in the graph.
  std::shared_ptr<node_impl>
  add(sycl::detail::CG::CGTYPE CGType,
      std::unique_ptr<sycl::detail::CG> CommandGroup,
      const std::vector<std::shared_ptr<node_impl>> &Dep = {});

  /// Create a CGF node in the graph.
  /// @param Impl Graph implementation pointer to create a handler with.
  /// @param CGF Command-group function to create node with.
  /// @param Args Node arguments.
  /// @param Dep Dependencies of the created node.
  /// @return Created node in the graph.
  std::shared_ptr<node_impl>
  add(const std::shared_ptr<graph_impl> &Impl,
      std::function<void(handler &)> CGF,
      const std::vector<sycl::detail::ArgDesc> &Args,
      const std::vector<std::shared_ptr<node_impl>> &Dep = {});

  /// Create an empty node in the graph.
  /// @param Dep List of predecessor nodes.
  /// @return Created node in the graph.
  std::shared_ptr<node_impl>
  add(const std::vector<std::shared_ptr<node_impl>> &Dep = {});

  /// Create an empty node in the graph.
  /// @param Events List of events associated to this node.
  /// @return Created node in the graph.
  std::shared_ptr<node_impl>
  add(const std::vector<sycl::detail::EventImplPtr> Events);

  /// Add a queue to the set of queues which are currently recording to this
  /// graph.
  /// @param RecordingQueue Queue to add to set.
  void
  addQueue(const std::shared_ptr<sycl::detail::queue_impl> &RecordingQueue) {
    MRecordingQueues.insert(RecordingQueue);
  }

  /// Remove a queue from the set of queues which are currently recording to
  /// this graph.
  /// @param RecordingQueue Queue to remove from set.
  void
  removeQueue(const std::shared_ptr<sycl::detail::queue_impl> &RecordingQueue) {
    MRecordingQueues.erase(RecordingQueue);
  }

  /// Remove all queues which are recording to this graph, also sets all queues
  /// cleared back to the executing state.
  ///
  /// @return True if any queues were removed.
  bool clearQueues();

  /// Associate a sycl event with a node in the graph.
  /// @param EventImpl Event to associate with a node in map.
  /// @param NodeImpl Node to associate with event in map.
  void addEventForNode(std::shared_ptr<sycl::detail::event_impl> EventImpl,
                       std::shared_ptr<node_impl> NodeImpl) {
    MEventsMap[EventImpl] = NodeImpl;
  }

  /// Find the sycl event associated with a node.
  /// @param NodeImpl Node to find event for.
  /// @return Event associated with node.
  std::shared_ptr<sycl::detail::event_impl>
  getEventForNode(std::shared_ptr<node_impl> NodeImpl) const {
    ReadLock Lock(MMutex);
    if (auto EventImpl = std::find_if(
            MEventsMap.begin(), MEventsMap.end(),
            [NodeImpl](auto &it) { return it.second == NodeImpl; });
        EventImpl != MEventsMap.end()) {
      return EventImpl->first;
    }

    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "No event has been recorded for the specified graph node");
  }

  /// Duplicates and Adds sub-graph nodes from an executable graph to this
  /// graph.
  /// @param SubGraphExec sub-graph to add to the parent.
  /// @return An empty node is used to schedule dependencies on this sub-graph.
  std::shared_ptr<node_impl>
  addSubgraphNodes(const std::shared_ptr<exec_graph_impl> &SubGraphExec);

  /// Query for the context tied to this graph.
  /// @return Context associated with graph.
  sycl::context getContext() const { return MContext; }

  /// Query for the device tied to this graph.
  /// @return Device associated with graph.
  sycl::device getDevice() const { return MDevice; }

  /// List of root nodes.
  std::set<std::weak_ptr<node_impl>, std::owner_less<std::weak_ptr<node_impl>>>
      MRoots;

  /// Storage for all nodes contained within a graph. Nodes are connected to
  /// each other via weak_ptrs and so do not extend each other's lifetimes.
  /// This storage allows easy iteration over all nodes in the graph, rather
  /// than needing an expensive depth first search.
  std::vector<std::shared_ptr<node_impl>> MNodeStorage;

  /// Find the last node added to this graph from an in-order queue.
  /// @param Queue In-order queue to find the last node added to the graph from.
  /// @return Last node in this graph added from \p Queue recording, or empty
  /// shared pointer if none.
  std::shared_ptr<node_impl>
  getLastInorderNode(std::shared_ptr<sycl::detail::queue_impl> Queue) {
    std::weak_ptr<sycl::detail::queue_impl> QueueWeakPtr(Queue);
    if (0 == MInorderQueueMap.count(QueueWeakPtr)) {
      return {};
    }
    return MInorderQueueMap[QueueWeakPtr];
  }

  /// Track the last node added to this graph from an in-order queue.
  /// @param Queue In-order queue to register \p Node for.
  /// @param Node Last node that was added to this graph from \p Queue.
  void setLastInorderNode(std::shared_ptr<sycl::detail::queue_impl> Queue,
                          std::shared_ptr<node_impl> Node) {
    std::weak_ptr<sycl::detail::queue_impl> QueueWeakPtr(Queue);
    MInorderQueueMap[QueueWeakPtr] = Node;
  }

  /// Prints the contents of the graph to a text file in DOT format.
  /// @param FilePath Path to the output file.
  /// @param Verbose If true, print additional information about the nodes such
  /// as kernel args or memory access where applicable.
  void printGraphAsDot(const std::string FilePath, bool Verbose) const {
    /// Vector of nodes visited during the graph printing
    std::vector<node_impl *> VisitedNodes;

    std::fstream Stream(FilePath, std::ios::out);
    Stream << "digraph dot {" << std::endl;

    for (std::weak_ptr<node_impl> Node : MRoots)
      Node.lock()->printDotRecursive(Stream, VisitedNodes, Verbose);

    Stream << "}" << std::endl;

    Stream.close();
  }

  /// Make an edge between two nodes in the graph. Performs some mandatory
  /// error checks as well as an optional check for cycles introduced by making
  /// this edge.
  /// @param Src The source of the new edge.
  /// @param Dest The destination of the new edge.
  void makeEdge(std::shared_ptr<node_impl> Src,
                std::shared_ptr<node_impl> Dest);

  /// Throws an invalid exception if this function is called
  /// while a queue is recording commands to the graph.
  /// @param ExceptionMsg Message to append to the exception message
  void throwIfGraphRecordingQueue(const std::string ExceptionMsg) const {
    if (MRecordingQueues.size()) {
      throw sycl::exception(make_error_code(sycl::errc::invalid),
                            ExceptionMsg +
                                " cannot be called when a queue "
                                "is currently recording commands to a graph.");
    }
  }

  /// Recursively check successors of NodeA and NodeB to check they are similar.
  /// @param NodeA pointer to the first node for comparison
  /// @param NodeB pointer to the second node for comparison
  /// @return true is same structure found, false otherwise
  static bool checkNodeRecursive(const std::shared_ptr<node_impl> &NodeA,
                                 const std::shared_ptr<node_impl> &NodeB) {
    size_t FoundCnt = 0;
    for (std::weak_ptr<node_impl> &SuccA : NodeA->MSuccessors) {
      for (std::weak_ptr<node_impl> &SuccB : NodeB->MSuccessors) {
        if (NodeA->isSimilar(NodeB) &&
            checkNodeRecursive(SuccA.lock(), SuccB.lock())) {
          FoundCnt++;
          break;
        }
      }
    }
    if (FoundCnt != NodeA->MSuccessors.size()) {
      return false;
    }

    return true;
  }

  /// Checks if the graph_impl of Graph has a similar structure to
  /// the graph_impl of the caller.
  /// Graphs are considered similar if they have same numbers of nodes
  /// of the same type with similar predecessor and successor nodes (number and
  /// type). Two nodes are considered similar if they have the same
  /// command-group type. For command-groups of type "kernel", the "signature"
  /// of the kernel is also compared (i.e. the name of the command-group).
  /// @param Graph if reference to the graph to compare with.
  /// @param DebugPrint if set to true throw exception with additional debug
  /// information about the spotted graph differences.
  /// @return true if the two graphs are similar, false otherwise
  bool hasSimilarStructure(std::shared_ptr<detail::graph_impl> Graph,
                           bool DebugPrint = false) const {
    if (this == Graph.get())
      return true;

    if (MContext != Graph->MContext) {
      if (DebugPrint) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "MContext are not the same.");
      }
      return false;
    }

    if (MDevice != Graph->MDevice) {
      if (DebugPrint) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "MDevice are not the same.");
      }
      return false;
    }

    if (MEventsMap.size() != Graph->MEventsMap.size()) {
      if (DebugPrint) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "MEventsMap sizes are not the same.");
      }
      return false;
    }

    if (MInorderQueueMap.size() != Graph->MInorderQueueMap.size()) {
      if (DebugPrint) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "MInorderQueueMap sizes are not the same.");
      }
      return false;
    }

    if (MRoots.size() != Graph->MRoots.size()) {
      if (DebugPrint) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "MRoots sizes are not the same.");
      }
      return false;
    }

    size_t RootsFound = 0;
    for (std::weak_ptr<node_impl> NodeA : MRoots) {
      for (std::weak_ptr<node_impl> NodeB : Graph->MRoots) {
        auto NodeALocked = NodeA.lock();
        auto NodeBLocked = NodeB.lock();

        if (NodeALocked->isSimilar(NodeBLocked)) {
          if (checkNodeRecursive(NodeALocked, NodeBLocked)) {
            RootsFound++;
            break;
          }
        }
      }
    }

    if (RootsFound != MRoots.size()) {
      if (DebugPrint) {
        throw sycl::exception(sycl::make_error_code(errc::invalid),
                              "Root Nodes do NOT match.");
      }
      return false;
    }

    return true;
  }

  /// Returns the number of nodes in the Graph
  /// @return Number of nodes in the Graph
  size_t getNumberOfNodes() const { return MNodeStorage.size(); }

  /// Traverse the graph recursively to get the events associated with the
  /// output nodes of this graph.
  /// @return vector of events associated to exit nodes.
  std::vector<sycl::detail::EventImplPtr> getExitNodesEvents();

  /// Removes all Barrier nodes from the list of extra dependencies
  /// MExtraDependencies.
  /// @return vector of events associated to previous barrier nodes.
  std::vector<sycl::detail::EventImplPtr>
  removeBarriersFromExtraDependencies() {
    std::vector<sycl::detail::EventImplPtr> Events;
    for (auto It = MExtraDependencies.begin();
         It != MExtraDependencies.end();) {
      if ((*It)->MCGType == sycl::detail::CG::Barrier) {
        Events.push_back(getEventForNode(*It));
        It = MExtraDependencies.erase(It);
      } else {
        ++It;
      }
    }
    return Events;
  }

private:
  /// Iterate over the graph depth-first and run \p NodeFunc on each node.
  /// @param NodeFunc A function which receives as input a node in the graph to
  /// perform operations on as well as the stack of nodes encountered in the
  /// current path. The return value of this function determines whether an
  /// early exit is triggered, if true the depth-first search will end
  /// immediately and no further nodes will be visited.
  void
  searchDepthFirst(std::function<bool(std::shared_ptr<node_impl> &,
                                      std::deque<std::shared_ptr<node_impl>> &)>
                       NodeFunc);

  /// Check the graph for cycles by performing a depth-first search of the
  /// graph. If a node is visited more than once in a given path through the
  /// graph, a cycle is present and the search ends immediately.
  /// @return True if a cycle is detected, false if not.
  bool checkForCycles();

  /// Insert node into list of root nodes.
  /// @param Root Node to add to list of root nodes.
  void addRoot(const std::shared_ptr<node_impl> &Root);

  /// Adds nodes to the exit nodes of this graph.
  /// @param NodeList List of nodes from sub-graph in schedule order.
  /// @return An empty node is used to schedule dependencies on this sub-graph.
  std::shared_ptr<node_impl>
  addNodesToExits(const std::list<std::shared_ptr<node_impl>> &NodeList);

  /// Adds dependencies for a new node, if it has no deps it will be
  /// added as a root node.
  /// @param Node The node to add deps for
  /// @param Deps List of dependent nodes
  void addDepsToNode(std::shared_ptr<node_impl> Node,
                     const std::vector<std::shared_ptr<node_impl>> &Deps) {
    if (!Deps.empty()) {
      for (auto &N : Deps) {
        N->registerSuccessor(Node, N);
        this->removeRoot(Node);
      }
    } else {
      this->addRoot(Node);
    }
  }

  /// Context associated with this graph.
  sycl::context MContext;
  /// Device associated with this graph. All graph nodes will execute on this
  /// device.
  sycl::device MDevice;
  /// Unique set of queues which are currently recording to this graph.
  std::set<std::weak_ptr<sycl::detail::queue_impl>,
           std::owner_less<std::weak_ptr<sycl::detail::queue_impl>>>
      MRecordingQueues;
  /// Map of events to their associated recorded nodes.
  std::unordered_map<std::shared_ptr<sycl::detail::event_impl>,
                     std::shared_ptr<node_impl>>
      MEventsMap;
  /// Map for every in-order queue thats recorded a node to the graph, what
  /// the last node added was. We can use this to create new edges on the last
  /// node if any more nodes are added to the graph from the queue.
  std::map<std::weak_ptr<sycl::detail::queue_impl>, std::shared_ptr<node_impl>,
           std::owner_less<std::weak_ptr<sycl::detail::queue_impl>>>
      MInorderQueueMap;
  /// Controls whether we skip the cycle checks in makeEdge, set by the presence
  /// of the no_cycle_check property on construction.
  bool MSkipCycleChecks = false;
  /// Unique set of SYCL Memory Objects which are currently in use in the graph.
  std::set<sycl::detail::SYCLMemObjT *> MMemObjs;

  /// Controls whether we allow buffers to be used in the graph. Set by the
  /// presence of the assume_buffer_outlives_graph property.
  bool MAllowBuffers = false;

  /// List of nodes that must be added as extra dependencies to new nodes when
  /// added to this graph.
  /// This list is mainly used by barrier nodes which must be considered
  /// as predecessors for all nodes subsequently added to the graph.
  std::list<std::shared_ptr<node_impl>> MExtraDependencies;
};

/// Class representing the implementation of command_graph<executable>.
class exec_graph_impl {
public:
  using ReadLock = std::shared_lock<std::shared_mutex>;
  using WriteLock = std::unique_lock<std::shared_mutex>;

  /// Protects all the fields that can be changed by class' methods.
  mutable std::shared_mutex MMutex;

  /// Constructor.
  /// @param Context Context to create graph with.
  /// @param GraphImpl Modifiable graph implementation to create with.
  exec_graph_impl(sycl::context Context,
                  const std::shared_ptr<graph_impl> &GraphImpl)
      : MSchedule(), MGraphImpl(GraphImpl), MPiSyncPoints(), MContext(Context),
        MRequirements(), MExecutionEvents() {}

  /// Destructor.
  ///
  /// Releases any PI command-buffers the object has created.
  ~exec_graph_impl();

  /// Partition the graph nodes and put the partition in MPartitions.
  /// The partitioning splits the graph to allow synchronization between
  /// device events and events that do not run on the same device such as
  /// host_task.
  void makePartitions();

  /// Called by handler::ext_oneapi_command_graph() to schedule graph for
  /// execution.
  /// @param Queue Command-queue to schedule execution on.
  /// @param CGData Command-group data provided by the sycl::handler
  /// @return Event associated with the execution of the graph.
  sycl::event enqueue(const std::shared_ptr<sycl::detail::queue_impl> &Queue,
                      sycl::detail::CG::StorageInitHelper CGData);

  /// Turns the internal graph representation into UR command-buffers for a
  /// device.
  /// @param Device Device to create backend command-buffers for.
  /// @param Partion Partition to which the created command-buffer should be
  /// attached.
  void createCommandBuffers(sycl::device Device,
                            std::shared_ptr<partition> &Partition);

  /// Query for the context tied to this graph.
  /// @return Context associated with graph.
  sycl::context getContext() const { return MContext; }

  /// Query the scheduling of node execution.
  /// @return List of nodes in execution order.
  const std::list<std::shared_ptr<node_impl>> &getSchedule() const {
    return MSchedule;
  }

  /// Query the graph_impl.
  /// @return pointer to the graph_impl MGraphImpl
  const std::shared_ptr<graph_impl> &getGraphImpl() const { return MGraphImpl; }

  /// Query the vector of the partitions composing the exec_graph.
  /// @return Vector of partitions in execution order.
  const std::vector<std::shared_ptr<partition>> &getPartitions() const {
    return MPartitions;
  }

  /// Checks if the previous submissions of this graph have been completed
  /// This function checks the status of events associated to the previous graph
  /// submissions.
  /// @return true if all previous submissions have been completed, false
  /// otherwise.
  bool previousSubmissionCompleted() const {
    for (auto Event : MExecutionEvents) {
      if (!Event->isCompleted()) {
        return false;
      }
    }
    return true;
  }

  /// Returns the SyncPoint associated to the node passed in parameter for this exec graph.
  /// @param Node shared pointer to the node to look for.
  /// @return the associated SyncPoint if it exists.
  sycl::detail::pi::PiExtSyncPoint getSyncPointFromNode(std::shared_ptr<node_impl> Node) const {
    auto SyncPoint = MPiSyncPoints.find(Node);
    if (SyncPoint != MPiSyncPoints.end()) {
      return SyncPoint->second;
    }
    return -1;
  }

private:
  /// Create a command-group for the node and add it to command-buffer by going
  /// through the scheduler.
  /// @param Ctx Context to use.
  /// @param DeviceImpl Device associated with the enqueue.
  /// @param CommandBuffer Command-buffer to add node to as a command.
  /// @param Node The node being enqueued.
  /// @return PI sync point created for this node in the command-buffer.
  sycl::detail::pi::PiExtSyncPoint
  enqueueNode(sycl::context Ctx, sycl::detail::DeviceImplPtr DeviceImpl,
              sycl::detail::pi::PiExtCommandBuffer CommandBuffer,
              std::shared_ptr<node_impl> Node);

  /// Enqueue a node directly to the command-buffer without going through the
  /// scheduler.
  /// @param Ctx Context to use.
  /// @param DeviceImpl Device associated with the enqueue.
  /// @param CommandBuffer Command-buffer to add node to as a command.
  /// @param Node The node being enqueued.
  /// @return PI sync point created for this node in the command-buffer.
  sycl::detail::pi::PiExtSyncPoint
  enqueueNodeDirect(sycl::context Ctx, sycl::detail::DeviceImplPtr DeviceImpl,
                    sycl::detail::pi::PiExtCommandBuffer CommandBuffer,
                    std::shared_ptr<node_impl> Node);

  /// Iterates back through predecessors to find the real dependency.
  /// @param[out] Deps Found dependencies.
  /// @param[in] CurrentNode Node to find dependencies for.
  /// @param[in] ReferencePartitionNum Number of the partition containing the
  /// SyncPoint for CurrentNode, otherwise we need to
  /// synchronize on the host with the completion of previous partitions.
  void findRealDeps(std::vector<sycl::detail::pi::PiExtSyncPoint> &Deps,
                    std::shared_ptr<node_impl> CurrentNode,
                    int ReferencePartitionNum);

  /// Execution schedule of nodes in the graph.
  std::list<std::shared_ptr<node_impl>> MSchedule;
  /// Pointer to the modifiable graph impl associated with this executable
  /// graph.
  /// Thread-safe implementation note: in the current implementation
  /// multiple exec_graph_impl can reference the same graph_impl object.
  /// This specificity must be taken into account when trying to lock
  /// the graph_impl mutex from an exec_graph_impl to avoid deadlock.
  std::shared_ptr<graph_impl> MGraphImpl;
  /// Map of nodes in the exec graph to the sync point representing their
  /// execution in the command graph.
  std::unordered_map<std::shared_ptr<node_impl>,
                     sycl::detail::pi::PiExtSyncPoint>
      MPiSyncPoints;
  /// Map of nodes in the exec graph to the partition number to which they
  /// belong.
  std::unordered_map<std::shared_ptr<node_impl>, int> MPartitionNodes;
  /// Context associated with this executable graph.
  sycl::context MContext;
  /// List of requirements for enqueueing this command graph, accumulated from
  /// all nodes enqueued to the graph.
  std::vector<sycl::detail::AccessorImplHost *> MRequirements;
  /// Storage for accessors which are used by this graph, accumulated from
  /// all nodes enqueued to the graph.
  std::vector<sycl::detail::AccessorImplPtr> MAccessors;
  /// List of all execution events returned from command buffer enqueue calls.
  std::vector<sycl::detail::EventImplPtr> MExecutionEvents;
  /// List of the partitions that compose the exec graph.
  std::vector<std::shared_ptr<partition>> MPartitions;
};

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
