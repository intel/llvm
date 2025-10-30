//==--------- node_impl.hpp --- SYCL graph extension -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/accessor_impl.hpp> // for AccessorImplHost
#include <detail/cg.hpp>            // for CGExecKernel, CGHostTask, ArgDesc...
#include <detail/helpers.hpp>
#include <detail/host_task.hpp>        // for HostTask
#include <sycl/detail/cg_types.hpp>    // for CGType
#include <sycl/detail/kernel_desc.hpp> // for kernel_param_kind_t

#include <sycl/ext/oneapi/experimental/enqueue_types.hpp> // for prefetchType
#include <sycl/ext/oneapi/experimental/graph/node.hpp>    // for node

#include <cstring>
#include <fstream>
#include <iomanip>
#include <list>
#include <set>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {

namespace detail {
// Forward declarations
class node_impl;
class nodes_range;
class exec_graph_impl;

inline node_type getNodeTypeFromCG(sycl::detail::CGType CGType) {
  using sycl::detail::CG;

  switch (CGType) {
  case sycl::detail::CGType::None:
    return node_type::empty;
  case sycl::detail::CGType::Kernel:
    return node_type::kernel;
  case sycl::detail::CGType::CopyAccToPtr:
  case sycl::detail::CGType::CopyPtrToAcc:
  case sycl::detail::CGType::CopyAccToAcc:
  case sycl::detail::CGType::CopyUSM:
    return node_type::memcpy;
  case sycl::detail::CGType::Memset2DUSM:
    return node_type::memset;
  case sycl::detail::CGType::Fill:
  case sycl::detail::CGType::FillUSM:
    return node_type::memfill;
  case sycl::detail::CGType::PrefetchUSM:
    return node_type::prefetch;
  case sycl::detail::CGType::AdviseUSM:
    return node_type::memadvise;
  case sycl::detail::CGType::Barrier:
  case sycl::detail::CGType::BarrierWaitlist:
    return node_type::ext_oneapi_barrier;
  case sycl::detail::CGType::CodeplayHostTask:
    return node_type::host_task;
  case sycl::detail::CGType::ExecCommandBuffer:
    return node_type::subgraph;
  case sycl::detail::CGType::EnqueueNativeCommand:
    return node_type::native_command;
  case sycl::detail::CGType::AsyncAlloc:
    return node_type::async_malloc;
  case sycl::detail::CGType::AsyncFree:
    return node_type::async_free;

  default:
    assert(false && "Invalid Graph Node Type");
    return node_type::empty;
  }
}

/// Implementation of node class from SYCL_EXT_ONEAPI_GRAPH.
class node_impl : public std::enable_shared_from_this<node_impl> {
public:
  using id_type = uint64_t;

  /// Unique identifier for this node.
  id_type MID = getNextNodeID();
  /// List of successors to this node.
  std::vector<node_impl *> MSuccessors;
  /// List of predecessors to this node.
  ///
  /// Using weak_ptr here to prevent circular references between nodes.
  std::vector<node_impl *> MPredecessors;
  /// Type of the command-group for the node.
  sycl::detail::CGType MCGType = sycl::detail::CGType::None;
  /// User facing type of the node.
  node_type MNodeType = node_type::empty;
  /// Command group object which stores all args etc needed to enqueue the node
  std::shared_ptr<sycl::detail::CG> MCommandGroup;
  /// Stores the executable graph impl associated with this node if it is a
  /// subgraph node.
  std::shared_ptr<exec_graph_impl> MSubGraphImpl;

  /// Used for tracking visited status during cycle checks and node scheduling.
  size_t MTotalVisitedEdges = 0;

  /// Partition number needed to assign a Node to a a partition.
  /// Note : This number is only used during the partitionning process and
  /// cannot be used to find out the partion of a node outside of this process.
  int MPartitionNum = -1;

  // Out-of-class as need "complete" `nodes_range`:
  inline nodes_range successors() const;
  inline nodes_range predecessors() const;

  /// Add successor to the node.
  /// @param Node Node to add as a successor.
  void registerSuccessor(node_impl &Node) {
    if (std::find(MSuccessors.begin(), MSuccessors.end(), &Node) !=
        MSuccessors.end()) {
      return;
    }
    MSuccessors.push_back(&Node);
    Node.registerPredecessor(*this);
  }

  /// Add predecessor to the node.
  /// @param Node Node to add as a predecessor.
  void registerPredecessor(node_impl &Node) {
    if (std::find(MPredecessors.begin(), MPredecessors.end(), &Node) !=
        MPredecessors.end()) {
      return;
    }
    MPredecessors.push_back(&Node);
  }

  /// Construct an empty node.
  node_impl() {}

  /// Construct a node representing a command-group.
  /// @param NodeType Type of the command-group.
  /// @param CommandGroup The CG which stores the command information for this
  /// node.
  node_impl(node_type NodeType,
            const std::shared_ptr<sycl::detail::CG> &CommandGroup)
      : MCGType(CommandGroup->getType()), MNodeType(NodeType),
        MCommandGroup(CommandGroup) {
    if (NodeType == node_type::subgraph) {
      MSubGraphImpl =
          static_cast<sycl::detail::CGExecCommandBuffer *>(MCommandGroup.get())
              ->MExecGraph;
    }
  }

  /// Construct a node from another node. This will perform a deep-copy of the
  /// command group object associated with this node.
  node_impl(node_impl &Other)
      : enable_shared_from_this(Other), MSuccessors(Other.MSuccessors),
        MPredecessors(Other.MPredecessors), MCGType(Other.MCGType),
        MNodeType(Other.MNodeType), MCommandGroup(Other.getCGCopy()),
        MSubGraphImpl(Other.MSubGraphImpl) {}

  /// Copy-assignment operator. This will perform a deep-copy of the
  /// command group object associated with this node.
  node_impl &operator=(node_impl &Other) {
    if (this != &Other) {
      MSuccessors = Other.MSuccessors;
      MPredecessors = Other.MPredecessors;
      MCGType = Other.MCGType;
      MNodeType = Other.MNodeType;
      MCommandGroup = Other.getCGCopy();
      MSubGraphImpl = Other.MSubGraphImpl;
    }
    return *this;
  }

  ~node_impl() {}

  /// Checks if this node should be a dependency of another node based on
  /// accessor requirements. This is calculated using access modes if a
  /// requirement to the same buffer is found inside this node.
  /// @param IncomingReq Incoming requirement.
  /// @return True if a dependency is needed, false if not.
  bool hasRequirementDependency(sycl::detail::AccessorImplHost *IncomingReq) {
    if (!MCommandGroup)
      return false;

    access_mode InMode = IncomingReq->MAccessMode;
    switch (InMode) {
    case access_mode::read:
    case access_mode::read_write:
    case access_mode::atomic:
      break;
    // These access modes don't care about existing buffer data, so we don't
    // need a dependency.
    case access_mode::write:
    case access_mode::discard_read_write:
    case access_mode::discard_write:
      return false;
    }

    for (sycl::detail::AccessorImplHost *CurrentReq :
         MCommandGroup->getRequirements()) {
      if (IncomingReq->MSYCLMemObj == CurrentReq->MSYCLMemObj) {
        access_mode CurrentMode = CurrentReq->MAccessMode;
        // Since we have an incoming read requirement, we only care
        // about requirements on this node if they are write
        if (CurrentMode != access_mode::read) {
          return true;
        }
      }
    }
    // No dependency necessary
    return false;
  }

  /// Query if this is an empty node.
  /// Barrier nodes are also considered empty nodes since they do not embed any
  /// workload but only dependencies
  /// @return True if this is an empty node, false otherwise.
  bool isEmpty() const {
    return ((MCGType == sycl::detail::CGType::None) ||
            (MCGType == sycl::detail::CGType::Barrier));
  }

  /// Get a deep copy of this node's command group
  /// @return A unique ptr to the new command group object.
  std::unique_ptr<sycl::detail::CG> getCGCopy() const {
    switch (MCGType) {
    case sycl::detail::CGType::Kernel: {
      auto CGCopy = createCGCopy<sycl::detail::CGExecKernel>();
      rebuildArgStorage(CGCopy->MArgs, MCommandGroup->getArgsStorage(),
                        CGCopy->getArgsStorage());
      return std::move(CGCopy);
    }
    case sycl::detail::CGType::CopyAccToPtr:
    case sycl::detail::CGType::CopyPtrToAcc:
    case sycl::detail::CGType::CopyAccToAcc:
      return createCGCopy<sycl::detail::CGCopy>();
    case sycl::detail::CGType::Fill:
      return createCGCopy<sycl::detail::CGFill>();
    case sycl::detail::CGType::UpdateHost:
      return createCGCopy<sycl::detail::CGUpdateHost>();
    case sycl::detail::CGType::CopyUSM:
      return createCGCopy<sycl::detail::CGCopyUSM>();
    case sycl::detail::CGType::FillUSM:
      return createCGCopy<sycl::detail::CGFillUSM>();
    case sycl::detail::CGType::PrefetchUSM:
      return createCGCopy<sycl::detail::CGPrefetchUSM>();
    case sycl::detail::CGType::AdviseUSM:
      return createCGCopy<sycl::detail::CGAdviseUSM>();
    case sycl::detail::CGType::Copy2DUSM:
      return createCGCopy<sycl::detail::CGCopy2DUSM>();
    case sycl::detail::CGType::Fill2DUSM:
      return createCGCopy<sycl::detail::CGFill2DUSM>();
    case sycl::detail::CGType::Memset2DUSM:
      return createCGCopy<sycl::detail::CGMemset2DUSM>();
    case sycl::detail::CGType::EnqueueNativeCommand:
    case sycl::detail::CGType::CodeplayHostTask: {
      // The unique_ptr to the `sycl::detail::HostTask`, which is also used for
      // a EnqueueNativeCommand command, in the HostTask CG prevents from
      // copying the CG. We overcome this restriction by creating a new CG with
      // the same data.
      auto CommandGroupPtr =
          static_cast<sycl::detail::CGHostTask *>(MCommandGroup.get());
      sycl::detail::HostTask HostTask = *CommandGroupPtr->MHostTask.get();
      auto HostTaskSPtr = std::make_shared<sycl::detail::HostTask>(HostTask);

      sycl::detail::CG::StorageInitHelper Data(
          CommandGroupPtr->getArgsStorage(), CommandGroupPtr->getAccStorage(),
          CommandGroupPtr->getSharedPtrStorage(),
          CommandGroupPtr->getRequirements(), CommandGroupPtr->getEvents());

      std::vector<sycl::detail::ArgDesc> NewArgs = CommandGroupPtr->MArgs;

      rebuildArgStorage(NewArgs, CommandGroupPtr->getArgsStorage(),
                        Data.MArgsStorage);

      sycl::detail::code_location Loc(CommandGroupPtr->MFileName.data(),
                                      CommandGroupPtr->MFunctionName.data(),
                                      CommandGroupPtr->MLine,
                                      CommandGroupPtr->MColumn);

      return std::make_unique<sycl::detail::CGHostTask>(
          sycl::detail::CGHostTask(
              std::move(HostTaskSPtr), CommandGroupPtr->MQueue.get(),
              CommandGroupPtr->MContext.get(), std::move(NewArgs),
              std::move(Data), CommandGroupPtr->getType(), Loc));
    }
    case sycl::detail::CGType::Barrier:
    case sycl::detail::CGType::BarrierWaitlist:
      // Barrier nodes are stored in the graph with only the base CG class,
      // since they are treated internally as empty nodes.
      return createCGCopy<sycl::detail::CG>();
    case sycl::detail::CGType::CopyToDeviceGlobal:
      return createCGCopy<sycl::detail::CGCopyToDeviceGlobal>();
    case sycl::detail::CGType::CopyFromDeviceGlobal:
      return createCGCopy<sycl::detail::CGCopyFromDeviceGlobal>();
    case sycl::detail::CGType::ReadWriteHostPipe:
      return createCGCopy<sycl::detail::CGReadWriteHostPipe>();
    case sycl::detail::CGType::CopyImage:
      return createCGCopy<sycl::detail::CGCopyImage>();
    case sycl::detail::CGType::SemaphoreSignal:
      return createCGCopy<sycl::detail::CGSemaphoreSignal>();
    case sycl::detail::CGType::SemaphoreWait:
      return createCGCopy<sycl::detail::CGSemaphoreWait>();
    case sycl::detail::CGType::ProfilingTag:
      return createCGCopy<sycl::detail::CGProfilingTag>();
    case sycl::detail::CGType::ExecCommandBuffer:
      return createCGCopy<sycl::detail::CGExecCommandBuffer>();
    case sycl::detail::CGType::AsyncAlloc:
      return createCGCopy<sycl::detail::CGAsyncAlloc>();
    case sycl::detail::CGType::AsyncFree:
      return createCGCopy<sycl::detail::CGAsyncFree>();
    case sycl::detail::CGType::None:
      return nullptr;
    }
    return nullptr;
  }

  /// Tests if the caller is similar to Node, this is only used for testing.
  /// @param Node The node to check for similarity.
  /// @param CompareContentOnly Skip comparisons related to graph structure,
  /// compare only the type and command groups of the nodes
  /// @return True if the two nodes are similar
  bool isSimilar(node_impl &Node, bool CompareContentOnly = false) const {
    if (!CompareContentOnly) {
      if (MSuccessors.size() != Node.MSuccessors.size())
        return false;

      if (MPredecessors.size() != Node.MPredecessors.size())
        return false;
    }
    if (MCGType != Node.MCGType)
      return false;

    switch (MCGType) {
    case sycl::detail::CGType::Kernel: {
      sycl::detail::CGExecKernel *ExecKernelA =
          static_cast<sycl::detail::CGExecKernel *>(MCommandGroup.get());
      sycl::detail::CGExecKernel *ExecKernelB =
          static_cast<sycl::detail::CGExecKernel *>(Node.MCommandGroup.get());
      return ExecKernelA->getKernelName() == ExecKernelB->getKernelName();
    }
    case sycl::detail::CGType::CopyUSM: {
      sycl::detail::CGCopyUSM *CopyA =
          static_cast<sycl::detail::CGCopyUSM *>(MCommandGroup.get());
      sycl::detail::CGCopyUSM *CopyB =
          static_cast<sycl::detail::CGCopyUSM *>(Node.MCommandGroup.get());
      return (CopyA->getSrc() == CopyB->getSrc()) &&
             (CopyA->getDst() == CopyB->getDst()) &&
             (CopyA->getLength() == CopyB->getLength());
    }
    case sycl::detail::CGType::CopyAccToAcc:
    case sycl::detail::CGType::CopyAccToPtr:
    case sycl::detail::CGType::CopyPtrToAcc: {
      sycl::detail::CGCopy *CopyA =
          static_cast<sycl::detail::CGCopy *>(MCommandGroup.get());
      sycl::detail::CGCopy *CopyB =
          static_cast<sycl::detail::CGCopy *>(Node.MCommandGroup.get());
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
    for (node_impl *Pred : MPredecessors) {
      Stream << "  \"" << Pred << "\" -> \"" << this << "\"" << std::endl;
    }

    for (node_impl *Succ : MSuccessors) {
      if (MPartitionNum == Succ->MPartitionNum)
        Succ->printDotRecursive(Stream, Visited, Verbose);
    }
  }

  /// Test if the node contains a N-D copy
  /// @return true if the op is a N-D copy
  bool isNDCopyNode() const {
    if ((MCGType != sycl::detail::CGType::CopyAccToAcc) &&
        (MCGType != sycl::detail::CGType::CopyAccToPtr) &&
        (MCGType != sycl::detail::CGType::CopyPtrToAcc)) {
      return false;
    }

    auto Copy = static_cast<sycl::detail::CGCopy *>(MCommandGroup.get());
    auto ReqSrc = static_cast<sycl::detail::Requirement *>(Copy->getSrc());
    auto ReqDst = static_cast<sycl::detail::Requirement *>(Copy->getDst());
    return (ReqSrc->MDims > 1) || (ReqDst->MDims > 1);
  }

  template <int Dimensions>
  void updateNDRange(nd_range<Dimensions> ExecutionRange) {
    if (MCGType != sycl::detail::CGType::Kernel) {
      throw sycl::exception(
          sycl::errc::invalid,
          "Cannot update execution range of nodes which are not kernel nodes");
    }

    auto &NDRDesc =
        static_cast<sycl::detail::CGExecKernel *>(MCommandGroup.get())
            ->MNDRDesc;

    if (NDRDesc.Dims != Dimensions) {
      throw sycl::exception(sycl::errc::invalid,
                            "Cannot update execution range of a node with an "
                            "execution range of different dimensions than what "
                            "the node was originally created with.");
    }

    NDRDesc = sycl::detail::NDRDescT{ExecutionRange};
  }

  template <int Dimensions> void updateRange(range<Dimensions> ExecutionRange) {
    if (MCGType != sycl::detail::CGType::Kernel) {
      throw sycl::exception(
          sycl::errc::invalid,
          "Cannot update execution range of nodes which are not kernel nodes");
    }

    auto &NDRDesc =
        static_cast<sycl::detail::CGExecKernel *>(MCommandGroup.get())
            ->MNDRDesc;

    if (NDRDesc.Dims != Dimensions) {
      throw sycl::exception(sycl::errc::invalid,
                            "Cannot update execution range of a node with an "
                            "execution range of different dimensions than what "
                            "the node was originally created with.");
    }

    NDRDesc = sycl::detail::NDRDescT{ExecutionRange};
  }
  /// Update this node with the command-group from another node.
  /// @param Other The other node to update, must be of the same node type.
  void updateFromOtherNode(node_impl &Other) {
    assert(MNodeType == Other.MNodeType);
    MCommandGroup = Other.getCGCopy();
  }

  id_type getID() const { return MID; }

  /// Returns true if this node can be updated
  bool isUpdatable() const {
    switch (MNodeType) {
    case node_type::kernel:
    case node_type::host_task:
    case node_type::ext_oneapi_barrier:
    case node_type::empty:
      return true;

    default:
      return false;
    }
  }

  /// Returns true if this node should be enqueued to the backend, if not only
  /// its dependencies are considered.
  bool requiresEnqueue() const {
    switch (MNodeType) {
    case node_type::empty:
    case node_type::ext_oneapi_barrier:
    case node_type::async_malloc:
    case node_type::async_free:
      return false;

    default:
      return true;
    }
  }

private:
  void rebuildArgStorage(std::vector<sycl::detail::ArgDesc> &Args,
                         const std::vector<std::vector<char>> &OldArgStorage,
                         std::vector<std::vector<char>> &NewArgStorage) const {
    // Clear the arg storage so we can rebuild it
    NewArgStorage.clear();

    // Loop over all the args, any std_layout ones need their pointers updated
    // to point to the new arg storage.
    for (auto &Arg : Args) {
      if (Arg.MType != sycl::detail::kernel_param_kind_t::kind_std_layout) {
        continue;
      }
      // Find which ArgStorage Arg.MPtr is pointing to
      for (auto &ArgStorage : OldArgStorage) {
        if (ArgStorage.data() != Arg.MPtr) {
          continue;
        }
        NewArgStorage.emplace_back(Arg.MSize);
        // Memcpy contents from old storage to new storage
        std::memcpy(NewArgStorage.back().data(), ArgStorage.data(), Arg.MSize);
        // Update MPtr to point to the new storage instead of the old
        Arg.MPtr = NewArgStorage.back().data();

        break;
      }
    }
  }
  // Gets the next unique identifier for a node, should only be used when
  // constructing nodes.
  static id_type getNextNodeID() {
    static id_type nextID = 0;

    // Return the value then increment the next ID
    return nextID++;
  }

  /// Prints Node information to Stream.
  /// @param Stream Where to print the Node information
  /// @param Verbose If true, print additional information about the nodes
  /// such as kernel args or memory access where applicable.
  void printDotCG(std::ostream &Stream, bool Verbose) {
    Stream << "\"" << this << "\" [style=bold, label=\"";

    Stream << "ID = " << this << "\\n";
    Stream << "TYPE = ";

    switch (MCGType) {
    case sycl::detail::CGType::None:
      Stream << "None \\n";
      break;
    case sycl::detail::CGType::Kernel: {
      Stream << "CGExecKernel \\n";
      sycl::detail::CGExecKernel *Kernel =
          static_cast<sycl::detail::CGExecKernel *>(MCommandGroup.get());
      Stream << "NAME = " << Kernel->getKernelName() << "\\n";
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
            auto Fill = Stream.fill();
            Stream << i << ") Type: " << Type << " Ptr: " << Arg.MPtr << "(0x"
                   << std::hex << std::setfill('0');
            for (int i = Arg.MSize - 1; i >= 0; --i) {
              Stream << std::setw(2)
                     << static_cast<int16_t>(
                            (static_cast<unsigned char *>(Arg.MPtr))[i]);
            }
            Stream.fill(Fill);
            Stream << std::dec << ")\\n";
            continue;
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
    case sycl::detail::CGType::CopyAccToPtr:
      Stream << "CGCopy Device-to-Host \\n";
      if (Verbose) {
        sycl::detail::CGCopy *Copy =
            static_cast<sycl::detail::CGCopy *>(MCommandGroup.get());
        Stream << "Src: " << Copy->getSrc() << " Dst: " << Copy->getDst()
               << "\\n";
      }
      break;
    case sycl::detail::CGType::CopyPtrToAcc:
      Stream << "CGCopy Host-to-Device \\n";
      if (Verbose) {
        sycl::detail::CGCopy *Copy =
            static_cast<sycl::detail::CGCopy *>(MCommandGroup.get());
        Stream << "Src: " << Copy->getSrc() << " Dst: " << Copy->getDst()
               << "\\n";
      }
      break;
    case sycl::detail::CGType::CopyAccToAcc:
      Stream << "CGCopy Device-to-Device \\n";
      if (Verbose) {
        sycl::detail::CGCopy *Copy =
            static_cast<sycl::detail::CGCopy *>(MCommandGroup.get());
        Stream << "Src: " << Copy->getSrc() << " Dst: " << Copy->getDst()
               << "\\n";
      }
      break;
    case sycl::detail::CGType::Fill:
      Stream << "CGFill \\n";
      if (Verbose) {
        sycl::detail::CGFill *Fill =
            static_cast<sycl::detail::CGFill *>(MCommandGroup.get());
        Stream << "Ptr: " << Fill->MPtr << "\\n";
      }
      break;
    case sycl::detail::CGType::UpdateHost:
      Stream << "CGCUpdateHost \\n";
      if (Verbose) {
        sycl::detail::CGUpdateHost *Host =
            static_cast<sycl::detail::CGUpdateHost *>(MCommandGroup.get());
        Stream << "Ptr: " << Host->getReqToUpdate() << "\\n";
      }
      break;
    case sycl::detail::CGType::CopyUSM:
      Stream << "CGCopyUSM \\n";
      if (Verbose) {
        sycl::detail::CGCopyUSM *CopyUSM =
            static_cast<sycl::detail::CGCopyUSM *>(MCommandGroup.get());
        Stream << "Src: " << CopyUSM->getSrc() << " Dst: " << CopyUSM->getDst()
               << " Length: " << CopyUSM->getLength() << "\\n";
      }
      break;
    case sycl::detail::CGType::FillUSM:
      Stream << "CGFillUSM \\n";
      if (Verbose) {
        sycl::detail::CGFillUSM *FillUSM =
            static_cast<sycl::detail::CGFillUSM *>(MCommandGroup.get());
        Stream << "Dst: " << FillUSM->getDst()
               << " Length: " << FillUSM->getLength() << " Pattern: ";
        for (auto byte : FillUSM->getPattern())
          Stream << byte;
        Stream << "\\n";
      }
      break;
    case sycl::detail::CGType::PrefetchUSM:
      Stream << "CGPrefetchUSM \\n";
      if (Verbose) {
        sycl::detail::CGPrefetchUSM *Prefetch =
            static_cast<sycl::detail::CGPrefetchUSM *>(MCommandGroup.get());
        Stream << "Dst: " << Prefetch->getDst()
               << " Length: " << Prefetch->getLength() << " PrefetchType: "
               << sycl::ext::oneapi::experimental::prefetchTypeToString(
                      Prefetch->getPrefetchType())
               << "\\n";
      }
      break;
    case sycl::detail::CGType::AdviseUSM:
      Stream << "CGAdviseUSM \\n";
      if (Verbose) {
        sycl::detail::CGAdviseUSM *AdviseUSM =
            static_cast<sycl::detail::CGAdviseUSM *>(MCommandGroup.get());
        Stream << "Dst: " << AdviseUSM->getDst()
               << " Length: " << AdviseUSM->getLength() << "\\n";
      }
      break;
    case sycl::detail::CGType::CodeplayHostTask:
      Stream << "CGHostTask \\n";
      break;
    case sycl::detail::CGType::Barrier:
      Stream << "CGBarrier \\n";
      break;
    case sycl::detail::CGType::Copy2DUSM:
      Stream << "CGCopy2DUSM \\n";
      if (Verbose) {
        sycl::detail::CGCopy2DUSM *Copy2DUSM =
            static_cast<sycl::detail::CGCopy2DUSM *>(MCommandGroup.get());
        Stream << "Src:" << Copy2DUSM->getSrc()
               << " Dst: " << Copy2DUSM->getDst() << "\\n";
      }
      break;
    case sycl::detail::CGType::Fill2DUSM:
      Stream << "CGFill2DUSM \\n";
      if (Verbose) {
        sycl::detail::CGFill2DUSM *Fill2DUSM =
            static_cast<sycl::detail::CGFill2DUSM *>(MCommandGroup.get());
        Stream << "Dst: " << Fill2DUSM->getDst() << "\\n";
      }
      break;
    case sycl::detail::CGType::Memset2DUSM:
      Stream << "CGMemset2DUSM \\n";
      if (Verbose) {
        sycl::detail::CGMemset2DUSM *Memset2DUSM =
            static_cast<sycl::detail::CGMemset2DUSM *>(MCommandGroup.get());
        Stream << "Dst: " << Memset2DUSM->getDst() << "\\n";
      }
      break;
    case sycl::detail::CGType::ReadWriteHostPipe:
      Stream << "CGReadWriteHostPipe \\n";
      break;
    case sycl::detail::CGType::CopyToDeviceGlobal:
      Stream << "CGCopyToDeviceGlobal \\n";
      if (Verbose) {
        sycl::detail::CGCopyToDeviceGlobal *CopyToDeviceGlobal =
            static_cast<sycl::detail::CGCopyToDeviceGlobal *>(
                MCommandGroup.get());
        Stream << "Src: " << CopyToDeviceGlobal->getSrc()
               << " Dst: " << CopyToDeviceGlobal->getDeviceGlobalPtr() << "\\n";
      }
      break;
    case sycl::detail::CGType::CopyFromDeviceGlobal:
      Stream << "CGCopyFromDeviceGlobal \\n";
      if (Verbose) {
        sycl::detail::CGCopyFromDeviceGlobal *CopyFromDeviceGlobal =
            static_cast<sycl::detail::CGCopyFromDeviceGlobal *>(
                MCommandGroup.get());
        Stream << "Src: " << CopyFromDeviceGlobal->getDeviceGlobalPtr()
               << " Dst: " << CopyFromDeviceGlobal->getDest() << "\\n";
      }
      break;
    case sycl::detail::CGType::ExecCommandBuffer:
      Stream << "CGExecCommandBuffer \\n";
      break;
    case sycl::detail::CGType::EnqueueNativeCommand:
      Stream << "CGNativeCommand \\n";
      break;
    case sycl::detail::CGType::AsyncAlloc:
      Stream << "CGAsyncAlloc \\n";
      break;
    case sycl::detail::CGType::AsyncFree:
      Stream << "CGAsyncFree \\n";
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

template <typename... ContainerTy>
using nodes_iterator_impl =
    variadic_iterator<node, typename ContainerTy::const_iterator...>;

using nodes_iterator = nodes_iterator_impl<
    std::vector<std::shared_ptr<node_impl>>, std::vector<node_impl *>,
    std::set<std::shared_ptr<node_impl>>, std::set<node_impl *>,
    std::list<node_impl *>, std::vector<node>>;

class nodes_range : public iterator_range<nodes_iterator> {
private:
  using Base = iterator_range<nodes_iterator>;

public:
  using Base::Base;
};

inline nodes_range node_impl::successors() const { return MSuccessors; }
inline nodes_range node_impl::predecessors() const { return MPredecessors; }

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl

namespace std {
template <> struct __SYCL_EXPORT hash<sycl::ext::oneapi::experimental::node> {
  size_t operator()(const sycl::ext::oneapi::experimental::node &Node) const;
};
} // namespace std
