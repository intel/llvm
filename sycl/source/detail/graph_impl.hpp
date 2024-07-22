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
#include <detail/cg.hpp>
#include <detail/event_impl.hpp>
#include <detail/host_task.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/sycl_mem_obj_t.hpp>

#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
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
  default:
    assert(false && "Invalid Graph Node Type");
    return node_type::empty;
  }
}

/// Implementation of node class from SYCL_EXT_ONEAPI_GRAPH.
class node_impl {
public:
  using id_type = uint64_t;

  /// Unique identifier for this node.
  id_type MID = getNextNodeID();
  /// List of successors to this node.
  std::vector<std::weak_ptr<node_impl>> MSuccessors;
  /// List of predecessors to this node.
  ///
  /// Using weak_ptr here to prevent circular references between nodes.
  std::vector<std::weak_ptr<node_impl>> MPredecessors;
  /// Type of the command-group for the node.
  sycl::detail::CGType MCGType = sycl::detail::CGType::None;
  /// User facing type of the node.
  node_type MNodeType = node_type::empty;
  /// Command group object which stores all args etc needed to enqueue the node
  std::unique_ptr<sycl::detail::CG> MCommandGroup;
  /// Stores the executable graph impl associated with this node if it is a
  /// subgraph node.
  std::shared_ptr<exec_graph_impl> MSubGraphImpl;

  /// Used for tracking visited status during cycle checks.
  bool MVisited = false;

  /// Partition number needed to assign a Node to a a partition.
  /// Note : This number is only used during the partitionning process and
  /// cannot be used to find out the partion of a node outside of this process.
  int MPartitionNum = -1;

  /// Track whether an ND-Range was used for kernel nodes
  bool MNDRangeUsed = false;

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
  /// @param NodeType Type of the command-group.
  /// @param CommandGroup The CG which stores the command information for this
  /// node.
  node_impl(node_type NodeType,
            std::unique_ptr<sycl::detail::CG> &&CommandGroup)
      : MCGType(CommandGroup->getType()), MNodeType(NodeType),
        MCommandGroup(std::move(CommandGroup)) {
    if (NodeType == node_type::subgraph) {
      MSubGraphImpl =
          static_cast<sycl::detail::CGExecCommandBuffer *>(MCommandGroup.get())
              ->MExecGraph;
    }
  }

  /// Construct a node from another node. This will perform a deep-copy of the
  /// command group object associated with this node.
  node_impl(node_impl &Other)
      : MSuccessors(Other.MSuccessors), MPredecessors(Other.MPredecessors),
        MCGType(Other.MCGType), MNodeType(Other.MNodeType),
        MCommandGroup(Other.getCGCopy()), MSubGraphImpl(Other.MSubGraphImpl) {}

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
    case sycl::detail::CGType::CodeplayHostTask: {
      // The unique_ptr to the `sycl::detail::HostTask` in the HostTask CG
      // prevents from copying the CG.
      // We overcome this restriction by creating a new CG with the same data.
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
              std::move(HostTaskSPtr), CommandGroupPtr->MQueue,
              CommandGroupPtr->MContext, std::move(NewArgs), std::move(Data),
              CommandGroupPtr->getType(), Loc));
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
    case sycl::detail::CGType::Kernel: {
      sycl::detail::CGExecKernel *ExecKernelA =
          static_cast<sycl::detail::CGExecKernel *>(MCommandGroup.get());
      sycl::detail::CGExecKernel *ExecKernelB =
          static_cast<sycl::detail::CGExecKernel *>(Node->MCommandGroup.get());
      return ExecKernelA->MKernelName.compare(ExecKernelB->MKernelName) == 0;
    }
    case sycl::detail::CGType::CopyUSM: {
      sycl::detail::CGCopyUSM *CopyA =
          static_cast<sycl::detail::CGCopyUSM *>(MCommandGroup.get());
      sycl::detail::CGCopyUSM *CopyB =
          static_cast<sycl::detail::CGCopyUSM *>(Node->MCommandGroup.get());
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

  /// Update the value of an accessor inside this node. Accessors must be
  /// handled specifically compared to other argument values.
  /// @param ArgIndex The index of the accessor arg to be updated
  /// @param Acc Pointer to the new accessor value
  void updateAccessor(int ArgIndex, const sycl::detail::AccessorBaseHost *Acc) {
    auto &Args =
        static_cast<sycl::detail::CGExecKernel *>(MCommandGroup.get())->MArgs;
    auto NewAccImpl = sycl::detail::getSyclObjImpl(*Acc);
    for (auto &Arg : Args) {
      if (Arg.MIndex != ArgIndex) {
        continue;
      }
      assert(Arg.MType == sycl::detail::kernel_param_kind_t::kind_accessor);

      // Find old accessor in accessor storage and replace with new one
      if (static_cast<sycl::detail::SYCLMemObjT *>(NewAccImpl->MSYCLMemObj)
              ->needsWriteBack()) {
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Accessors to buffers which have write_back enabled "
            "are not allowed to be used in command graphs.");
      }

      // All accessors passed to this function will be placeholders, so we must
      // perform steps similar to what happens when handler::require() is
      // called here.
      sycl::detail::Requirement *NewReq = NewAccImpl.get();
      if (NewReq->MAccessMode != sycl::access_mode::read) {
        auto SYCLMemObj =
            static_cast<sycl::detail::SYCLMemObjT *>(NewReq->MSYCLMemObj);
        SYCLMemObj->handleWriteAccessorCreation();
      }

      for (auto &Acc : MCommandGroup->getAccStorage()) {
        if (auto OldAcc =
                static_cast<sycl::detail::AccessorImplHost *>(Arg.MPtr);
            Acc.get() == OldAcc) {
          Acc = NewAccImpl;
        }
      }

      for (auto &Req : MCommandGroup->getRequirements()) {
        if (auto OldReq =
                static_cast<sycl::detail::AccessorImplHost *>(Arg.MPtr);
            Req == OldReq) {
          Req = NewReq;
        }
      }
      Arg.MPtr = NewAccImpl.get();
      break;
    }
  }

  void updateArgValue(int ArgIndex, const void *NewValue, size_t Size) {

    auto &Args =
        static_cast<sycl::detail::CGExecKernel *>(MCommandGroup.get())->MArgs;
    for (auto &Arg : Args) {
      if (Arg.MIndex != ArgIndex) {
        continue;
      }
      assert(Arg.MSize == static_cast<int>(Size));
      // MPtr may be a pointer into arg storage so we memcpy the contents of
      // NewValue rather than assign it directly
      std::memcpy(Arg.MPtr, NewValue, Size);
      break;
    }
  }

  template <int Dimensions>
  void updateNDRange(nd_range<Dimensions> ExecutionRange) {
    if (MCGType != sycl::detail::CGType::Kernel) {
      throw sycl::exception(
          sycl::errc::invalid,
          "Cannot update execution range of nodes which are not kernel nodes");
    }
    if (!MNDRangeUsed) {
      throw sycl::exception(sycl::errc::invalid,
                            "Cannot update node which was created with a "
                            "sycl::range with a sycl::nd_range");
    }

    auto &NDRDesc =
        static_cast<sycl::detail::CGExecKernel *>(MCommandGroup.get())
            ->MNDRDesc;

    if (NDRDesc.Dims != Dimensions) {
      throw sycl::exception(sycl::errc::invalid,
                            "Cannot update execution range of a node with an "
                            "execution range of different dimensions than what "
                            "the node was originall created with.");
    }

    NDRDesc = sycl::detail::NDRDescT{ExecutionRange};
  }

  template <int Dimensions> void updateRange(range<Dimensions> ExecutionRange) {
    if (MCGType != sycl::detail::CGType::Kernel) {
      throw sycl::exception(
          sycl::errc::invalid,
          "Cannot update execution range of nodes which are not kernel nodes");
    }
    if (MNDRangeUsed) {
      throw sycl::exception(sycl::errc::invalid,
                            "Cannot update node which was created with a "
                            "sycl::nd_range with a sycl::range");
    }

    auto &NDRDesc =
        static_cast<sycl::detail::CGExecKernel *>(MCommandGroup.get())
            ->MNDRDesc;

    if (NDRDesc.Dims != Dimensions) {
      throw sycl::exception(sycl::errc::invalid,
                            "Cannot update execution range of a node with an "
                            "execution range of different dimensions than what "
                            "the node was originall created with.");
    }

    NDRDesc = sycl::detail::NDRDescT{ExecutionRange};
  }

  void updateFromOtherNode(const std::shared_ptr<node_impl> &Other) {
    auto ExecCG =
        static_cast<sycl::detail::CGExecKernel *>(MCommandGroup.get());
    auto OtherExecCG =
        static_cast<sycl::detail::CGExecKernel *>(Other->MCommandGroup.get());

    ExecCG->MArgs = OtherExecCG->MArgs;
    ExecCG->MNDRDesc = OtherExecCG->MNDRDesc;
    ExecCG->getAccStorage() = OtherExecCG->getAccStorage();
    ExecCG->getRequirements() = OtherExecCG->getRequirements();

    auto &OldArgStorage = OtherExecCG->getArgsStorage();
    auto &NewArgStorage = ExecCG->getArgsStorage();
    // Rebuild the arg storage and update the args
    rebuildArgStorage(ExecCG->MArgs, OldArgStorage, NewArgStorage);
  }

  id_type getID() const { return MID; }

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
               << " Length: " << Prefetch->getLength() << "\\n";
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
  /// List of predecessors to this partition.
  std::vector<std::shared_ptr<partition>> MPredecessors;
  /// True if the graph of this partition is a single path graph
  /// and in-order optmization can be applied on it.
  bool MIsInOrderGraph = false;

  /// @return True if the partition contains a host task
  bool isHostTask() const {
    return (MRoots.size() && ((*MRoots.begin()).lock()->MCGType ==
                              sycl::detail::CGType::CodeplayHostTask));
  }

  /// Checks if the graph is single path, i.e. each node has a single successor.
  /// @return True if the graph is a single path
  bool checkIfGraphIsSinglePath() {
    if (MRoots.size() > 1) {
      return false;
    }
    for (const auto &Node : MSchedule) {
      // In version 1.3.28454 of the L0 driver, 2D Copy ops cannot not
      // be enqueued in an in-order cmd-list (causing execution to stall).
      // The 2D Copy test should be removed from here when the bug is fixed.
      if ((Node->MSuccessors.size() > 1) || (Node->isNDCopyNode())) {
        return false;
      }
    }

    return true;
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

    if (!SyclDevice.has(aspect::ext_oneapi_limited_graph) &&
        !SyclDevice.has(aspect::ext_oneapi_graph)) {
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
  /// @param NodeType User facing type of the node.
  /// @param CommandGroup The CG which stores all information for this node.
  /// @param Dep Dependencies of the created node.
  /// @return Created node in the graph.
  std::shared_ptr<node_impl>
  add(node_type NodeType, std::unique_ptr<sycl::detail::CG> CommandGroup,
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
  /// @param Impl Graph implementation pointer.
  /// @param Dep List of predecessor nodes.
  /// @return Created node in the graph.
  std::shared_ptr<node_impl>
  add(const std::shared_ptr<graph_impl> &Impl,
      const std::vector<std::shared_ptr<node_impl>> &Dep = {});

  /// Create an empty node in the graph.
  /// @param Impl Graph implementation pointer.
  /// @param Events List of events associated to this node.
  /// @return Created node in the graph.
  std::shared_ptr<node_impl>
  add(const std::shared_ptr<graph_impl> &Impl,
      const std::vector<sycl::detail::EventImplPtr> Events);

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
  /// @param GraphImpl shared_ptr to Graph impl associated with this event, aka
  /// this.
  /// @param EventImpl Event to associate with a node in map.
  /// @param NodeImpl Node to associate with event in map.
  void addEventForNode(std::shared_ptr<graph_impl> GraphImpl,
                       std::shared_ptr<sycl::detail::event_impl> EventImpl,
                       std::shared_ptr<node_impl> NodeImpl) {
    if (!(EventImpl->getCommandGraph()))
      EventImpl->setCommandGraph(GraphImpl);
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

  /// Find the node associated with a SYCL event. Throws if no node is found for
  /// the given event.
  /// @param EventImpl Event to find the node for.
  /// @return Node associated with the event.
  std::shared_ptr<node_impl>
  getNodeForEvent(std::shared_ptr<sycl::detail::event_impl> EventImpl) {
    ReadLock Lock(MMutex);

    if (auto NodeFound = MEventsMap.find(EventImpl);
        NodeFound != std::end(MEventsMap)) {
      return NodeFound->second;
    }

    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "No node in this graph is associated with this event");
  }

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
  /// output nodes of this graph associated with a specific queue.
  /// @param[in] Queue The queue exit nodes must have been recorded from.
  /// @return vector of events associated to exit nodes.
  std::vector<sycl::detail::EventImplPtr>
  getExitNodesEvents(std::weak_ptr<sycl::detail::queue_impl> Queue);

  /// Store the last barrier node that was submitted to the queue.
  /// @param[in] Queue The queue the barrier was recorded from.
  /// @param[in] BarrierNodeImpl The created barrier node.
  void setBarrierDep(std::weak_ptr<sycl::detail::queue_impl> Queue,
                     std::shared_ptr<node_impl> BarrierNodeImpl) {
    MBarrierDependencyMap[Queue] = BarrierNodeImpl;
  }

  /// Get the last barrier node that was submitted to the queue.
  /// @param[in] Queue The queue to find the last barrier node of. An empty
  /// shared_ptr is returned if no barrier node has been recorded to the queue.
  std::shared_ptr<node_impl>
  getBarrierDep(std::weak_ptr<sycl::detail::queue_impl> Queue) {
    return MBarrierDependencyMap[Queue];
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
  /// @param Impl Graph implementation pointer.
  /// @param NodeList List of nodes from sub-graph in schedule order.
  /// @return An empty node is used to schedule dependencies on this sub-graph.
  std::shared_ptr<node_impl>
  addNodesToExits(const std::shared_ptr<graph_impl> &Impl,
                  const std::list<std::shared_ptr<node_impl>> &NodeList);

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

  /// Mapping from queues to barrier nodes. For each queue the last barrier
  /// node recorded to the graph from the queue is stored.
  std::map<std::weak_ptr<sycl::detail::queue_impl>, std::shared_ptr<node_impl>,
           std::owner_less<std::weak_ptr<sycl::detail::queue_impl>>>
      MBarrierDependencyMap;
};

/// Class representing the implementation of command_graph<executable>.
class exec_graph_impl {
public:
  using ReadLock = std::shared_lock<std::shared_mutex>;
  using WriteLock = std::unique_lock<std::shared_mutex>;

  /// Protects all the fields that can be changed by class' methods.
  mutable std::shared_mutex MMutex;

  /// Constructor.
  ///
  /// Nodes from GraphImpl will be copied when constructing this
  /// exec_graph_impl so that nodes may be modified (e.g. when merging subgraph
  /// nodes).
  /// @param Context Context to create graph with.
  /// @param GraphImpl Modifiable graph implementation to create with.
  /// @param PropList List of properties for constructing this object
  exec_graph_impl(sycl::context Context,
                  const std::shared_ptr<graph_impl> &GraphImpl,
                  const property_list &PropList);

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

  /// Query for the device tied to this graph.
  /// @return Device associated with graph.
  sycl::device getDevice() const { return MDevice; }

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

  /// Returns a list of all the accessor requirements for this graph.
  std::vector<sycl::detail::AccessorImplHost *> getRequirements() const {
    return MRequirements;
  }

  void update(std::shared_ptr<graph_impl> GraphImpl);
  void update(std::shared_ptr<node_impl> Node);
  void update(const std::vector<std::shared_ptr<node_impl>> Nodes);

  void updateImpl(std::shared_ptr<node_impl> NodeImpl);

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

  /// Duplicate nodes from the modifiable graph associated with this executable
  /// graph and store them locally. Any subgraph nodes in the modifiable graph
  /// will be expanded and merged into this new set of nodes.
  void duplicateNodes();

  /// Prints the contents of the graph to a text file in DOT format.
  /// @param FilePath Path to the output file.
  /// @param Verbose If true, print additional information about the nodes such
  /// as kernel args or memory access where applicable.
  void printGraphAsDot(const std::string FilePath, bool Verbose) const {
    /// Vector of nodes visited during the graph printing
    std::vector<node_impl *> VisitedNodes;

    std::fstream Stream(FilePath, std::ios::out);
    Stream << "digraph dot {" << std::endl;

    std::vector<std::shared_ptr<node_impl>> Roots;
    for (auto &Node : MNodeStorage) {
      if (Node->MPredecessors.size() == 0) {
        Roots.push_back(Node);
      }
    }

    for (std::shared_ptr<node_impl> Node : Roots)
      Node->printDotRecursive(Stream, VisitedNodes, Verbose);

    Stream << "}" << std::endl;

    Stream.close();
  }

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
  /// Device associated with this executable graph.
  sycl::device MDevice;
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
  /// Storage for copies of nodes from the original modifiable graph.
  std::vector<std::shared_ptr<node_impl>> MNodeStorage;
  /// Map of nodes to their associated PI command handles.
  std::unordered_map<std::shared_ptr<node_impl>,
                     sycl::detail::pi::PiExtCommandBufferCommand>
      MCommandMap;
  /// True if this graph can be updated (set with property::updatable)
  bool MIsUpdatable;
  /// If true, the graph profiling is enabled.
  bool MEnableProfiling;

  // Stores a cache of node ids from modifiable graph nodes to the companion
  // node(s) in this graph. Used for quick access when updating this graph.
  std::multimap<node_impl::id_type, std::shared_ptr<node_impl>> MIDCache;
};

class dynamic_parameter_impl {
public:
  dynamic_parameter_impl(std::shared_ptr<graph_impl> GraphImpl,
                         size_t ParamSize, const void *Data)
      : MGraph(GraphImpl), MValueStorage(ParamSize) {
    std::memcpy(MValueStorage.data(), Data, ParamSize);
  }

  /// Register a node with this dynamic parameter
  /// @param NodeImpl The node to be registered
  /// @param ArgIndex The arg index for the kernel arg associated with this
  /// dynamic_parameter in NodeImpl
  void registerNode(std::shared_ptr<node_impl> NodeImpl, int ArgIndex) {
    MNodes.emplace_back(NodeImpl, ArgIndex);
  }

  /// Get a pointer to the internal value of this dynamic parameter
  void *getValue() { return MValueStorage.data(); }

  /// Update the internal value of this dynamic parameter as well as the value
  /// of this parameter in all registered nodes.
  /// @param NewValue Pointer to the new value
  /// @param Size Size of the data pointer to by NewValue
  void updateValue(const void *NewValue, size_t Size) {
    for (auto &[NodeWeak, ArgIndex] : MNodes) {
      auto NodeShared = NodeWeak.lock();
      if (NodeShared) {
        NodeShared->updateArgValue(ArgIndex, NewValue, Size);
      }
    }
    std::memcpy(MValueStorage.data(), NewValue, Size);
  }

  /// Update the internal value of this dynamic parameter as well as the value
  /// of this parameter in all registered nodes. Should only be called for
  /// accessor dynamic_parameters.
  /// @param Acc The new accessor value
  void updateAccessor(const sycl::detail::AccessorBaseHost *Acc) {
    for (auto &[NodeWeak, ArgIndex] : MNodes) {
      auto NodeShared = NodeWeak.lock();
      // Should we fail here if the node isn't alive anymore?
      if (NodeShared) {
        NodeShared->updateAccessor(ArgIndex, Acc);
      }
    }
    std::memcpy(MValueStorage.data(), Acc,
                sizeof(sycl::detail::AccessorBaseHost));
  }

  // Weak ptrs to node_impls which will be updated
  std::vector<std::pair<std::weak_ptr<node_impl>, int>> MNodes;

  std::shared_ptr<graph_impl> MGraph;
  std::vector<std::byte> MValueStorage;
};

} // namespace detail
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
