//==--------- dynamic_impl.cpp - SYCL graph extension ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __SYCL_GRAPH_DYNAMIC_PARAM_IMPL_CPP

#include "dynamic_impl.hpp"
#include "node_impl.hpp"               // for NodeShared
#include <detail/cg.hpp>               // for CG
#include <detail/handler_impl.hpp>     // for handler_impl
#include <detail/sycl_mem_obj_t.hpp>   // for SYCLMemObjT
#include <sycl/detail/kernel_desc.hpp> // for kernel_param_kind_t
#include <sycl/ext/oneapi/experimental/detail/properties/graph_properties.hpp> // for checkGraphPropertiesAndThrow
#include <sycl/ext/oneapi/experimental/graph/command_graph.hpp> // for command_graph
#include <sycl/ext/oneapi/experimental/graph/common.hpp> // for graph_state
#include <sycl/ext/oneapi/experimental/graph/dynamic.hpp> // for dynamic parameters

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
namespace detail {

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
dynamic_parameter_base::dynamic_parameter_base()
    : impl(std::make_shared<dynamic_parameter_impl>()) {}
#endif

dynamic_parameter_base::dynamic_parameter_base(
    const std::shared_ptr<detail::dynamic_parameter_impl> &impl)
    : impl(impl) {}

dynamic_parameter_base::dynamic_parameter_base(
    const command_graph<graph_state::modifiable>)
    : impl(std::make_shared<dynamic_parameter_impl>()) {}
dynamic_parameter_base::dynamic_parameter_base(
    const command_graph<graph_state::modifiable>, size_t ParamSize,
    const void *Data)
    : impl(std::make_shared<dynamic_parameter_impl>(ParamSize, Data)) {}

void dynamic_parameter_base::updateValue(const void *NewValue, size_t Size) {
  impl->updateValue(NewValue, Size);
}

void dynamic_parameter_base::updateValue(const raw_kernel_arg *NewRawValue,
                                         size_t Size) {
  impl->updateValue(NewRawValue, Size);
}

void dynamic_parameter_base::updateAccessor(
    const sycl::detail::AccessorBaseHost *Acc) {
  impl->updateAccessor(Acc);
}

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
dynamic_work_group_memory_base::dynamic_work_group_memory_base(
    size_t BufferSizeInBytes)
    : dynamic_parameter_base(
          std::make_shared<dynamic_work_group_memory_impl>(BufferSizeInBytes)) {
}
#endif

dynamic_work_group_memory_base::dynamic_work_group_memory_base(
    const experimental::command_graph<graph_state::modifiable> /* Graph */,
    size_t BufferSizeInBytes)
    : dynamic_parameter_base(
          std::make_shared<dynamic_work_group_memory_impl>(BufferSizeInBytes)) {
}

void dynamic_work_group_memory_base::updateWorkGroupMem(
    size_t NewBufferSizeInBytes) {
  static_cast<dynamic_work_group_memory_impl *>(impl.get())
      ->updateWorkGroupMem(NewBufferSizeInBytes);
}

dynamic_local_accessor_base::dynamic_local_accessor_base(
    sycl::range<3> AllocationSize, int Dims, int ElemSize,
    const property_list &PropList)
    : dynamic_parameter_base(std::make_shared<dynamic_local_accessor_impl>(
          AllocationSize, Dims, ElemSize, PropList)) {}

void dynamic_local_accessor_base::updateLocalAccessor(
    sycl::range<3> NewAllocationSize) {
  static_cast<dynamic_local_accessor_impl *>(impl.get())
      ->updateLocalAccessor(NewAllocationSize);
}

void dynamic_parameter_impl::registerNode(node_impl &NodeImpl, int ArgIndex) {
  MNodes.emplace_back(NodeImpl.weak_from_this(), ArgIndex);
}

void dynamic_parameter_impl::updateValue(const raw_kernel_arg *NewRawValue,
                                         size_t Size) {
  // Number of bytes is taken from member of raw_kernel_arg object rather
  // than using the size parameter which represents sizeof(raw_kernel_arg).
  std::ignore = Size;
  size_t RawArgSize = NewRawValue->MArgSize;
  const void *RawArgData = NewRawValue->MArgData;

  updateValue(RawArgData, RawArgSize);
}

void dynamic_parameter_impl::updateValue(const void *NewValue, size_t Size) {
  for (auto &[NodeWeak, ArgIndex] : MNodes) {
    auto NodeShared = NodeWeak.lock();
    if (NodeShared) {
      dynamic_parameter_impl::updateCGArgValue(NodeShared->MCommandGroup,
                                               ArgIndex, NewValue, Size);
    }
  }

  for (auto &DynCGInfo : MDynCGs) {
    auto DynCG = DynCGInfo.DynCG.lock();
    if (DynCG) {
      auto &CG = DynCG->MCommandGroups[DynCGInfo.CGIndex];
      dynamic_parameter_impl::updateCGArgValue(CG, DynCGInfo.ArgIndex, NewValue,
                                               Size);
    }
  }

  std::memcpy(MValueStorage.data(), NewValue, Size);
}

void dynamic_parameter_impl::updateAccessor(
    const sycl::detail::AccessorBaseHost *Acc) {
  for (auto &[NodeWeak, ArgIndex] : MNodes) {
    auto NodeShared = NodeWeak.lock();
    // Should we fail here if the node isn't alive anymore?
    if (NodeShared) {
      dynamic_parameter_impl::updateCGAccessor(NodeShared->MCommandGroup,
                                               ArgIndex, Acc);
    }
  }

  for (auto &DynCGInfo : MDynCGs) {
    auto DynCG = DynCGInfo.DynCG.lock();
    if (DynCG) {
      auto &CG = DynCG->MCommandGroups[DynCGInfo.CGIndex];
      dynamic_parameter_impl::updateCGAccessor(CG, DynCGInfo.ArgIndex, Acc);
    }
  }

  std::memcpy(MValueStorage.data(), Acc,
              sizeof(sycl::detail::AccessorBaseHost));
}

void dynamic_parameter_impl::updateCGArgValue(
    std::shared_ptr<sycl::detail::CG> CG, int ArgIndex, const void *NewValue,
    size_t Size) {
  auto &Args = static_cast<sycl::detail::CGExecKernel *>(CG.get())->MArgs;
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

void dynamic_parameter_impl::updateCGAccessor(
    std::shared_ptr<sycl::detail::CG> CG, int ArgIndex,
    const sycl::detail::AccessorBaseHost *Acc) {
  auto &Args = static_cast<sycl::detail::CGExecKernel *>(CG.get())->MArgs;

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

    for (auto &Acc : CG->getAccStorage()) {
      if (auto OldAcc = static_cast<sycl::detail::AccessorImplHost *>(Arg.MPtr);
          Acc.get() == OldAcc) {
        Acc = NewAccImpl;
      }
    }

    for (auto &Req : CG->getRequirements()) {
      if (auto OldReq = static_cast<sycl::detail::AccessorImplHost *>(Arg.MPtr);
          Req == OldReq) {
        Req = NewReq;
      }
    }
    Arg.MPtr = NewAccImpl.get();
    break;
  }
}

void dynamic_work_group_memory_impl::updateWorkGroupMem(
    size_t NewBufferSizeInBytes) {
  for (auto &[NodeWeak, ArgIndex] : MNodes) {
    auto NodeShared = NodeWeak.lock();
    if (NodeShared) {
      dynamic_work_group_memory_impl::updateCGWorkGroupMem(
          NodeShared->MCommandGroup, ArgIndex, NewBufferSizeInBytes);
    }
  }

  for (auto &DynCGInfo : MDynCGs) {
    auto DynCG = DynCGInfo.DynCG.lock();
    if (DynCG) {
      auto &CG = DynCG->MCommandGroups[DynCGInfo.CGIndex];
      dynamic_work_group_memory_impl::updateCGWorkGroupMem(
          CG, DynCGInfo.ArgIndex, NewBufferSizeInBytes);
    }
  }
}

void dynamic_work_group_memory_impl::updateCGWorkGroupMem(
    std::shared_ptr<sycl::detail::CG> &CG, int ArgIndex,
    size_t NewBufferSizeInBytes) {

  auto &Args = static_cast<sycl::detail::CGExecKernel *>(CG.get())->MArgs;
  for (auto &Arg : Args) {
    if (Arg.MIndex != ArgIndex) {
      continue;
    }
    assert(Arg.MType == sycl::detail::kernel_param_kind_t::kind_std_layout);
    Arg.MSize = NewBufferSizeInBytes;
    break;
  }
}

dynamic_local_accessor_impl::dynamic_local_accessor_impl(
    sycl::range<3> AllocationSize, int Dims, int ElemSize,
    const property_list &PropList)
    : dynamic_parameter_impl(),
      LAccImplHost(AllocationSize, Dims, ElemSize, {}) {
  checkGraphPropertiesAndThrow(PropList);
}

void dynamic_local_accessor_impl::updateLocalAccessor(
    range<3> NewAllocationSize) {
  for (auto &[NodeWeak, ArgIndex] : MNodes) {
    auto NodeShared = NodeWeak.lock();
    if (NodeShared) {
      dynamic_local_accessor_impl::updateCGLocalAccessor(
          NodeShared->MCommandGroup, ArgIndex, NewAllocationSize);
    }
  }

  for (auto &DynCGInfo : MDynCGs) {
    auto DynCG = DynCGInfo.DynCG.lock();
    if (DynCG) {
      auto &CG = DynCG->MCommandGroups[DynCGInfo.CGIndex];
      dynamic_local_accessor_impl::updateCGLocalAccessor(CG, DynCGInfo.ArgIndex,
                                                         NewAllocationSize);
    }
  }
}

void dynamic_local_accessor_impl::updateCGLocalAccessor(
    std::shared_ptr<sycl::detail::CG> &CG, int ArgIndex,
    range<3> NewAllocationSize) {

  auto &Args = static_cast<sycl::detail::CGExecKernel *>(CG.get())->MArgs;
  for (auto &Arg : Args) {
    if (Arg.MIndex != ArgIndex) {
      continue;
    }
    assert(Arg.MType == sycl::detail::kernel_param_kind_t::kind_std_layout);

    // Update the local memory Size Argument
    Arg.MSize = NewAllocationSize.size() * LAccImplHost.MElemSize;

    // MSize is used as an argument to the AccField kernel parameters.
    LAccImplHost.MSize = NewAllocationSize;

    break;
  }
}

dynamic_command_group_impl::dynamic_command_group_impl(
    const command_graph<graph_state::modifiable> &Graph)
    : MGraph{sycl::detail::getSyclObjImpl(Graph)}, MActiveCGF(0),
      MID(NextAvailableID.fetch_add(1, std::memory_order_relaxed)) {}

void dynamic_command_group_impl::finalizeCGFList(
    const std::vector<std::function<void(handler &)>> &CGFList) {
  for (size_t CGFIndex = 0; CGFIndex < CGFList.size(); CGFIndex++) {
    const auto &CGF = CGFList[CGFIndex];
    // Handler defined inside the loop so it doesn't appear to the runtime
    // as a single command-group with multiple commands inside.
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
    detail::handler_impl HandlerImpl{*MGraph};
    sycl::handler Handler{HandlerImpl};
#else
    sycl::handler Handler{MGraph};
#endif
    CGF(Handler);

    if (Handler.getType() != sycl::detail::CGType::Kernel &&
        Handler.getType() != sycl::detail::CGType::CodeplayHostTask) {
      throw sycl::exception(
          make_error_code(errc::invalid),
          "The only types of command-groups that can be used in "
          "dynamic command-groups are kernels and host-tasks.");
    }

    // We need to store the first CG's type so we can check they are all the
    // same
    if (CGFIndex == 0) {
      MCGType = Handler.getType();
    } else if (MCGType != Handler.getType()) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Command-groups in a dynamic command-group must "
                            "all be the same type.");
    }

    Handler.finalize();

    // Take unique_ptr<detail::CG> object from handler and convert to
    // shared_ptr<detail::CG> to store
    sycl::detail::CG *RawCGPtr = Handler.impl->MGraphNodeCG.release();
    MCommandGroups.push_back(std::shared_ptr<sycl::detail::CG>(RawCGPtr));

    // Track dynamic_parameter usage in command-group
    auto &DynamicParams = Handler.impl->MKernelData.getDynamicParameters();

    if (DynamicParams.size() > 0 &&
        Handler.getType() == sycl::detail::CGType::CodeplayHostTask) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Cannot use dynamic parameters in a host_task");
    }
    for (auto &[DynamicParam, ArgIndex] : DynamicParams) {
      DynamicParam->registerDynCG(shared_from_this(), CGFIndex, ArgIndex);
    }
  }

  // Host tasks don't need to store alternative kernels
  if (MCGType == sycl::detail::CGType::CodeplayHostTask) {
    return;
  }

  // For each Kernel CG store the list of alternative kernels, not
  // including itself.
  using CGExecKernelSP = std::shared_ptr<sycl::detail::CGExecKernel>;
  using CGExecKernelWP = std::weak_ptr<sycl::detail::CGExecKernel>;
  for (std::shared_ptr<sycl::detail::CG> CommandGroup : MCommandGroups) {
    CGExecKernelSP KernelCG =
        std::dynamic_pointer_cast<sycl::detail::CGExecKernel>(CommandGroup);
    std::vector<CGExecKernelWP> Alternatives;

    // Add all other command groups except for the current one to the list of
    // alternatives
    for (auto &OtherCG : MCommandGroups) {
      CGExecKernelSP OtherKernelCG =
          std::dynamic_pointer_cast<sycl::detail::CGExecKernel>(OtherCG);
      if (KernelCG != OtherKernelCG) {
        Alternatives.push_back(OtherKernelCG);
      }
    }

    KernelCG->MAlternativeKernels = std::move(Alternatives);
  }
}

void dynamic_command_group_impl::setActiveIndex(size_t Index) {
  if (Index >= getNumCGs()) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Index is out of range.");
  }
  MActiveCGF = Index;

  // Update nodes using the dynamic command-group to use the new active CG
  for (auto &Node : MNodes) {
    if (auto NodeSP = Node.lock()) {
      NodeSP->MCommandGroup = getActiveCG();
    }
  }
}
} // namespace detail

dynamic_command_group::dynamic_command_group(
    const command_graph<graph_state::modifiable> &Graph,
    const std::vector<std::function<void(handler &)>> &CGFList)
    : impl(std::make_shared<detail::dynamic_command_group_impl>(Graph)) {
  if (CGFList.empty()) {
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Dynamic command-group cannot be created with an "
                          "empty CGF list.");
  }
  impl->finalizeCGFList(CGFList);
}

size_t dynamic_command_group::get_active_index() const {
  return impl->getActiveIndex();
}
void dynamic_command_group::set_active_index(size_t Index) {
  return impl->setActiveIndex(Index);
}
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl

size_t
std::hash<sycl::ext::oneapi::experimental::dynamic_command_group>::operator()(
    const sycl::ext::oneapi::experimental::dynamic_command_group &DynamicCG)
    const {
  auto ID = sycl::detail::getSyclObjImpl(DynamicCG)->getID();
  return std::hash<decltype(ID)>()(ID);
}
