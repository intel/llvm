//==-------------- CG.hpp - SYCL standard header file ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/detail/cg_types.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/host_profiling_info.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/group.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/interop_handle.hpp>
#include <CL/sycl/interop_handler.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/nd_item.hpp>
#include <CL/sycl/range.hpp>

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declarations
class queue;

namespace detail {

// Periodically there is a need to extend handler and CG classes to hold more
// data(members) than it has now. But any modification of the layout of those
// classes is an ABI break. To have an ability to have more data the following
// approach is implemented:
//
// Those classes have a member - MSharedPtrStorage which is an std::vector of
// std::shared_ptr's and is supposed to hold reference counters of user
// provided shared_ptr's.
//
// The first element of this vector is reused to store a vector of additional
// members handler and CG need to have.
//
// These additional arguments are represented using "ExtendedMemberT" structure
// which has a pointer to an arbitrary value and an integer which is used to
// understand how the value the pointer points to should be interpreted.
//
// ========  ========      ========
// |      |  |      | ...  |      | std::vector<std::shared_ptr<void>>
// ========  ========      ========
//    ||        ||            ||
//    ||        \/            \/
//    ||       user          user
//    ||       data          data
//    \/
// ========  ========      ========
// | Type |  | Type | ...  | Type | std::vector<ExtendedMemberT>
// |      |  |      |      |      |
// | Ptr  |  | Ptr  | ...  | Ptr  |
// ========  ========      ========
//
// Prior to this change this vector was supposed to have user's values only, so
// it is not legal to expect that the first argument is a special one.
// Versioning is implemented to overcome this problem - if the first element of
// the MSharedPtrStorage is a pointer to the special vector then CGType value
// has version "1" encoded.
//
// The version of CG type is encoded in the highest byte of the value:
//
// 0x00000001 - CG type KERNEL version 0
// 0x01000001 - CG type KERNEL version 1
//    ^
//    |
// The byte specifies the version
//
// A user of this vector should not expect that a specific data is stored at a
// specific position, but iterate over all looking for an ExtendedMemberT value
// with the desired type.
// This allows changing/extending the contents of this vector without changing
// the version.
//

// Used to represent a type of an extended member
enum class ExtendedMembersType : unsigned int {
  HANDLER_KERNEL_BUNDLE = 0,
  HANDLER_MEM_ADVICE,
  // handler_impl is stored in the exended members to avoid breaking ABI.
  // TODO: This should be made a member of the handler class once ABI can be
  //       broken.
  HANDLER_IMPL,
};

// Holds a pointer to an object of an arbitrary type and an ID value which
// should be used to understand what type pointer points to.
// Used as to extend handler class without introducing new class members which
// would change handler layout.
struct ExtendedMemberT {
  ExtendedMembersType MType;
  std::shared_ptr<void> MData;
};

static std::shared_ptr<std::vector<ExtendedMemberT>>
convertToExtendedMembers(const std::shared_ptr<const void> &SPtr) {
  return std::const_pointer_cast<std::vector<ExtendedMemberT>>(
      std::static_pointer_cast<const std::vector<ExtendedMemberT>>(SPtr));
}

class stream_impl;
class queue_impl;
class kernel_bundle_impl;

// The constant is used to left shift a CG type value to access it's version
constexpr unsigned int ShiftBitsForVersion = 24;

// Constructs versioned type
constexpr unsigned int getVersionedCGType(unsigned int Type,
                                          unsigned char Version) {
  return Type | (static_cast<unsigned int>(Version) << ShiftBitsForVersion);
}

// Returns the type without version encoded
constexpr unsigned char getUnversionedCGType(unsigned int Type) {
  unsigned int Mask = -1;
  Mask >>= (sizeof(Mask) * 8 - ShiftBitsForVersion);
  return Type & Mask;
}

// Returns the version encoded to the type
constexpr unsigned char getCGTypeVersion(unsigned int Type) {
  return Type >> ShiftBitsForVersion;
}

/// Base class for all types of command groups.
class CG {
public:
  // Used to version CG and handler classes. Using unsigned char as the version
  // is encoded in the highest byte of CGType value. So it is not possible to
  // encode a value > 255 anyway which should be big enough room for version
  // bumping.
  enum class CG_VERSION : unsigned char {
    V0 = 0,
    V1 = 1,
  };

  /// Type of the command group.
  enum CGTYPE : unsigned int {
    None = 0,
    Kernel = 1,
    CopyAccToPtr = 2,
    CopyPtrToAcc = 3,
    CopyAccToAcc = 4,
    Barrier = 5,
    BarrierWaitlist = 6,
    Fill = 7,
    UpdateHost = 8,
    RunOnHostIntel = 9,
    CopyUSM = 10,
    FillUSM = 11,
    PrefetchUSM = 12,
    CodeplayInteropTask = 13,
    CodeplayHostTask = 14,
    AdviseUSM = 15,
  };

  CG(CGTYPE Type, std::vector<std::vector<char>> ArgsStorage,
     std::vector<detail::AccessorImplPtr> AccStorage,
     std::vector<std::shared_ptr<const void>> SharedPtrStorage,
     std::vector<Requirement *> Requirements,
     std::vector<detail::EventImplPtr> Events, detail::code_location loc = {})
      : MType(Type), MArgsStorage(std::move(ArgsStorage)),
        MAccStorage(std::move(AccStorage)),
        MSharedPtrStorage(std::move(SharedPtrStorage)),
        MRequirements(std::move(Requirements)), MEvents(std::move(Events)) {
    // Capture the user code-location from Q.submit(), Q.parallel_for()
    // etc for later use; if code location information is not available,
    // the file name and function name members will be empty strings
    if (loc.functionName())
      MFunctionName = loc.functionName();
    if (loc.fileName())
      MFileName = loc.fileName();
    MLine = loc.lineNumber();
    MColumn = loc.columnNumber();
  }

  CG(CG &&CommandGroup) = default;

  CGTYPE getType() { return static_cast<CGTYPE>(getUnversionedCGType(MType)); }

  CG_VERSION getVersion() {
    return static_cast<CG_VERSION>(getCGTypeVersion(MType));
  }

  std::shared_ptr<std::vector<ExtendedMemberT>> getExtendedMembers() {
    if (getCGTypeVersion(MType) == static_cast<unsigned int>(CG_VERSION::V0) ||
        MSharedPtrStorage.empty())
      return nullptr;

    // The first value in shared_ptr storage is supposed to store a vector of
    // extended members.
    return convertToExtendedMembers(MSharedPtrStorage[0]);
  }

  virtual ~CG() = default;

private:
  CGTYPE MType;
  // The following storages are needed to ensure that arguments won't die while
  // we are using them.
  /// Storage for standard layout arguments.
  std::vector<std::vector<char>> MArgsStorage;
  /// Storage for accessors.
  std::vector<detail::AccessorImplPtr> MAccStorage;
  /// Storage for shared_ptrs.
  std::vector<std::shared_ptr<const void>> MSharedPtrStorage;

public:
  /// List of requirements that specify which memory is needed for the command
  /// group to be executed.
  std::vector<Requirement *> MRequirements;
  /// List of events that order the execution of this CG
  std::vector<detail::EventImplPtr> MEvents;
  // Member variables to capture the user code-location
  // information from Q.submit(), Q.parallel_for() etc
  // Storage for function name and source file name
  std::string MFunctionName, MFileName;
  // Storage for line and column of code location
  int32_t MLine, MColumn;
};

/// "Execute kernel" command group class.
class CGExecKernel : public CG {
public:
  /// Stores ND-range description.
  NDRDescT MNDRDesc;
  std::unique_ptr<HostKernelBase> MHostKernel;
  std::shared_ptr<detail::kernel_impl> MSyclKernel;
  std::vector<ArgDesc> MArgs;
  std::string MKernelName;
  detail::OSModuleHandle MOSModuleHandle;
  std::vector<std::shared_ptr<detail::stream_impl>> MStreams;

  CGExecKernel(NDRDescT NDRDesc, std::unique_ptr<HostKernelBase> HKernel,
               std::shared_ptr<detail::kernel_impl> SyclKernel,
               std::vector<std::vector<char>> ArgsStorage,
               std::vector<detail::AccessorImplPtr> AccStorage,
               std::vector<std::shared_ptr<const void>> SharedPtrStorage,
               std::vector<Requirement *> Requirements,
               std::vector<detail::EventImplPtr> Events,
               std::vector<ArgDesc> Args, std::string KernelName,
               detail::OSModuleHandle OSModuleHandle,
               std::vector<std::shared_ptr<detail::stream_impl>> Streams,
               CGTYPE Type, detail::code_location loc = {})
      : CG(Type, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MNDRDesc(std::move(NDRDesc)), MHostKernel(std::move(HKernel)),
        MSyclKernel(std::move(SyclKernel)), MArgs(std::move(Args)),
        MKernelName(std::move(KernelName)), MOSModuleHandle(OSModuleHandle),
        MStreams(std::move(Streams)) {
    assert((getType() == RunOnHostIntel || getType() == Kernel) &&
           "Wrong type of exec kernel CG.");
  }

  std::vector<ArgDesc> getArguments() const { return MArgs; }
  std::string getKernelName() const { return MKernelName; }
  std::vector<std::shared_ptr<detail::stream_impl>> getStreams() const {
    return MStreams;
  }

  std::shared_ptr<detail::kernel_bundle_impl> getKernelBundle() {
    const std::shared_ptr<std::vector<ExtendedMemberT>> &ExtendedMembers =
        getExtendedMembers();
    if (!ExtendedMembers)
      return nullptr;
    for (const ExtendedMemberT &EMember : *ExtendedMembers)
      if (ExtendedMembersType::HANDLER_KERNEL_BUNDLE == EMember.MType)
        return std::static_pointer_cast<detail::kernel_bundle_impl>(
            EMember.MData);
    return nullptr;
  }

  void clearStreams() { MStreams.clear(); }
  bool hasStreams() { return !MStreams.empty(); }
};

/// "Copy memory" command group class.
class CGCopy : public CG {
  void *MSrc;
  void *MDst;

public:
  CGCopy(CGTYPE CopyType, void *Src, void *Dst,
         std::vector<std::vector<char>> ArgsStorage,
         std::vector<detail::AccessorImplPtr> AccStorage,
         std::vector<std::shared_ptr<const void>> SharedPtrStorage,
         std::vector<Requirement *> Requirements,
         std::vector<detail::EventImplPtr> Events,
         detail::code_location loc = {})
      : CG(CopyType, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MSrc(Src), MDst(Dst) {}
  void *getSrc() { return MSrc; }
  void *getDst() { return MDst; }
};

/// "Fill memory" command group class.
class CGFill : public CG {
public:
  std::vector<char> MPattern;
  Requirement *MPtr;

  CGFill(std::vector<char> Pattern, void *Ptr,
         std::vector<std::vector<char>> ArgsStorage,
         std::vector<detail::AccessorImplPtr> AccStorage,
         std::vector<std::shared_ptr<const void>> SharedPtrStorage,
         std::vector<Requirement *> Requirements,
         std::vector<detail::EventImplPtr> Events,
         detail::code_location loc = {})
      : CG(Fill, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MPattern(std::move(Pattern)), MPtr((Requirement *)Ptr) {}
  Requirement *getReqToFill() { return MPtr; }
};

/// "Update host" command group class.
class CGUpdateHost : public CG {
  Requirement *MPtr;

public:
  CGUpdateHost(void *Ptr, std::vector<std::vector<char>> ArgsStorage,
               std::vector<detail::AccessorImplPtr> AccStorage,
               std::vector<std::shared_ptr<const void>> SharedPtrStorage,
               std::vector<Requirement *> Requirements,
               std::vector<detail::EventImplPtr> Events,
               detail::code_location loc = {})
      : CG(UpdateHost, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MPtr((Requirement *)Ptr) {}

  Requirement *getReqToUpdate() { return MPtr; }
};

/// "Copy USM" command group class.
class CGCopyUSM : public CG {
  void *MSrc;
  void *MDst;
  size_t MLength;

public:
  CGCopyUSM(void *Src, void *Dst, size_t Length,
            std::vector<std::vector<char>> ArgsStorage,
            std::vector<detail::AccessorImplPtr> AccStorage,
            std::vector<std::shared_ptr<const void>> SharedPtrStorage,
            std::vector<Requirement *> Requirements,
            std::vector<detail::EventImplPtr> Events,
            detail::code_location loc = {})
      : CG(CopyUSM, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MSrc(Src), MDst(Dst), MLength(Length) {}

  void *getSrc() { return MSrc; }
  void *getDst() { return MDst; }
  size_t getLength() { return MLength; }
};

/// "Fill USM" command group class.
class CGFillUSM : public CG {
  std::vector<char> MPattern;
  void *MDst;
  size_t MLength;

public:
  CGFillUSM(std::vector<char> Pattern, void *DstPtr, size_t Length,
            std::vector<std::vector<char>> ArgsStorage,
            std::vector<detail::AccessorImplPtr> AccStorage,
            std::vector<std::shared_ptr<const void>> SharedPtrStorage,
            std::vector<Requirement *> Requirements,
            std::vector<detail::EventImplPtr> Events,
            detail::code_location loc = {})
      : CG(FillUSM, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MPattern(std::move(Pattern)), MDst(DstPtr), MLength(Length) {}
  void *getDst() { return MDst; }
  size_t getLength() { return MLength; }
  int getFill() { return MPattern[0]; }
};

/// "Prefetch USM" command group class.
class CGPrefetchUSM : public CG {
  void *MDst;
  size_t MLength;

public:
  CGPrefetchUSM(void *DstPtr, size_t Length,
                std::vector<std::vector<char>> ArgsStorage,
                std::vector<detail::AccessorImplPtr> AccStorage,
                std::vector<std::shared_ptr<const void>> SharedPtrStorage,
                std::vector<Requirement *> Requirements,
                std::vector<detail::EventImplPtr> Events,
                detail::code_location loc = {})
      : CG(PrefetchUSM, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MDst(DstPtr), MLength(Length) {}
  void *getDst() { return MDst; }
  size_t getLength() { return MLength; }
};

/// "Advise USM" command group class.
class CGAdviseUSM : public CG {
  void *MDst;
  size_t MLength;

public:
  CGAdviseUSM(void *DstPtr, size_t Length,
              std::vector<std::vector<char>> ArgsStorage,
              std::vector<detail::AccessorImplPtr> AccStorage,
              std::vector<std::shared_ptr<const void>> SharedPtrStorage,
              std::vector<Requirement *> Requirements,
              std::vector<detail::EventImplPtr> Events, CGTYPE Type,
              detail::code_location loc = {})
      : CG(Type, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MDst(DstPtr), MLength(Length) {}
  void *getDst() { return MDst; }
  size_t getLength() { return MLength; }

  pi_mem_advice getAdvice() {
    auto ExtendedMembers = getExtendedMembers();
    if (!ExtendedMembers)
      return PI_MEM_ADVISE_UNKNOWN;
    for (const ExtendedMemberT &EM : *ExtendedMembers)
      if ((ExtendedMembersType::HANDLER_MEM_ADVICE == EM.MType) && EM.MData)
        return *std::static_pointer_cast<pi_mem_advice>(EM.MData);
    return PI_MEM_ADVISE_UNKNOWN;
  }
};

class CGInteropTask : public CG {
public:
  std::unique_ptr<InteropTask> MInteropTask;

  CGInteropTask(std::unique_ptr<InteropTask> InteropTask,
                std::vector<std::vector<char>> ArgsStorage,
                std::vector<detail::AccessorImplPtr> AccStorage,
                std::vector<std::shared_ptr<const void>> SharedPtrStorage,
                std::vector<Requirement *> Requirements,
                std::vector<detail::EventImplPtr> Events, CGTYPE Type,
                detail::code_location loc = {})
      : CG(Type, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MInteropTask(std::move(InteropTask)) {}
};

class CGHostTask : public CG {
public:
  std::unique_ptr<HostTask> MHostTask;
  // queue for host-interop task
  std::shared_ptr<detail::queue_impl> MQueue;
  // context for host-interop task
  std::shared_ptr<detail::context_impl> MContext;
  std::vector<ArgDesc> MArgs;

  CGHostTask(std::unique_ptr<HostTask> HostTask,
             std::shared_ptr<detail::queue_impl> Queue,
             std::shared_ptr<detail::context_impl> Context,
             std::vector<ArgDesc> Args,
             std::vector<std::vector<char>> ArgsStorage,
             std::vector<detail::AccessorImplPtr> AccStorage,
             std::vector<std::shared_ptr<const void>> SharedPtrStorage,
             std::vector<Requirement *> Requirements,
             std::vector<detail::EventImplPtr> Events, CGTYPE Type,
             detail::code_location loc = {})
      : CG(Type, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MHostTask(std::move(HostTask)), MQueue(Queue), MContext(Context),
        MArgs(std::move(Args)) {}
};

class CGBarrier : public CG {
public:
  std::vector<detail::EventImplPtr> MEventsWaitWithBarrier;

  CGBarrier(std::vector<detail::EventImplPtr> EventsWaitWithBarrier,
            std::vector<std::vector<char>> ArgsStorage,
            std::vector<detail::AccessorImplPtr> AccStorage,
            std::vector<std::shared_ptr<const void>> SharedPtrStorage,
            std::vector<Requirement *> Requirements,
            std::vector<detail::EventImplPtr> Events, CGTYPE Type,
            detail::code_location loc = {})
      : CG(Type, std::move(ArgsStorage), std::move(AccStorage),
           std::move(SharedPtrStorage), std::move(Requirements),
           std::move(Events), std::move(loc)),
        MEventsWaitWithBarrier(std::move(EventsWaitWithBarrier)) {}
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
