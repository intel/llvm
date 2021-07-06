//==------------- PiMock.hpp --- Mock unit testing library -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This mini-library provides facilities to test the DPC++ Runtime behavior upon
// specific results of the underlying low-level API calls. By exploiting the
// Plugin Interface API, the stored addresses of the actual plugin-specific
// implementations can be overwritten to point at user-defined mock functions.
//
// While this could be done manually for each unit-testing scenario, the library
// aims to rule out the boilerplate, providing helper APIs which can be re-used
// by all such unit tests. The test code stemming from this can be more consise,
// with little difference from non-mock classes' usage.
//
// The following unit testing scenarios are thereby simplified:
// 1) testing the DPC++ RT management of specific PI return codes;
// 2) coverage of corner-cases related to specific data outputs
//    from underlying runtimes;
// 3) testing the order of PI API calls;
// ..., etc.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <detail/platform_impl.hpp>

#include <functional>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace unittest {

namespace detail = cl::sycl::detail;
namespace RT = detail::pi;

/// Overwrites the input PiPlugin's PiFunctionTable entry for the given PI API
/// with a given function pointer.
///
/// \param MPlugin is a pointer to the PiPlugin instance that will be modified.
/// \param FuncPtr is a pointer to the function that will override the original.
///        function table entry
#define _PI_API(api)                                                           \
  template <detail::PiApiKind PiApiOffset>                                     \
  inline void setFuncPtr(RT::PiPlugin *MPlugin, decltype(&::api) FuncPtr);     \
  template <>                                                                  \
  inline void setFuncPtr<detail::PiApiKind::api>(RT::PiPlugin * MPlugin,       \
                                                 decltype(&::api) FuncPtr) {   \
    MPlugin->PiFunctionTable.api = FuncPtr;                                    \
  }
#include <CL/sycl/detail/pi.def>
#undef _PI_API

/// The PiMock class wraps an instance of a SYCL platform class,
/// and manages all mock redefinitions for the underlying plugin.
///
/// Mock platform instances must not share the plugin resources with
/// any other SYCL platform within the given context. Otherwise, mock
/// redefinitions would also affect other platforms' behavior.
/// Therefore, any plugin-related information is fully copied whenever
/// a user-passed SYCL object instance is being mocked.
/// The underlying SYCL platform must be a non-host plaftorm to facilitate
/// plugin usage.
///
/// Simple usage examples would look like this:
/// ```
/// pi_result redefinePiProgramRetain(pi_program program) { /*code*/ }
/// /*...*/
/// unittest::PiMock Mock;
/// Mock.redefine<PiApiKind::piProgramRetain>(redefinePiProgramRetain);
/// platform &MockP = Mock.getPlatform();
/// /*...*/
/// ```
/// ```
/// pi_result redefinePiProgramRetain(pi_program program) { /*code*/ }
/// /*...*/
/// queue Q;
/// unittest::PiMock Mock(Q);
/// Mock.redefine<PiApiKind::piProgramRetain>(redefinePiProgramRetain);
/// Q.submit(/* expecting mock behavior */);
/// /*...*/
/// ```
// TODO: Consider reworking the class into a `detail::plugin` derivative.
class PiMock {
public:
  /// Constructs PiMock from a device_selector, provided that
  /// a non-host device can and will be selected. Default-constructs
  /// from a default_selector.
  ///
  /// \param DevSelector is a reference to a device_selector instance.
  explicit PiMock(const cl::sycl::device_selector &DevSelector =
                      cl::sycl::default_selector{})
      : PiMock(cl::sycl::platform{DevSelector}) {}

  /// Constructs PiMock from a queue.
  ///
  /// \param Queue is a reference to a SYCL queue to which
  ///        the mock redefinitions will apply.
  explicit PiMock(cl::sycl::queue &Queue)
      : PiMock(Queue.get_device().get_platform()) {}

  /// Constructs PiMock from a reference to a SYCL platform instance.
  ///
  /// A new plugin will be stored into the platform instance, which
  /// will no longer share the plugin with other platform instances
  /// within the given context. A separate platform instance will be
  /// held by the PiMock instance.
  ///
  /// \param OriginalPlatform is a reference to a SYCL platform.
  explicit PiMock(const cl::sycl::platform &OriginalPlatform) {
    assert(!OriginalPlatform.is_host() && "PI mock isn't supported for host");
    // Extract impl and plugin handles
    std::shared_ptr<detail::platform_impl> ImplPtr =
        detail::getSyclObjImpl(OriginalPlatform);
    const detail::plugin &OriginalPiPlugin = ImplPtr->getPlugin();
    // Copy the PiPlugin, thus untying our to-be mock platform from other
    // platforms within the context. Reset our platform to use the new plugin.
    auto NewPluginPtr = std::make_shared<detail::plugin>(
        OriginalPiPlugin.getPiPlugin(), OriginalPiPlugin.getBackend(),
        OriginalPiPlugin.getLibraryHandle());
    ImplPtr->setPlugin(NewPluginPtr);
    // Extract the new PiPlugin instance by a non-const pointer,
    // explicitly allowing modification
    MPiPluginMockPtr = &NewPluginPtr->getPiPlugin();
    // Save a copy of the platform resource
    MPlatform = OriginalPlatform;
  }

  /// Explicit construction from a host_selector is forbidden.
  PiMock(const cl::sycl::host_selector &HostSelector) = delete;

  PiMock(const PiMock &) = delete;
  PiMock &operator=(const PiMock &) = delete;
  ~PiMock() = default;

  /// Returns a handle to the SYCL platform instance.
  ///
  /// \return A reference to the SYCL platform.
  cl::sycl::platform &getPlatform() { return MPlatform; }

  template <detail::PiApiKind PiApiOffset>
  using FuncPtrT = typename RT::PiFuncInfo<PiApiOffset>::FuncPtrT;
  template <detail::PiApiKind PiApiOffset>
  using SignatureT = typename std::remove_pointer<FuncPtrT<PiApiOffset>>::type;

  /// Redefines the implementation of a given PI API to the input
  /// function object.
  ///
  /// \param Replacement is a mock std::function instance to be
  ///        called instead of the given PI API. This function must
  ///        not have been constructed from a lambda.
  template <detail::PiApiKind PiApiOffset>
  void redefine(const std::function<SignatureT<PiApiOffset>> &Replacement) {
    // TODO: Find a way to store FPointer first so that real PI functions can
    // be called alongside the mock ones. Something like:
    // `enum class MockPIPolicy { InsteadOf, Before, After};`
    // may need to be introduced.
    FuncPtrT<PiApiOffset> FuncPtr =
        *Replacement.template target<FuncPtrT<PiApiOffset>>();
    assert(FuncPtr &&
           "Function target is empty, try passing a lambda directly");
    setFuncPtr<PiApiOffset>(MPiPluginMockPtr, *FuncPtr);
  }

  /// A `redefine` overload for function pointer/captureless lambda
  /// arguments.
  ///
  /// \param Replacement is a mock callable assignable to a function
  ///        pointer (function pointer/captureless lambda).
  template <detail::PiApiKind PiApiOffset, typename FunctorT>
  void redefine(const FunctorT &Replacement) {
    // TODO: Check for matching signatures/assignability
    setFuncPtr<PiApiOffset>(MPiPluginMockPtr, Replacement);
  }

private:
  cl::sycl::platform MPlatform;
  // Extracted at initialization for convenience purposes. The resource
  // itself is owned by the platform instance.
  RT::PiPlugin *MPiPluginMockPtr;
};

} // namespace unittest
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
