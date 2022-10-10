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
// To make testing independent of existing plugins and devices, all plugins are
// forcefully unloaded and the mock plugin is registered as the only plugin.
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

#include "PiMockPlugin.hpp"
#include <detail/global_handler.hpp>
#include <detail/platform_impl.hpp>
#include <detail/plugin.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/device.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/platform.hpp>
#include <sycl/queue.hpp>

#include <functional>
#include <optional>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace unittest {

namespace detail = sycl::detail;
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
#include <sycl/detail/pi.def>
#undef _PI_API

/// The PiMock class manages the mock PI plugin and wraps an instance of a SYCL
/// platform class created from this plugin. Additionally it allows for the
/// redefinitions of functions in the PI API allowing tests to customize the
/// behavior of the underlying plugin to fit the need of the tests.
///
/// Mock platform instances must not share the plugin resources with
/// any other SYCL platform within the given context. Otherwise, mock
/// redefinitions would also affect other platforms' behavior.
/// Therefore, any plugin-related information is fully copied whenever
/// a user-passed SYCL object instance is being mocked.
/// The underlying SYCL platform must be a non-host platform to facilitate
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
// TODO: Consider reworking the class into a `detail::plugin` derivative.
class PiMock {
public:
  /// Constructs PiMock using the mock PI plugin.
  ///
  /// A new plugin will be stored into the platform instance, which
  /// will no longer share the plugin with other platform instances
  /// within the given context. A separate platform instance will be
  /// held by the PiMock instance.
  ///
  PiMock() {
    // Create new mock plugin platform and plugin handles
    // Note: Mock plugin will be generated if it has not been yet.
    MPlatformImpl = GetMockPlatformImpl();
    std::shared_ptr<detail::plugin> NewPluginPtr;
    {
      const detail::plugin &OriginalPiPlugin = MPlatformImpl->getPlugin();
      // Copy the PiPlugin, thus untying our to-be mock platform from other
      // platforms within the context. Reset our platform to use the new plugin.
      NewPluginPtr = std::make_shared<detail::plugin>(
          OriginalPiPlugin.getPiPluginPtr(), OriginalPiPlugin.getBackend(),
          OriginalPiPlugin.getLibraryHandle());
      // Save a copy of the platform resource
      OrigFuncTable = OriginalPiPlugin.getPiPlugin().PiFunctionTable;
    }
    MPlatformImpl->setPlugin(NewPluginPtr);
    // Extract the new PiPlugin instance by a non-const pointer,
    // explicitly allowing modification
    MPiPluginMockPtr = &NewPluginPtr->getPiPlugin();
  }

  PiMock(PiMock &&Other) {
    MPlatformImpl = std::move(Other.MPlatformImpl);
    OrigFuncTable = std::move(Other.OrigFuncTable);
    Other.OrigFuncTable = {}; // Move above doesn't reset the optional.
    MPiPluginMockPtr = std::move(Other.MPiPluginMockPtr);
  }
  PiMock(const PiMock &) = delete;
  PiMock &operator=(const PiMock &) = delete;
  ~PiMock() {
    if (!OrigFuncTable)
      return;

    MPiPluginMockPtr->PiFunctionTable = *OrigFuncTable;
  }

  /// Returns a handle to the SYCL platform instance.
  ///
  /// \return A reference to the SYCL platform.
  sycl::platform getPlatform() {
    return sycl::detail::createSyclObjFromImpl<sycl::platform>(MPlatformImpl);
  }

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

  /// Ensures that the mock plugin has been initialized and has been registered
  /// in the global handler. Additionally, all existing plugins will be removed
  /// and unloaded to avoid them being accidentally picked up by tests using
  /// selectors.
  static void EnsureMockPluginInitialized() {
    // Only initialize the plugin once.
    if (MMockPluginPtr)
      return;

    // Ensure that the other plugins are initialized so we can unload them.
    // This makes sure that the mock plugin is the only available plugin.
    detail::pi::initialize();
    detail::GlobalHandler::instance().unloadPlugins();
    std::vector<detail::plugin> &Plugins =
        detail::GlobalHandler::instance().getPlugins();

    assert(Plugins.empty() && "Clear failed to remove all plugins.");

    auto RTPlugin = std::make_shared<RT::PiPlugin>(
        RT::PiPlugin{"pi.ver.mock", "plugin.ver.mock", /*Targets=*/nullptr,
                     getMockedFunctionPointers()});

    // FIXME: which backend to pass here? does it affect anything?
    MMockPluginPtr = std::make_unique<detail::plugin>(RTPlugin, backend::opencl,
                                                      /*Library=*/nullptr);
    Plugins.push_back(*MMockPluginPtr);
  }

private:
  /// Ensures that the mock PI plugin has been registered and creates a
  /// platform_impl from it.
  ///
  /// \return a shared_ptr to a platform_impl created from the mock PI plugin.
  static std::shared_ptr<sycl::detail::platform_impl> GetMockPlatformImpl() {
    EnsureMockPluginInitialized();

    pi_uint32 NumPlatforms = 0;
    MMockPluginPtr->call_nocheck<detail::PiApiKind::piPlatformsGet>(
        0, nullptr, &NumPlatforms);
    assert(NumPlatforms > 0 && "No platforms returned by mock plugin.");
    pi_platform PiPlatform;
    MMockPluginPtr->call_nocheck<detail::PiApiKind::piPlatformsGet>(
        1, &PiPlatform, nullptr);
    return detail::platform_impl::getOrMakePlatformImpl(PiPlatform,
                                                        *MMockPluginPtr);
  }

  std::shared_ptr<sycl::detail::platform_impl> MPlatformImpl;
  std::optional<pi_plugin::FunctionPointers> OrigFuncTable;
  // Extracted at initialization for convenience purposes. The resource
  // itself is owned by the platform instance.
  RT::PiPlugin *MPiPluginMockPtr;

  // Pointer to the mock plugin pointer. This is static to avoid
  // reinitialization and re-registration of the same plugin.
  static inline std::unique_ptr<detail::plugin> MMockPluginPtr = nullptr;
};

} // namespace unittest
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
