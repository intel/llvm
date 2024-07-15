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
#include <detail/scheduler/scheduler.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/device.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/platform.hpp>
#include <sycl/queue.hpp>

#include <functional>
#include <optional>

namespace sycl {
inline namespace _V1 {
namespace unittest {

namespace detail = sycl::detail;

/// The macro below defines a proxy functions for each PI API call.
/// This proxy function calls all the functions registered in CallBefore*
/// function pointer array, then calls Original function, then calls functions
/// registered in CallAfter* array.
///
/// If a function from CallBefore* returns a non-PI_SUCCESS return code the
/// proxy function bails out.

/// Number of functions that can be registered as CallBefore and CallAfter
inline constexpr size_t CallStackSize = 16;
#define _PI_API(api)                                                           \
                                                                               \
  inline decltype(&::api) CallBefore_##api[CallStackSize] = {nullptr};         \
  inline decltype(&::api) CallOriginal_##api = mock_##api;                     \
  inline decltype(&::api) CallAfter_##api[CallStackSize] = {nullptr};          \
                                                                               \
  template <class RetT, class... ArgsT> RetT proxy_mock_##api(ArgsT... Args) { \
    for (size_t I = 0; I < CallStackSize && CallBefore_##api[I]; ++I) {        \
      /* If before function returns an error bail out */                       \
      const RetT Res = CallBefore_##api[I](Args...);                           \
      if (Res != PI_SUCCESS)                                                   \
        return Res;                                                            \
    }                                                                          \
                                                                               \
    RetT Ret = CallOriginal_##api(Args...);                                    \
                                                                               \
    for (size_t I = 0; I < CallStackSize && CallAfter_##api[I]; ++I)           \
      CallAfter_##api[I](Args...);                                             \
                                                                               \
    return Ret;                                                                \
  }                                                                            \
                                                                               \
  /* A helper function for instantiating proxy functions for a given */        \
  /* PI API signature */                                                       \
  template <class RetT_, class... ArgsT_>                                      \
  int ConverterT_##api(RetT_ (*FuncArg)(ArgsT_...)) {                          \
    [[maybe_unused]] constexpr static RetT_ (*Func)(ArgsT_...) =               \
        proxy_mock_##api<RetT_, ArgsT_...>;                                    \
    return 42;                                                                 \
  }                                                                            \
  inline int Anchor_##api = ConverterT_##api(decltype (&::api)(0x0));          \
                                                                               \
  /*Overrides a plugin PI function with a given one */                         \
  template <detail::PiApiKind PiApiOffset>                                     \
  inline void setFuncPtr(sycl::detail::pi::PiPlugin *MPlugin,                  \
                         decltype(&::api) FuncPtr);                            \
  template <>                                                                  \
  inline void setFuncPtr<detail::PiApiKind::api>(                              \
      sycl::detail::pi::PiPlugin * MPlugin, decltype(&::api) FuncPtr) {        \
    CallOriginal_##api = FuncPtr;                                              \
  }                                                                            \
                                                                               \
  /*Adds a function to be called before the PI function*/                      \
  template <detail::PiApiKind PiApiOffset>                                     \
  inline void setFuncPtrBefore(sycl::detail::pi::PiPlugin *MPlugin,            \
                               decltype(&::api) FuncPtr);                      \
  template <>                                                                  \
  inline void setFuncPtrBefore<detail::PiApiKind::api>(                        \
      sycl::detail::pi::PiPlugin * MPlugin, decltype(&::api) FuncPtr) {        \
    /* Find free slot */                                                       \
    size_t I = 0;                                                              \
    for (; I < CallStackSize && CallBefore_##api[I]; ++I)                      \
      ;                                                                        \
    assert(I < CallStackSize && "Too many calls before");                      \
    CallBefore_##api[I] = FuncPtr;                                             \
  }                                                                            \
                                                                               \
  /*Adds a function to be called after the PI function*/                       \
  template <detail::PiApiKind PiApiOffset>                                     \
  inline void setFuncPtrAfter(sycl::detail::pi::PiPlugin *MPlugin,             \
                              decltype(&::api) FuncPtr);                       \
  template <>                                                                  \
  inline void setFuncPtrAfter<detail::PiApiKind::api>(                         \
      sycl::detail::pi::PiPlugin * MPlugin, decltype(&::api) FuncPtr) {        \
    /* Find free slot */                                                       \
    size_t I = 0;                                                              \
    for (; I < CallStackSize && CallAfter_##api[I]; ++I)                       \
      ;                                                                        \
    assert(I < CallStackSize && "Too many calls after");                       \
    CallAfter_##api[I] = FuncPtr;                                              \
  }
#include <sycl/detail/pi.def>
#undef _PI_API

// Unregister functions set for calling before and after PI API
inline void clearRedefinedCalls() {
  for (size_t I = 0; I < CallStackSize; ++I) {
#define _PI_API(api)                                                           \
  CallBefore_##api[I] = nullptr;                                               \
  CallOriginal_##api = mock_##api;                                             \
  CallAfter_##api[I] = nullptr;
#include <sycl/detail/pi.def>
#undef _PI_API
  }
}

#define _PI_MOCK_PLUGIN_CONCAT(A, B) A##B
#define PI_MOCK_PLUGIN_CONCAT(A, B) _PI_MOCK_PLUGIN_CONCAT(A, B)

inline pi_plugin::FunctionPointers getProxyMockedFunctionPointers() {
  return {
#define _PI_API(api) PI_MOCK_PLUGIN_CONCAT(proxy_mock_, api),
#include <sycl/detail/pi.def>
#undef _PI_API
  };
}

#undef PI_MOCK_PLUGIN_CONCAT
#undef _PI_MOCK_PLUGIN_CONCAT

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
/// Mock.redefineBefore<PiApiKind::piProgramRetain>(redefinePiProgramRetain);
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
  /// \param Backend is the backend type to mock, intended for testing backend
  /// specific runtime logic.
  PiMock(backend Backend = backend::opencl) {
    // Create new mock plugin platform and plugin handles
    // Note: Mock plugin will be generated if it has not been yet.
    MPlatformImpl = GetMockPlatformImpl(Backend);
    detail::PluginPtr NewPluginPtr;
    {
      const detail::PluginPtr &OriginalPlugin = MPlatformImpl->getPlugin();
      // Copy the PiPlugin, thus untying our to-be mock platform from other
      // platforms within the context. Reset our platform to use the new plugin.
      NewPluginPtr = std::make_shared<detail::plugin>(
          OriginalPlugin->getPiPluginPtr(), Backend,
          OriginalPlugin->getLibraryHandle());
      // Save a copy of the platform resource
      OrigFuncTable = OriginalPlugin->getPiPlugin().PiFunctionTable;
    }
    MPlatformImpl->setPlugin(NewPluginPtr, Backend);
    // Extract the new PiPlugin instance by a non-const pointer,
    // explicitly allowing modification
    MPiPluginMockPtr = &NewPluginPtr->getPiPlugin();
  }

  PiMock(PiMock &&Other) {
    MPlatformImpl = std::move(Other.MPlatformImpl);
    OrigFuncTable = std::move(Other.OrigFuncTable);
    Other.OrigFuncTable = {}; // Move above doesn't reset the optional.
    MPiPluginMockPtr = std::move(Other.MPiPluginMockPtr);
    Other.MIsMoved = true;
  }
  PiMock(const PiMock &) = delete;
  PiMock &operator=(const PiMock &) = delete;
  ~PiMock() {
    // Do nothing if mock was moved.
    if (MIsMoved)
      return;

    // Since the plugin relies on the global vars to store function pointers we
    // need to reset them for the new PiMock plugin instance
    // TODO: Make function pointers array for each PiMock instance?
    clearRedefinedCalls();
    if (!OrigFuncTable)
      return;

    MPiPluginMockPtr->PiFunctionTable = *OrigFuncTable;
    // calling drainThreadPool and releaseResources explicitly due to win
    // related WA in shutdown process
    detail::GlobalHandler::instance().drainThreadPool();
    detail::GlobalHandler::instance().getScheduler().releaseResources(
        detail::BlockingT::BLOCKING);
    detail::GlobalHandler::instance().releaseDefaultContexts();
  }

  /// Returns a handle to the SYCL platform instance.
  ///
  /// \return A reference to the SYCL platform.
  sycl::platform getPlatform() {
    return sycl::detail::createSyclObjFromImpl<sycl::platform>(MPlatformImpl);
  }

  template <detail::PiApiKind PiApiOffset>
  using FuncPtrT = typename sycl::detail::pi::PiFuncInfo<PiApiOffset>::FuncPtrT;
  template <detail::PiApiKind PiApiOffset>
  using SignatureT = typename std::remove_pointer<FuncPtrT<PiApiOffset>>::type;

  /// Adds a function to be called before a given PI API
  ///
  /// \param Replacement is a mock std::function instance to be
  ///        called instead of the given PI API. This function must
  ///        not have been constructed from a lambda.
  template <detail::PiApiKind PiApiOffset>
  void
  redefineBefore(const std::function<SignatureT<PiApiOffset>> &Replacement) {
    FuncPtrT<PiApiOffset> FuncPtr =
        *Replacement.template target<FuncPtrT<PiApiOffset>>();
    assert(FuncPtr &&
           "Function target is empty, try passing a lambda directly");
    setFuncPtrBefore<PiApiOffset>(MPiPluginMockPtr, *FuncPtr);
  }

  /// redefineBefore overload for function pointer/captureless lambda arguments.
  ///
  /// \param Replacement is a mock callable assignable to a function
  ///        pointer (function pointer/captureless lambda).

  template <detail::PiApiKind PiApiOffset, typename FunctorT>
  void redefineBefore(const FunctorT &Replacement) {
    // TODO: Check for matching signatures/assignability
    setFuncPtrBefore<PiApiOffset>(MPiPluginMockPtr, Replacement);
  }
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

  /// Adds a function to be called after a given PI API
  ///
  /// \param Replacement is a mock std::function instance to be
  ///        called instead of the given PI API. This function must
  ///        not have been constructed from a lambda.
  template <detail::PiApiKind PiApiOffset>
  void
  redefineAfter(const std::function<SignatureT<PiApiOffset>> &Replacement) {
    FuncPtrT<PiApiOffset> FuncPtr =
        *Replacement.template target<FuncPtrT<PiApiOffset>>();
    assert(FuncPtr &&
           "Function target is empty, try passing a lambda directly");
    setFuncPtrAfter<PiApiOffset>(MPiPluginMockPtr, *FuncPtr);
  }

  /// redefineAfter overload for function pointer/captureless lambda arguments.
  ///
  /// \param Replacement is a mock callable assignable to a function
  ///        pointer (function pointer/captureless lambda).
  template <detail::PiApiKind PiApiOffset, typename FunctorT>
  void redefineAfter(const FunctorT &Replacement) {
    // TODO: Check for matching signatures/assignability
    setFuncPtrAfter<PiApiOffset>(MPiPluginMockPtr, Replacement);
  }

  /// Ensures that the mock plugin has been initialized and has been registered
  /// in the global handler. Additionally, all existing plugins will be removed
  /// and unloaded to avoid them being accidentally picked up by tests using
  /// selectors.
  /// \param Backend is the backend type to mock, intended for testing backend
  /// specific runtime logic.
  static void EnsureMockPluginInitialized(backend Backend = backend::opencl) {
    // Only initialize the plugin once.
    if (MMockPluginPtr)
      return;

    // Ensure that the other plugins are initialized so we can unload them.
    // This makes sure that the mock plugin is the only available plugin.
    detail::pi::initialize();
    detail::GlobalHandler::instance().unloadPlugins();
    std::vector<detail::PluginPtr> &Plugins =
        detail::GlobalHandler::instance().getPlugins();

    assert(Plugins.empty() && "Clear failed to remove all plugins.");

    auto RTPlugin =
        std::make_shared<sycl::detail::pi::PiPlugin>(sycl::detail::pi::PiPlugin{
            "pi.ver.mock", "plugin.ver.mock", /*Targets=*/nullptr,
            getProxyMockedFunctionPointers(), _PI_SANITIZE_TYPE_NONE});

    MMockPluginPtr = std::make_shared<detail::plugin>(RTPlugin, Backend,
                                                      /*Library=*/nullptr);
    Plugins.push_back(MMockPluginPtr);
  }

private:
  /// Ensures that the mock PI plugin has been registered and creates a
  /// platform_impl from it.
  ///
  /// \return a shared_ptr to a platform_impl created from the mock PI plugin.
  static std::shared_ptr<sycl::detail::platform_impl>
  GetMockPlatformImpl(backend Backend) {
    EnsureMockPluginInitialized(Backend);

    pi_uint32 NumPlatforms = 0;
    MMockPluginPtr->call_nocheck<detail::PiApiKind::piPlatformsGet>(
        0, nullptr, &NumPlatforms);
    assert(NumPlatforms > 0 && "No platforms returned by mock plugin.");
    pi_platform PiPlatform;
    MMockPluginPtr->call_nocheck<detail::PiApiKind::piPlatformsGet>(
        1, &PiPlatform, nullptr);
    return detail::platform_impl::getOrMakePlatformImpl(PiPlatform,
                                                        MMockPluginPtr);
  }

  std::shared_ptr<sycl::detail::platform_impl> MPlatformImpl;
  std::optional<pi_plugin::FunctionPointers> OrigFuncTable;
  // Extracted at initialization for convenience purposes. The resource
  // itself is owned by the platform instance.
  sycl::detail::pi::PiPlugin *MPiPluginMockPtr;

  // Marker to indicate if the mock was moved.
  bool MIsMoved = false;

  // Pointer to the mock plugin pointer. This is static to avoid
  // reinitialization and re-registration of the same plugin.
  static inline detail::PluginPtr MMockPluginPtr = nullptr;
};

} // namespace unittest
} // namespace _V1
} // namespace sycl
