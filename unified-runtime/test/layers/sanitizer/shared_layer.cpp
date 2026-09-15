/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file shared_layer.cpp
 *
 */

// RUN: UR_LOG_LOADER="level:debug;flush:debug;output:stdout" shared_layer-test
// REQUIRES: sanitizer

#include "ur_layer_interface.h"
#include "ur_util.hpp"

#include <dlfcn.h>
#include <gtest/gtest.h>

namespace {

constexpr const char *LibraryName =
    MAKE_LIBRARY_NAME("ur_sanitizer_layer", "0");

struct SharedSanitizerLayer : public ::testing::Test {
  void SetUp() override {
    handle = dlopen(LibraryName, RTLD_LAZY | RTLD_LOCAL);
    ASSERT_NE(handle, nullptr) << dlerror();
  }

  void TearDown() override {
    if (handle) {
      dlclose(handle);
    }
  }

  ur_pfnLoaderLayerGetInterface_t getInterfaceFn() {
    return reinterpret_cast<ur_pfnLoaderLayerGetInterface_t>(
        dlsym(handle, UR_LAYER_GET_INTERFACE_FUNC_NAME));
  }

  void *handle = nullptr;
};

TEST_F(SharedSanitizerLayer, GetInterface) {
  auto pfnGetInterface = getInterfaceFn();
  ASSERT_NE(pfnGetInterface, nullptr) << dlerror();

  ur_layer_interface_t interface = {};
  ASSERT_EQ(pfnGetInterface(UR_LAYER_INTERFACE_VERSION, &interface),
            UR_RESULT_SUCCESS);
  ASSERT_EQ(interface.version, UR_LAYER_INTERFACE_VERSION);
  ASSERT_NE(interface.pfnInit, nullptr);
  ASSERT_NE(interface.pfnTearDown, nullptr);
}

TEST_F(SharedSanitizerLayer, GetInterfaceRejectsUnknownVersion) {
  auto pfnGetInterface = getInterfaceFn();
  ASSERT_NE(pfnGetInterface, nullptr) << dlerror();

  ur_layer_interface_t interface = {};
  ASSERT_EQ(pfnGetInterface(UR_LAYER_INTERFACE_VERSION + 1, &interface),
            UR_RESULT_ERROR_UNSUPPORTED_VERSION);
  ASSERT_EQ(pfnGetInterface(UR_LAYER_INTERFACE_VERSION, nullptr),
            UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

// Consumers have their own copy of the layer's dependencies, so nothing but the
// entry point may be visible.
TEST_F(SharedSanitizerLayer, EntryPointIsTheOnlyExportedSymbol) {
  ASSERT_NE(dlsym(handle, UR_LAYER_GET_INTERFACE_FUNC_NAME), nullptr);

  for (const char *name : {"urLoaderInit", "urContextCreate",
                           "_ZN20ur_sanitizer_layer10getContextEv"}) {
    EXPECT_EQ(dlsym(handle, name), nullptr) << name << " must not be exported";
  }
}

// The shared state has to outlive all but the last loader instance.
TEST_F(SharedSanitizerLayer, InitIsReferenceCounted) {
  auto pfnGetInterface = getInterfaceFn();
  ASSERT_NE(pfnGetInterface, nullptr) << dlerror();

  ur_layer_interface_t interface = {};
  ASSERT_EQ(pfnGetInterface(UR_LAYER_INTERFACE_VERSION, &interface),
            UR_RESULT_SUCCESS);

  // No sanitizer is enabled here, so this only exercises the book-keeping.
  ur_dditable_t firstTable = {};
  ur_dditable_t secondTable = {};
  ASSERT_EQ(interface.pfnInit(&firstTable, nullptr, 0), UR_RESULT_SUCCESS);
  ASSERT_EQ(interface.pfnInit(&secondTable, nullptr, 0), UR_RESULT_SUCCESS);

  ASSERT_EQ(interface.pfnTearDown(), UR_RESULT_SUCCESS);
  ASSERT_EQ(interface.pfnTearDown(), UR_RESULT_SUCCESS);
  // Tearing down more often than initializing must not underflow the count.
  ASSERT_EQ(interface.pfnTearDown(), UR_RESULT_SUCCESS);

  ASSERT_EQ(interface.pfnInit(nullptr, nullptr, 0),
            UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

} // namespace
