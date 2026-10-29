/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_sanitizer_layer_entry.cpp
 *
 */

#include "logger/ur_logger.hpp"
#include "ur_layer_interface.h"
#include "ur_sanitizer_layer.hpp"

#include <mutex>
#include <set>
#include <string>
#include <type_traits>

namespace {

std::mutex layerMutex;
// Loader instances that have the layer initialized.
uint32_t initCount = 0;

const char *getSanitizerName(ur_sanitizer_layer::SanitizerType type) {
  switch (type) {
  case ur_sanitizer_layer::SanitizerType::AddressSanitizer:
    return "ASAN";
  case ur_sanitizer_layer::SanitizerType::MemorySanitizer:
    return "MSAN";
  case ur_sanitizer_layer::SanitizerType::ThreadSanitizer:
    return "TSAN";
  default:
    return nullptr;
  }
}

ur_result_t UR_APICALL layerInit(ur_dditable_t *pDdiTable,
                                 const char *const *ppEnabledLayerNames,
                                 uint32_t numEnabledLayerNames) {
  if (!pDdiTable || (numEnabledLayerNames && !ppEnabledLayerNames)) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  std::set<std::string> enabledLayerNames;
  for (uint32_t i = 0; i < numEnabledLayerNames; i++) {
    enabledLayerNames.insert(ppEnabledLayerNames[i]);
  }

  std::lock_guard<std::mutex> lock(layerMutex);

  auto *context = ur_sanitizer_layer::getContext();

  ur_result_t result;
  if (initCount == 0) {
    result = context->init(pDdiTable, enabledLayerNames, codeloc_data{});
    // Announced once per process, unlike the per-loader ddi table setup.
    const char *name = getSanitizerName(context->enabledType);
    if (result == UR_RESULT_SUCCESS && name) {
      UR_LOG_L(context->logger, QUIET, "==== DeviceSanitizer: {}", name);
    }
  } else {
    // The state is already up, so only hook up this loader's ddi table.
    UR_LOG(DEBUG, "sanitizer layer is already initialized, intercepting an "
                  "additional ddi table");
    result = context->interceptDdiTable(pDdiTable);
  }

  if (result == UR_RESULT_SUCCESS) {
    initCount++;
  }

  return result;
}

ur_result_t UR_APICALL layerTearDown() {
  std::lock_guard<std::mutex> lock(layerMutex);

  if (initCount == 0 || --initCount != 0) {
    return UR_RESULT_SUCCESS;
  }

  ur_result_t result = ur_sanitizer_layer::getContext()->tearDown();
  ur_sanitizer_layer::context_t::forceDelete();

  return result;
}

} // namespace

extern "C" UR_APIEXPORT ur_result_t UR_APICALL
urLoaderLayerGetInterface(uint32_t version, ur_layer_interface_t *pInterface) {
  // This library has its own copy of the logger; keep it on UR_LOG_LOADER.
  logger::init("loader");

  if (version != UR_LAYER_INTERFACE_VERSION) {
    UR_LOG(ERR, "unsupported layer interface version {}, expected {}", version,
           UR_LAYER_INTERFACE_VERSION);
    return UR_RESULT_ERROR_UNSUPPORTED_VERSION;
  }

  if (!pInterface) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  pInterface->version = UR_LAYER_INTERFACE_VERSION;
  pInterface->pfnInit = layerInit;
  pInterface->pfnTearDown = layerTearDown;

  return UR_RESULT_SUCCESS;
}

static_assert(std::is_same_v<decltype(&urLoaderLayerGetInterface),
                             ur_pfnLoaderLayerGetInterface_t>,
              "the loader casts the entry point to this type");
