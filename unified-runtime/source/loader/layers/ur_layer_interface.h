/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_layer_interface.h
 *
 */

#ifndef UR_LAYER_INTERFACE_H
#define UR_LAYER_INTERFACE_H 1

#include "unified-runtime/ur_api.h"
#include "unified-runtime/ur_ddi.h"

#if defined(__cplusplus)
extern "C" {
#endif

/// @brief The only symbol a layer shared library exports.
#define UR_LAYER_GET_INTERFACE_FUNC_NAME "urLoaderLayerGetInterface"

/// @brief Bump whenever ur_layer_interface_t changes. The loader only accepts a
///        library implementing the exact version it asks for.
#define UR_LAYER_INTERFACE_VERSION 1

/// @brief Entry points of a layer implemented in a shared library.
typedef struct ur_layer_interface_t {
  /// [out] Set to UR_LAYER_INTERFACE_VERSION by the layer.
  uint32_t version;

  /// @brief Initialize the layer and route pDdiTable through it. Called once
  ///        per loader instance; all of them share one layer state.
  ur_result_t(UR_APICALL *pfnInit)(ur_dditable_t *pDdiTable,
                                   const char *const *ppEnabledLayerNames,
                                   uint32_t numEnabledLayerNames);

  /// @brief Tear the layer down. The state is destroyed once every loader
  ///        instance that initialized the layer has torn it down.
  ur_result_t(UR_APICALL *pfnTearDown)(void);
} ur_layer_interface_t;

/// @brief Type of the UR_LAYER_GET_INTERFACE_FUNC_NAME entry point. Returns
///        UR_RESULT_ERROR_UNSUPPORTED_VERSION and leaves pInterface untouched
///        if the layer doesn't implement the requested version.
typedef ur_result_t(UR_APICALL *ur_pfnLoaderLayerGetInterface_t)(
    uint32_t version, ur_layer_interface_t *pInterface);

#if defined(__cplusplus)
} // extern "C"
#endif

#endif /* UR_LAYER_INTERFACE_H */
