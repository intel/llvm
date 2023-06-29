/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMA_BASE_H
#define UMA_BASE_H 1

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// \brief Generates generic 'UMA' API versions
#define UMA_MAKE_VERSION(_major, _minor)                                       \
    ((_major << 16) | (_minor & 0x0000ffff))

/// \brief Extracts 'UMA' API major version
#define UMA_MAJOR_VERSION(_ver) (_ver >> 16)

/// \brief Extracts 'UMA' API minor version
#define UMA_MINOR_VERSION(_ver) (_ver & 0x0000ffff)

/// \brief Current version of the UMA headers
#define UMA_VERSION_CURRENT UMA_MAKE_VERSION(0, 9)

/// \brief Operation results
enum uma_result_t {
    UMA_RESULT_SUCCESS = 0, ///< Success
    UMA_RESULT_ERROR_OUT_OF_HOST_MEMORY =
        1, ///< Insufficient host memory to satisfy call,
    UMA_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC =
        2, ///< A provider specific warning/error has been reported and can be
    ///< Retrieved via the umaMemoryProviderGetLastNativeError entry point.
    UMA_RESULT_ERROR_INVALID_ARGUMENT =
        3, ///< Generic error code for invalid arguments
    UMA_RESULT_ERROR_INVALID_ALIGNMENT = 4, /// Invalid alignment of an argument
    UMA_RESULT_ERROR_NOT_SUPPORTED = 5,     /// Operation not supported

    UMA_RESULT_ERROR_UNKNOWN = 0x7ffffffe ///< Unknown or internal error
};

#ifdef __cplusplus
}
#endif

#endif /* UMA_BASE_H */
