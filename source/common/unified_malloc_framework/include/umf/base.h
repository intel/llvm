/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_BASE_H
#define UMF_BASE_H 1

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// \brief Generates generic 'UMF' API versions
#define UMF_MAKE_VERSION(_major, _minor)                                       \
    ((_major << 16) | (_minor & 0x0000ffff))

/// \brief Extracts 'UMF' API major version
#define UMF_MAJOR_VERSION(_ver) (_ver >> 16)

/// \brief Extracts 'UMF' API minor version
#define UMF_MINOR_VERSION(_ver) (_ver & 0x0000ffff)

/// \brief Current version of the UMF headers
#define UMF_VERSION_CURRENT UMF_MAKE_VERSION(0, 9)

/// \brief Operation results
enum umf_result_t {
    UMF_RESULT_SUCCESS = 0, ///< Success
    UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY =
        1, ///< Insufficient host memory to satisfy call,
    UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC =
        2, ///< A provider specific warning/error has been reported and can be
    ///< Retrieved via the umfMemoryProviderGetLastNativeError entry point.
    UMF_RESULT_ERROR_INVALID_ARGUMENT =
        3, ///< Generic error code for invalid arguments
    UMF_RESULT_ERROR_INVALID_ALIGNMENT = 4, /// Invalid alignment of an argument
    UMF_RESULT_ERROR_NOT_SUPPORTED = 5,     /// Operation not supported

    UMF_RESULT_ERROR_UNKNOWN = 0x7ffffffe ///< Unknown or internal error
};

#ifdef __cplusplus
}
#endif

#endif /* UMF_BASE_H */
