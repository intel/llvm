# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#
# FindLIBBACKTRACE.cmake -- module searching for libbacktrace library.
#                           LIBBACKTRACE_FOUND is set to true if libbacktrace is found.
#

find_library(LIBBACKTRACE_LIBRARIES NAMES backtrace)
find_path(LIBBACKTRACE_INCLUDE_DIR NAMES backtrace.h)

if (LIBBACKTRACE_LIBRARIES AND LIBBACKTRACE_INCLUDE_DIR)
    set(LIBBACKTRACE_FOUND TRUE)
endif()

if (LIBBACKTRACE_FOUND)
    add_library(Libbacktrace INTERFACE IMPORTED)
    set_target_properties(Libbacktrace PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LIBBACKTRACE_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${LIBBACKTRACE_LIBRARIES}"
    )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Libbacktrace DEFAULT_MSG LIBBACKTRACE_LIBRARIES LIBBACKTRACE_INCLUDE_DIR)
