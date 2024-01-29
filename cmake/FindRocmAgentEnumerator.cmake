# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#
# FindRocmAgentEnumerator.cmake -- module searching for rocm_agent_enumerator.
#                                  ROCM_AGENT_ENUMERATOR_FOUND is set to true if
#                                  rocm_agent_enumerator is found.
#

find_program(ROCM_AGENT_ENUMERATOR NAMES rocm_agent_enumerator)

if(ROCM_AGENT_ENUMERATOR)
    set(ROCM_AGENT_ENUMERATOR_FOUND TRUE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RocmAgentEnumerator DEFAULT_MSG ROCM_AGENT_ENUMERATOR)
