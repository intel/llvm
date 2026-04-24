//===----------- usm_p2p.hpp - L0 Adapter ---------------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <ur_api.h>

namespace ur::level_zero::common {

ur_result_t urUsmP2PEnablePeerAccessExp(ur_device_handle_t commandDevice,
                                        ur_device_handle_t peerDevice);
ur_result_t urUsmP2PDisablePeerAccessExp(ur_device_handle_t commandDevice,
                                         ur_device_handle_t peerDevice);
ur_result_t urUsmP2PPeerAccessGetInfoExp(ur_device_handle_t commandDevice,
                                         ur_device_handle_t peerDevice,
                                         ur_exp_peer_info_t propName,
                                         size_t propSize, void *pPropValue,
                                         size_t *pPropSizeRet);

} // namespace ur::level_zero::common
