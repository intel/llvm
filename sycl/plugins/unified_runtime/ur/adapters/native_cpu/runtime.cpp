//===---------------- runtime.cpp - Native CPU Adapter --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur_api.h"

UR_APIEXPORT ur_result_t UR_APICALL urInit(ur_device_init_flags_t) {
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urTearDown(void *) {
  return UR_RESULT_SUCCESS;
}
