<%!
from templates import helper as th
%>/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions.
 * See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ${name}.hpp
 *
 */

 // Auto-generated file, do not edit.

#pragma once

#include <string>
#include <vector>

#include "ur_util.hpp"
#include <ur_api.h>

namespace ur_loader {
struct ur_adapter_manifest {
  std::string name;
  std::string library;
  ur_backend_t backend;
  std::vector<ur_device_type_t> device_types;
};

const std::vector<ur_adapter_manifest> ur_adapter_manifests = {
%for manifest in th.get_adapter_manifests(specs):
{
  "${manifest['name']}",
  MAKE_LIBRARY_NAME("ur_adapter_${manifest['name']}", "0"),
  ${th.subt(namespace, tags, manifest['backend'])},
  {
  %for device_type in manifest['device_types']:
    ${th.subt(namespace, tags, device_type)},
  %endfor
  }
},
%endfor
};
}
