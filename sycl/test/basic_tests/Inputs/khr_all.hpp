//==---- khr_all.hpp - Test umbrella for all SYCL KHR headers ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test-only umbrella that pulls in every public KHR header (top-level
// <sycl/khr/*.hpp> plus the split-header set under
// <sycl/khr/split_headers/*.hpp>). Used by device-safety regression tests
// that must verify the entire KHR surface stays free of host-only standard
// library machinery (iostream, fstream, filesystem, ...).
//
// NOT intended for production code: keep production includes targeted.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/khr/dynamic_addrspace_cast.hpp>
#include <sycl/khr/free_function_commands.hpp>
#include <sycl/khr/group_interface.hpp>
#include <sycl/khr/split_headers/accessor.hpp>
#include <sycl/khr/split_headers/atomic.hpp>
#include <sycl/khr/split_headers/backend.hpp>
#include <sycl/khr/split_headers/bit.hpp>
#include <sycl/khr/split_headers/buffer.hpp>
#include <sycl/khr/split_headers/builtins_common.hpp>
#include <sycl/khr/split_headers/builtins_geometric.hpp>
#include <sycl/khr/split_headers/builtins_integer.hpp>
#include <sycl/khr/split_headers/builtins_math.hpp>
#include <sycl/khr/split_headers/builtins_relational.hpp>
#include <sycl/khr/split_headers/byte.hpp>
#include <sycl/khr/split_headers/context.hpp>
#include <sycl/khr/split_headers/device.hpp>
#include <sycl/khr/split_headers/event.hpp>
#include <sycl/khr/split_headers/exception.hpp>
#include <sycl/khr/split_headers/functional.hpp>
#include <sycl/khr/split_headers/group_algorithms.hpp>
#include <sycl/khr/split_headers/groups.hpp>
#include <sycl/khr/split_headers/half.hpp>
#include <sycl/khr/split_headers/handler.hpp>
#include <sycl/khr/split_headers/hierarchical_parallelism.hpp>
#include <sycl/khr/split_headers/images.hpp>
#include <sycl/khr/split_headers/index_space.hpp>
#include <sycl/khr/split_headers/interop_handle.hpp>
#include <sycl/khr/split_headers/kernel_bundle.hpp>
#include <sycl/khr/split_headers/kernel_handler.hpp>
#include <sycl/khr/split_headers/marray.hpp>
#include <sycl/khr/split_headers/math.hpp>
#include <sycl/khr/split_headers/multi_ptr.hpp>
#include <sycl/khr/split_headers/platform.hpp>
#include <sycl/khr/split_headers/property_list.hpp>
#include <sycl/khr/split_headers/queue.hpp>
#include <sycl/khr/split_headers/reduction.hpp>
#include <sycl/khr/split_headers/span.hpp>
#include <sycl/khr/split_headers/stream.hpp>
#include <sycl/khr/split_headers/type_traits.hpp>
#include <sycl/khr/split_headers/usm.hpp>
#include <sycl/khr/split_headers/vec.hpp>
#include <sycl/khr/split_headers/version.hpp>
#include <sycl/khr/static_addrspace_cast.hpp>
#include <sycl/khr/work_item_queries.hpp>
