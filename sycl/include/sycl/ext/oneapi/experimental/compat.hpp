/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL compatibility extension
 *
 *  compat.hpp
 *
 *  Description:
 *    Main include header for the SYCL compatibility extension
 **************************************************************************/

#pragma once

#define SYCL_EXT_ONEAPI_COMPAT 1

#include <sycl/ext/oneapi/experimental/compat/atomic.hpp>
#include <sycl/ext/oneapi/experimental/compat/device.hpp>
#include <sycl/ext/oneapi/experimental/compat/dims.hpp>
#include <sycl/ext/oneapi/experimental/compat/defs.hpp>
#include <sycl/ext/oneapi/experimental/compat/kernel.hpp>
#include <sycl/ext/oneapi/experimental/compat/kernel_function.hpp>
#include <sycl/ext/oneapi/experimental/compat/launch.hpp>
#include <sycl/ext/oneapi/experimental/compat/memory.hpp>
#include <sycl/ext/oneapi/experimental/compat/util.hpp>
