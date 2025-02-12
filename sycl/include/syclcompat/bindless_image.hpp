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
 *  bindless_image.hpp
 *
 *  Description:
 *    bindless image functionality for the SYCL compatibility extension
 **************************************************************************/

// The original source was under the license below:
//==---- bindless_image.hpp -------------------------------*- C++
//-*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/ext/oneapi/bindless_images.hpp>

namespace syclcompat {
namespace experimental {

template <typename DataT, typename HintT = DataT, typename CoordT>
DataT sample_image(
    const sycl::ext::oneapi::experimental::sampled_image_handle &imageHandle,
    CoordT &&coords) {
  if constexpr (std::is_scalar_v<CoordT>) {
    return sycl::ext::oneapi::experimental::sample_image<DataT, HintT, CoordT>(
        imageHandle, coords / sizeof(DataT));
  } else {
    coords[0] = coords[0] / sizeof(DataT);
    return sycl::ext::oneapi::experimental::sample_image<DataT, HintT, CoordT>(
        imageHandle, coords);
  }
}
} // namespace experimental
} // namespace syclcompat
