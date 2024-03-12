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
 *  util.hpp
 *
 *  Description:
 *    util functionality for the SYCL compatibility extension
 **************************************************************************/

// The original source was under the license below:
//==---- util.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <syclcompat/math.hpp>
#include <syclcompat/memory.hpp>

namespace syclcompat {

namespace detail {
template <typename T> struct DataType {
  using T2 = T;
};
template <typename T> struct DataType<sycl::vec<T, 2>> {
  using T2 = detail::complex_type<T>;
};

inline void matrix_mem_copy(void *to_ptr, const void *from_ptr, int to_ld,
                            int from_ld, int rows, int cols, int elem_size,
                            sycl::queue queue = syclcompat::get_default_queue(),
                            bool async = false) {
  if (to_ptr == from_ptr && to_ld == from_ld) {
    return;
  }

  if (to_ld == from_ld) {
    size_t copy_size = elem_size * ((cols - 1) * (size_t)to_ld + rows);
    if (async)
      detail::memcpy(queue, (void *)to_ptr, (void *)from_ptr, copy_size);
    else
      detail::memcpy(queue, (void *)to_ptr, (void *)from_ptr, copy_size).wait();
  } else {
    if (async)
      detail::memcpy(queue, to_ptr, from_ptr, elem_size * to_ld,
                     elem_size * from_ld, elem_size * rows, cols);
    else
      sycl::event::wait(detail::memcpy(queue, to_ptr, from_ptr,
                                       elem_size * to_ld, elem_size * from_ld,
                                       elem_size * rows, cols));
  }
}

/// Copy matrix data. The default leading dimension is column.
/// \param [out] to_ptr A pointer points to the destination location.
/// \param [in] from_ptr A pointer points to the source location.
/// \param [in] to_ld The leading dimension the destination matrix.
/// \param [in] from_ld The leading dimension the source matrix.
/// \param [in] rows The number of rows of the source matrix.
/// \param [in] cols The number of columns of the source matrix.
/// \param [in] queue The queue where the routine should be executed.
/// \param [in] async If this argument is true, the return of the function
/// does NOT guarantee the copy is completed.
template <typename T>
inline void matrix_mem_copy(T *to_ptr, const T *from_ptr, int to_ld,
                            int from_ld, int rows, int cols,
                            sycl::queue queue = get_default_queue(),
                            bool async = false) {
  using Ty = typename DataType<T>::T2;
  matrix_mem_copy((void *)to_ptr, (void *)from_ptr, to_ld, from_ld, rows, cols,
                  sizeof(Ty), queue, async);
}
} // namespace detail

/// Cast the high or low 32 bits of a double to an integer.
/// \param [in] d The double value.
/// \param [in] use_high32 Cast the high 32 bits of the double if true;
/// otherwise cast the low 32 bits.
inline int cast_double_to_int(double d, bool use_high32 = true) {
  sycl::vec<double, 1> v0{d};
  auto v1 = v0.as<sycl::int2>();
  if (use_high32)
    return v1[0];
  return v1[1];
}

/// Combine two integers, the first as the high 32 bits and the second
/// as the low 32 bits, into a double.
/// \param [in] high32 The integer as the high 32 bits
/// \param [in] low32 The integer as the low 32 bits
inline double cast_ints_to_double(int high32, int low32) {
  sycl::int2 v0{high32, low32};
  auto v1 = v0.as<sycl::vec<double, 1>>();
  return v1;
}

/// Reverse the bit order of an unsigned integer
/// \param [in] a Input unsigned integer value
/// \returns Value of a with the bit order reversed
template <typename T> inline T reverse_bits(T a) {
  static_assert(std::is_unsigned<T>::value && std::is_integral<T>::value,
                "unsigned integer required");
  if (!a)
    return 0;
  T mask = 0;
  size_t count = 4 * sizeof(T);
  mask = ~mask >> count;
  while (count) {
    a = ((a & mask) << count) | ((a & ~mask) >> count);
    count = count >> 1;
    mask = mask ^ (mask << count);
  }
  return a;
}

/// \param [in] a The first value contains 4 bytes
/// \param [in] b The second value contains 4 bytes
/// \param [in] s The selector value, only lower 16bit used
/// \returns the permutation result of 4 bytes selected in the way
/// specified by \p s from \p a and \p b
inline unsigned int byte_level_permute(unsigned int a, unsigned int b,
                                       unsigned int s) {
  unsigned int ret;
  ret =
      ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff) << 8) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff) << 16) |
      (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff) << 24);
  return ret;
}

/// Find position of first least significant set bit in an integer.
/// ffs(0) returns 0.
///
/// \param [in] a Input integer value
/// \returns The position
template <typename T> inline int ffs(T a) {
  static_assert(std::is_integral<T>::value, "integer required");
  return (sycl::ctz(a) + 1) % (sizeof(T) * 8 + 1);
}

/// select_from_sub_group allows work-items to obtain a copy of a value held by
/// any other work-item in the sub_group. The input sub_group will be divided
/// into several logical sub_groups with id range [0, \p logical_sub_group_size
/// - 1]. Each work-item in logical sub_group gets value from another work-item
/// whose id is \p remote_local_id. If \p remote_local_id is outside the
/// logical sub_group id range, \p remote_local_id will modulo with \p
/// logical_sub_group_size. The \p logical_sub_group_size must be a power of 2
/// and not exceed input sub_group size.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] remote_local_id Input source work item id
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T select_from_sub_group(sycl::sub_group g, T x, int remote_local_id,
                        int logical_sub_group_size = 32) {
  unsigned int start_index =
      g.get_local_linear_id() / logical_sub_group_size * logical_sub_group_size;
  return sycl::select_from_group(
      g, x, start_index + remote_local_id % logical_sub_group_size);
}

/// shift_sub_group_left move values held by the work-items in a sub_group
/// directly to another work-item in the sub_group, by shifting values a fixed
/// number of work-items to the left. The input sub_group will be divided into
/// several logical sub_groups with id range [0, \p logical_sub_group_size - 1].
/// Each work-item in logical sub_group gets value from another work-item whose
/// id is caller's id adds \p delta. If calculated id is outside the logical
/// sub_group id range, the work-item will get value from itself. The \p
/// logical_sub_group_size must be a power of 2 and not exceed input sub_group
/// size.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] delta Input delta
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T shift_sub_group_left(sycl::sub_group g, T x, unsigned int delta,
                       int logical_sub_group_size = 32) {
  unsigned int id = g.get_local_linear_id();
  unsigned int end_index =
      (id / logical_sub_group_size + 1) * logical_sub_group_size;
  T result = sycl::shift_group_left(g, x, delta);
  if ((id + delta) >= end_index) {
    result = x;
  }
  return result;
}

/// shift_sub_group_right move values held by the work-items in a sub_group
/// directly to another work-item in the sub_group, by shifting values a fixed
/// number of work-items to the right. The input sub_group will be divided into
/// several logical_sub_groups with id range [0, \p logical_sub_group_size - 1].
/// Each work-item in logical_sub_group gets value from another work-item whose
/// id is caller's id subtracts \p delta. If calculated id is outside the
/// logical sub_group id range, the work-item will get value from itself. The \p
/// logical_sub_group_size must be a power of 2 and not exceed input sub_group
/// size.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] delta Input delta
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T shift_sub_group_right(sycl::sub_group g, T x, unsigned int delta,
                        int logical_sub_group_size = 32) {
  unsigned int id = g.get_local_linear_id();
  unsigned int start_index =
      id / logical_sub_group_size * logical_sub_group_size;
  T result = sycl::shift_group_right(g, x, delta);
  if ((id - start_index) < delta) {
    result = x;
  }
  return result;
}

/// permute_sub_group_by_xor permutes values by exchanging values held by pairs
/// of work-items identified by computing the bitwise exclusive OR of the
/// work-item id and some fixed mask. The input sub_group will be divided into
/// several logical sub_groups with id range [0, \p logical_sub_group_size - 1].
/// Each work-item in logical sub_group gets value from another work-item whose
/// id is bitwise exclusive OR of the caller's id and \p mask. If calculated id
/// is outside the logical sub_group id range, the work-item will get value from
/// itself. The \p logical_sub_group_size must be a power of 2 and not exceed
/// input sub_group size.
/// \tparam T Input value type
/// \param [in] g Input sub_group
/// \param [in] x Input value
/// \param [in] mask Input mask
/// \param [in] logical_sub_group_size Input logical sub_group size
/// \returns The result
template <typename T>
T permute_sub_group_by_xor(sycl::sub_group g, T x, unsigned int mask,
                           int logical_sub_group_size = 32) {
  unsigned int id = g.get_local_linear_id();
  unsigned int start_index =
      id / logical_sub_group_size * logical_sub_group_size;
  unsigned int target_offset = (id % logical_sub_group_size) ^ mask;
  return sycl::select_from_group(g, x,
                                 target_offset < logical_sub_group_size
                                     ? start_index + target_offset
                                     : id);
}

/// Inherited from the original SYCLomatic compatibility headers.
/// @return compiler's SYCL version if defined, 202000 otherwise.
inline int get_sycl_language_version() {
#ifdef SYCL_LANGUAGE_VERSION
  return SYCL_LANGUAGE_VERSION;
#else
  return 202000;
#endif
}

namespace experimental {
/// Synchronize work items from all work groups within a SYCL kernel.
/// \param [in] item:  Represents a work group.
/// \param [in] counter: An atomic object defined on a device memory which can
/// be accessed by work items in all work groups. The initial value of the
/// counter should be zero.
/// Note: Please make sure that all the work items of all work groups within
/// a SYCL kernel can be scheduled actively at the same time on a device.
template <int dimensions = 3>
inline void nd_range_barrier(
    sycl::nd_item<dimensions> item,
    sycl::atomic_ref<unsigned int, sycl::memory_order::acq_rel,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> &counter) {

  static_assert(dimensions == 3, "dimensions must be 3.");
  constexpr unsigned int MSB32_MASK = 0x80000000;

  unsigned int num_groups = item.get_group_range(2) * item.get_group_range(1) *
                            item.get_group_range(0);

  item.barrier();

  if (item.get_local_linear_id() == 0) {
    unsigned int inc = 1;
    unsigned int old_arrive = 0;
    bool is_group0 =
        (item.get_group(2) + item.get_group(1) + item.get_group(0) == 0);
    if (is_group0) {
      inc = MSB32_MASK - (num_groups - 1);
    }

    old_arrive = counter.fetch_add(inc);
    // Synchronize all the work groups
    while (((old_arrive ^ counter.load()) & MSB32_MASK) == 0)
      ;
  }

  item.barrier();
}

/// Synchronize work items from all work groups within a SYCL kernel.
/// \param [in] item:  Represents a work group.
/// \param [in] counter: An atomic object defined on a device memory which can
/// be accessed by work items in all work groups. The initial value of the
/// counter should be zero.
/// Note: Please make sure that all the work items of all work groups within
/// a SYCL kernel can be scheduled actively at the same time on a device.
template <>
inline void nd_range_barrier(
    sycl::nd_item<1> item,
    sycl::atomic_ref<unsigned int, sycl::memory_order::acq_rel,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> &counter) {
  unsigned int num_groups = item.get_group_range(0);
  constexpr unsigned int MSB32_MASK = 0x80000000;

  item.barrier();

  if (item.get_local_linear_id() == 0) {
    unsigned int inc = 1;
    unsigned int old_arrive = 0;
    bool is_group0 = (item.get_group(0) == 0);
    if (is_group0) {
      inc = MSB32_MASK - (num_groups - 1);
    }

    old_arrive = counter.fetch_add(inc);
    // Synchronize all the work groups
    while (((old_arrive ^ counter.load()) & MSB32_MASK) == 0)
      ;
  }

  item.barrier();
}

/// The logical-group is a logical collection of some work-items within a
/// work-group.
/// Note: Please make sure that the logical-group size is a power of 2 in the
/// range [1, current_sub_group_size].
class logical_group {
  sycl::nd_item<3> _item;
  sycl::group<3> _g;
  uint32_t _logical_group_size;
  uint32_t _group_linear_range_in_parent;

public:
  /// Dividing \p parent_group into several logical-groups.
  /// \param [in] item Current work-item.
  /// \param [in] parent_group The group to be divided.
  /// \param [in] size The logical-group size.
  logical_group(sycl::nd_item<3> item, sycl::group<3> parent_group,
                uint32_t size)
      : _item(item), _g(parent_group), _logical_group_size(size) {
    _group_linear_range_in_parent =
        (_g.get_local_linear_range() - 1) / _logical_group_size + 1;
  }
  /// Returns the index of the work-item within the logical-group.
  uint32_t get_local_linear_id() const {
    return _item.get_local_linear_id() % _logical_group_size;
  }
  /// Returns the index of the logical-group in the parent group.
  uint32_t get_group_linear_id() const {
    return _item.get_local_linear_id() / _logical_group_size;
  }
  /// Returns the number of work-items in the logical-group.
  uint32_t get_local_linear_range() const {
    if (_g.get_local_linear_range() % _logical_group_size == 0) {
      return _logical_group_size;
    }
    uint32_t last_item_group_id =
        _g.get_local_linear_range() / _logical_group_size;
    uint32_t first_of_last_group = last_item_group_id * _logical_group_size;
    if (_item.get_local_linear_id() >= first_of_last_group) {
      return _g.get_local_linear_range() - first_of_last_group;
    } else {
      return _logical_group_size;
    }
  }
  /// Returns the number of logical-group in the parent group.
  uint32_t get_group_linear_range() const {
    return _group_linear_range_in_parent;
  }
};

// The original source of the functions calculate_max_active_wg_per_xecore and
// calculate_max_potential_wg were under the license below:
//
// Copyright (C) Intel Corporation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
/// This function is used for occupancy calculation, it computes the max active
/// work-group number per Xe-Core. Ref to
/// https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/GPU-Occupancy-Calculator
/// \param [out] num_wg Active work-group number.
/// \param [in] wg_size Work-group size.
/// \param [in] slm_size Share local memory size.
/// \param [in] sg_size Sub-group size.
/// \param [in] used_barrier Whether barrier is used.
/// \param [in] used_large_grf Whether large General Register File is used.
/// \return If no error, returns 0.
/// If \p wg_size exceeds the max work-group size, the max work-group size will
/// be used instead of \p wg_size and returns -1.
inline int calculate_max_active_wg_per_xecore(int *num_wg, int wg_size,
                                              int slm_size = 0,
                                              int sg_size = 32,
                                              bool used_barrier = false,
                                              bool used_large_grf = false) {
  int ret = 0;
  const int slm_size_per_xe_core = 64 * 1024;
  const int max_barrier_registers = 32;
  syclcompat::device_ext &dev = syclcompat::get_current_device();

  size_t max_wg_size = dev.get_info<sycl::info::device::max_work_group_size>();
  if (wg_size > max_wg_size) {
    wg_size = max_wg_size;
    ret = -1;
  }

  int num_threads_ss = 56;
  int max_num_wg = 56;
  if (dev.has(sycl::aspect::ext_intel_gpu_eu_count_per_subslice) &&
      dev.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu)) {
    auto eu_count =
        dev.get_info<sycl::info::device::ext_intel_gpu_eu_count_per_subslice>();
    auto threads_count =
        dev.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>();
    num_threads_ss = eu_count * threads_count;
    max_num_wg = eu_count * threads_count;
  }

  if (used_barrier) {
    max_num_wg = max_barrier_registers;
  }

  // Calculate num_wg_slm
  int num_wg_slm = 0;
  if (slm_size == 0) {
    num_wg_slm = max_num_wg;
  } else {
    num_wg_slm = std::floor((float)slm_size_per_xe_core / slm_size);
  }

  // Calculate num_wg_threads
  if (used_large_grf)
    num_threads_ss = num_threads_ss / 2;
  int num_threads = std::ceil((float)wg_size / sg_size);
  int num_wg_threads = std::floor((float)num_threads_ss / num_threads);

  // Calculate num_wg
  *num_wg = std::min(num_wg_slm, num_wg_threads);
  *num_wg = std::min(*num_wg, max_num_wg);
  return ret;
}

/// This function is used for occupancy calculation, it computes the work-group
/// number and the work-group size which achieves the maximum occupancy of the
/// device potentially. Ref to
/// https://github.com/oneapi-src/oneAPI-samples/tree/master/Tools/GPU-Occupancy-Calculator
/// \param [out] num_wg Work-group number.
/// \param [out] wg_size Work-group size.
/// \param [in] max_wg_size_for_device_code The maximum working work-group size
/// for current device code logic. Zero means no limitation.
/// \param [in] slm_size Share local memory size.
/// \param [in] sg_size Sub-group size.
/// \param [in] used_barrier Whether barrier is used.
/// \param [in] used_large_grf Whether large General Register File is used.
/// \return Returns 0.
inline int calculate_max_potential_wg(int *num_wg, int *wg_size,
                                      int max_wg_size_for_device_code,
                                      int slm_size = 0, int sg_size = 32,
                                      bool used_barrier = false,
                                      bool used_large_grf = false) {
  sycl::device &dev = syclcompat::get_current_device();
  size_t max_wg_size = dev.get_info<sycl::info::device::max_work_group_size>();
  if (max_wg_size_for_device_code == 0 ||
      max_wg_size_for_device_code >= max_wg_size)
    *wg_size = (int)max_wg_size;
  else
    *wg_size = max_wg_size_for_device_code;
  calculate_max_active_wg_per_xecore(num_wg, *wg_size, slm_size, sg_size,
                                     used_barrier, used_large_grf);
  std::uint32_t num_ss = 1;
  if (dev.has(sycl::aspect::ext_intel_gpu_slices) &&
      dev.has(sycl::aspect::ext_intel_gpu_subslices_per_slice)) {
    num_ss =
        dev.get_info<sycl::ext::intel::info::device::gpu_slices>() *
        dev.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
  }
  num_wg[0] = num_ss * num_wg[0];
  return 0;
}

} // namespace experimental
} // namespace syclcompat
