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
 *  memory.hpp
 *
 *  Description:
 *    memory functionality for the SYCL compatibility extension
 **************************************************************************/

// The original source was under the license below:
//==---- memory.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <sycl/builtins.hpp>
#include <sycl/ext/oneapi/group_local_memory.hpp>
#include <sycl/usm.hpp>

#include <syclcompat/device.hpp>

#if defined(__linux__)
#include <sys/mman.h>
#elif defined(_WIN64)
#define NOMINMAX
#include <windows.h>
#else
#error "Only support Windows and Linux."
#endif

namespace syclcompat {

template <typename AllocT> auto *local_mem() {
  sycl::multi_ptr<AllocT, sycl::access::address_space::local_space>
      As_multi_ptr = sycl::ext::oneapi::group_local_memory<AllocT>(
          sycl::ext::oneapi::experimental::this_nd_item<3>().get_group());
  auto *As = *As_multi_ptr;
  return As;
}

namespace detail {
enum memcpy_direction {
  host_to_host,
  host_to_device,
  device_to_host,
  device_to_device,
  automatic
};
} // namespace detail

enum class memory_region {
  global = 0, // device global memory
  constant,   // device read-only memory
  local,      // device local memory
  usm_shared, // memory which can be accessed by host and device
};

enum class target { device, local };

using byte_t = uint8_t;

/// Pitched 2D/3D memory data.
class pitched_data {
public:
  pitched_data() : pitched_data(nullptr, 0, 0, 0) {}
  pitched_data(void *data, size_t pitch, size_t x, size_t y)
      : _data(data), _pitch(pitch), _x(x), _y(y) {}

  void *get_data_ptr() { return _data; }
  void set_data_ptr(void *data) { _data = data; }

  size_t get_pitch() { return _pitch; }
  void set_pitch(size_t pitch) { _pitch = pitch; }

  size_t get_x() { return _x; }
  void set_x(size_t x) { _x = x; };

  size_t get_y() { return _y; }
  void set_y(size_t y) { _y = y; }

private:
  void *_data;
  size_t _pitch, _x, _y;
};

namespace detail {

template <class T, memory_region Memory, size_t Dimension> class accessor;
template <memory_region Memory, class T = byte_t> class memory_traits {
public:
  static constexpr sycl::access::address_space asp =
      (Memory == memory_region::local)
          ? sycl::access::address_space::local_space
          : sycl::access::address_space::global_space;
  static constexpr syclcompat::target target = (Memory == memory_region::local)
                                                   ? syclcompat::target::local
                                                   : syclcompat::target::device;
  static constexpr sycl::access_mode mode = (Memory == memory_region::constant)
                                                ? sycl::access_mode::read
                                                : sycl::access_mode::read_write;
  static constexpr size_t type_size = sizeof(T);
  using element_t =
      typename std::conditional_t<Memory == memory_region::constant, const T,
                                  T>;
  using value_t = typename std::remove_cv_t<T>;
  template <size_t Dimension = 1>
  using accessor_t =
      typename std::conditional_t<target == syclcompat::target::local,
                                  sycl::local_accessor<T, Dimension>,
                                  sycl::accessor<T, Dimension, mode>>;
  using pointer_t = T *;
};

static inline void *malloc(size_t size, sycl::queue q) {
  return sycl::malloc_device(size, q.get_device(), q.get_context());
}

/// Calculate pitch (padded length of major dimension \p x) by rounding up to
/// multiple of 32.
/// \param x The dimension to be padded
/// \returns size_t representing pitched length of dimension x.
static inline constexpr size_t get_pitch(size_t x) {
  return ((x) + 31) & ~(0x1F);
}

static inline void *malloc(size_t &pitch, size_t x, size_t y, size_t z,
                           sycl::queue q) {
  pitch = get_pitch(x);
  return malloc(pitch * y * z, q);
}

/// Set \p pattern to the first \p count elements of type \p T starting from \p
/// dev_ptr.
///
/// \tparam T Datatype of the pattern to be set.
/// \param q The queue in which the operation is done.
/// \param dev_ptr Pointer to the device memory address.
/// \param pattern Pattern of type T to be set.
/// \param count Number of elements to be set to the patten.
/// \returns An event representing the fill operation.
template <class T>
static inline sycl::event fill(sycl::queue q, void *dev_ptr, const T &pattern,
                               size_t count) {
  return q.fill(dev_ptr, pattern, count);
}

/// Set \p value to the first \p size bytes starting from \p dev_ptr in \p q.
///
/// \param q The queue in which the operation is done.
/// \param dev_ptr Pointer to the device memory address.
/// \param value Value to be set.
/// \param size Number of bytes to be set to the value.
/// \returns An event representing the memset operation.
static inline sycl::event memset(sycl::queue q, void *dev_ptr, int value,
                                 size_t size) {
  return q.memset(dev_ptr, value, size);
}

/// Set \p value to the 3D memory region pointed by \p data in \p q. \p size
/// specifies the 3D memory size to set.
///
/// \param q The queue in which the operation is done.
/// \param data Pointer to the device memory region.
/// \param value Value to be set.
/// \param size Memory region size.
/// \returns An event list representing the memset operations.
static inline std::vector<sycl::event> memset(sycl::queue q, pitched_data data,
                                              int value, sycl::range<3> size) {
  std::vector<sycl::event> event_list;
  size_t slice = data.get_pitch() * data.get_y();
  unsigned char *data_surface = (unsigned char *)data.get_data_ptr();
  for (size_t z = 0; z < size.get(2); ++z) {
    unsigned char *data_ptr = data_surface;
    for (size_t y = 0; y < size.get(1); ++y) {
      event_list.push_back(memset(q, data_ptr, value, size.get(0)));
      data_ptr += data.get_pitch();
    }
    data_surface += slice;
  }
  return event_list;
}

/// memset 2D matrix with pitch.
static inline std::vector<sycl::event>
memset(sycl::queue q, void *ptr, size_t pitch, int val, size_t x, size_t y) {
  return memset(q, pitched_data(ptr, pitch, x, 1), val,
                sycl::range<3>(x, y, 1));
}

enum class pointer_access_attribute {
  host_only = 0,
  device_only,
  host_device,
  end
};

static pointer_access_attribute get_pointer_attribute(sycl::queue q,
                                                      const void *ptr) {
  switch (sycl::get_pointer_type(ptr, q.get_context())) {
  case sycl::usm::alloc::unknown:
    return pointer_access_attribute::host_only;
  case sycl::usm::alloc::device:
    return pointer_access_attribute::device_only;
  case sycl::usm::alloc::shared:
  case sycl::usm::alloc::host:
    return pointer_access_attribute::host_device;
  }
}

static memcpy_direction deduce_memcpy_direction(sycl::queue q, void *to_ptr,
                                                const void *from_ptr) {
  // table[to_attribute][from_attribute]
  static const memcpy_direction
      direction_table[static_cast<unsigned>(pointer_access_attribute::end)]
                     [static_cast<unsigned>(pointer_access_attribute::end)] = {
                         {memcpy_direction::host_to_host,
                          memcpy_direction::device_to_host,
                          memcpy_direction::host_to_host},
                         {memcpy_direction::host_to_device,
                          memcpy_direction::device_to_device,
                          memcpy_direction::device_to_device},
                         {memcpy_direction::host_to_host,
                          memcpy_direction::device_to_device,
                          memcpy_direction::device_to_device}};
  return direction_table[static_cast<unsigned>(get_pointer_attribute(
      q, to_ptr))][static_cast<unsigned>(get_pointer_attribute(q, from_ptr))];
}

static sycl::event memcpy(sycl::queue q, void *to_ptr, const void *from_ptr,
                          size_t size,
                          const std::vector<sycl::event> &dep_events = {}) {
  if (!size)
    return sycl::event{};
  return q.memcpy(to_ptr, from_ptr, size, dep_events);
}

// Get actual copy range and make sure it will not exceed range.
static inline size_t get_copy_range(sycl::range<3> size, size_t slice,
                                    size_t pitch) {
  return slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
}

static inline size_t get_offset(sycl::id<3> id, size_t slice, size_t pitch) {
  return slice * id.get(2) + pitch * id.get(1) + id.get(0);
}

/// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
/// and \p from_range to another specified by \p to_ptr and \p to_range.
static inline std::vector<sycl::event>
memcpy(sycl::queue q, void *to_ptr, const void *from_ptr,
       sycl::range<3> to_range, sycl::range<3> from_range, sycl::id<3> to_id,
       sycl::id<3> from_id, sycl::range<3> size,
       const std::vector<sycl::event> &dep_events = {}) {
  // RAII for host pointer
  class host_buffer {
    void *_buf;
    size_t _size;
    sycl::queue _q;
    const std::vector<sycl::event> &_deps; // free operation depends

  public:
    host_buffer(size_t size, sycl::queue q,
                const std::vector<sycl::event> &deps)
        : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
    void *get_ptr() const { return _buf; }
    size_t get_size() const { return _size; }
    ~host_buffer() {
      if (_buf) {
        _q.submit([&](sycl::handler &cgh) {
          cgh.depends_on(_deps);
          cgh.host_task([buf = _buf] { std::free(buf); });
        });
      }
    }
  };
  std::vector<sycl::event> event_list;

  size_t to_slice = to_range.get(1) * to_range.get(0);
  size_t from_slice = from_range.get(1) * from_range.get(0);
  unsigned char *to_surface =
      (unsigned char *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
  const unsigned char *from_surface =
      (const unsigned char *)from_ptr +
      get_offset(from_id, from_slice, from_range.get(0));

  if (to_slice == from_slice && to_slice == size.get(1) * size.get(0)) {
    return {memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                   dep_events)};
  }
  memcpy_direction direction = deduce_memcpy_direction(q, to_ptr, from_ptr);
  size_t size_slice = size.get(1) * size.get(0);
  switch (direction) {
  case host_to_host:
    for (size_t z = 0; z < size.get(2); ++z) {
      unsigned char *to_ptr = to_surface;
      const unsigned char *from_ptr = from_surface;
      if (to_range.get(0) == from_range.get(0) &&
          to_range.get(0) == size.get(0)) {
        event_list.push_back(
            memcpy(q, to_ptr, from_ptr, size_slice, dep_events));
      } else {
        for (size_t y = 0; y < size.get(1); ++y) {
          event_list.push_back(
              memcpy(q, to_ptr, from_ptr, size.get(0), dep_events));
          to_ptr += to_range.get(0);
          from_ptr += from_range.get(0);
        }
      }
      to_surface += to_slice;
      from_surface += from_slice;
    }
    break;
  case host_to_device: {
    host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                    event_list);
    std::vector<sycl::event> host_events;
    if (to_slice == size_slice) {
      // Copy host data to a temp host buffer with the shape of target.
      host_events =
          memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                 sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, dep_events);
    } else {
      // Copy host data to a temp host buffer with the shape of target.
      host_events =
          memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                 sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                 // If has padding data, not sure whether it is useless. So fill
                 // temp buffer with it.
                 std::vector<sycl::event>{memcpy(q, buf.get_ptr(), to_surface,
                                                 buf.get_size(), dep_events)});
    }
    // Copy from temp host buffer to device with only one submit.
    event_list.push_back(
        memcpy(q, to_surface, buf.get_ptr(), buf.get_size(), host_events));
    break;
  }
  case device_to_host: {
    host_buffer buf(get_copy_range(size, from_slice, from_range.get(0)), q,
                    event_list);
    // Copy from host temp buffer to host target with reshaping.
    event_list =
        memcpy(q, to_surface, buf.get_ptr(), to_range, from_range,
               sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
               // Copy from device to temp host buffer with only one submit.
               std::vector<sycl::event>{memcpy(q, buf.get_ptr(), from_surface,
                                               buf.get_size(), dep_events)});
    break;
  }
  case device_to_device:
    event_list.push_back(q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(dep_events);
      cgh.parallel_for<class memcpy_3d_detail>(size, [=](sycl::id<3> id) {
        to_surface[get_offset(id, to_slice, to_range.get(0))] =
            from_surface[get_offset(id, from_slice, from_range.get(0))];
      });
    }));
    break;
  default:
    throw std::runtime_error("syclcompat::"
                             "memcpy: invalid direction value");
  }
  return event_list;
}

/// memcpy 2D/3D matrix specified by pitched_data.
static inline std::vector<sycl::event>
memcpy(sycl::queue q, pitched_data to, sycl::id<3> to_id, pitched_data from,
       sycl::id<3> from_id, sycl::range<3> size) {
  return memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id,
                from_id, size);
}

/// memcpy 2D matrix with pitch.
static inline std::vector<sycl::event>
memcpy(sycl::queue q, void *to_ptr, const void *from_ptr, size_t to_pitch,
       size_t from_pitch, size_t x, size_t y) {
  return memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                sycl::range<3>(from_pitch, y, 1), sycl::id<3>(0, 0, 0),
                sycl::id<3>(0, 0, 0), sycl::range<3>(x, y, 1));
}

// Takes a std::vector<sycl::event> & returns a single event
// which simply depends on all of them
static sycl::event combine_events(std::vector<sycl::event> &events,
                                  sycl::queue q) {
  return q.submit([&events](sycl::handler &cgh) {
    cgh.depends_on(events);
    cgh.host_task([]() {});
  });
}

} // namespace detail

/// Allocate memory block on the device.
/// \param num_bytes Number of bytes to allocate.
/// \param q Queue to execute the allocate task.
/// \returns A pointer to the newly allocated memory.
static inline void *malloc(size_t num_bytes,
                           sycl::queue q = get_default_queue()) {
  return detail::malloc(num_bytes, q);
}

/// Allocate memory block on the device.
/// \param T Datatype to allocate
/// \param count Number of elements to allocate.
/// \param q Queue to execute the allocate task.
/// \returns A pointer to the newly allocated memory.
template <typename T>
static inline T *malloc(size_t count, sycl::queue q = get_default_queue()) {
  return static_cast<T *>(detail::malloc(count * sizeof(T), q));
}

/// Allocate memory block on the host.
/// \param num_bytes Number of bytes to allocate.
/// \param q Queue to execute the allocate task.
/// \returns A pointer to the newly allocated memory.
static inline void *malloc_host(size_t num_bytes,
                                sycl::queue q = get_default_queue()) {
  return sycl::malloc_host(num_bytes, q);
}

/// Allocate memory block on the host.
/// \param T Datatype to allocate
/// \param num_bytes Number of bytes to allocate.
/// \param q Queue to execute the allocate task.
/// \returns A pointer to the newly allocated memory.
template <typename T>
static inline T *malloc_host(size_t count,
                             sycl::queue q = get_default_queue()) {
  return static_cast<T *>(sycl::malloc_host(count * sizeof(T), q));
}

/// Allocate memory block of usm_shared memory.
/// \param num_bytes Number of bytes to allocate.
/// \param q Queue to execute the allocate task.
/// \returns A pointer to the newly allocated memory.
static inline void *malloc_shared(size_t num_bytes,
                                  sycl::queue q = get_default_queue()) {
  return sycl::malloc_shared(num_bytes, q);
}

/// Allocate memory block of usm_shared memory.
/// \param num_bytes Number of bytes to allocate.
/// \param q Queue to execute the allocate task.
/// \returns A pointer to the newly allocated memory.
template <typename T>
static inline T *malloc_shared(size_t count,
                               sycl::queue q = get_default_queue()) {
  return static_cast<T *>(sycl::malloc_shared(count * sizeof(T), q));
}

/// Allocate memory block for 3D array on the device.
/// \param size Size of the memory block, in bytes.
/// \param q Queue to execute the allocate task.
/// \returns A pitched_data object which stores the memory info.
static inline pitched_data malloc(sycl::range<3> size,
                                  sycl::queue q = get_default_queue()) {
  pitched_data pitch(nullptr, 0, size.get(0), size.get(1));
  size_t pitch_size;
  pitch.set_data_ptr(
      detail::malloc(pitch_size, size.get(0), size.get(1), size.get(2), q));
  pitch.set_pitch(pitch_size);
  return pitch;
}

/// Allocate memory block for 2D array on the device.
/// \param [out] pitch Aligned size of x in bytes.
/// \param x Range in dim x.
/// \param y Range in dim y.
/// \param q Queue to execute the allocate task.
/// \returns A pointer to the newly allocated memory.
static inline void *malloc(size_t &pitch, size_t x, size_t y,
                           sycl::queue q = get_default_queue()) {
  return detail::malloc(pitch, x, y, 1, q);
}

/// free
/// \param ptr Point to free.
/// \param q Queue to execute the free task.
/// \returns no return value.
static inline void free(void *ptr, sycl::queue q = get_default_queue()) {
  if (ptr) {
    sycl::free(ptr, q.get_context());
  }
}

/// Free the device memory pointed by a batch of pointers in \p pointers which
/// are related to \p q after \p events completed.
///
/// \param pointers The pointers point to the device memory requested to be
/// freed. \param events The events to be waited. \param q The sycl::queue the
/// memory relates to.
inline sycl::event free_async(const std::vector<void *> &pointers,
                              const std::vector<sycl::event> &events,
                              sycl::queue q = get_default_queue()) {
  auto event = q.submit(
      [&pointers, &events, ctxt = q.get_context()](sycl::handler &cgh) {
        cgh.depends_on(events);
        cgh.host_task([=]() {
          for (auto p : pointers)
            sycl::free(p, ctxt);
        });
      });
  get_current_device().add_event(event);
  return event;
}

/// Synchronously copies \p size bytes from the address specified by \p from_ptr
/// to the address specified by \p to_ptr. The function will
/// return after the copy is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param from_ptr Pointer to source memory address.
/// \param size Number of bytes to be copied.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static void memcpy(void *to_ptr, const void *from_ptr, size_t size,
                   sycl::queue q = get_default_queue()) {
  detail::memcpy(q, to_ptr, from_ptr, size).wait();
}

/// Asynchronously copies \p size bytes from the address specified by \p
/// from_ptr to the address specified by \p to_ptr. The return of the function
/// does NOT guarantee the copy is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param from_ptr Pointer to source memory address.
/// \param size Number of bytes to be copied.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static sycl::event memcpy_async(void *to_ptr, const void *from_ptr, size_t size,
                                sycl::queue q = get_default_queue()) {
  return detail::memcpy(q, to_ptr, from_ptr, size);
}

namespace detail {
template <class T> struct dont_deduce {
  using type = T;
};
template <class T> using dont_deduce_t = typename dont_deduce<T>::type;
} // namespace detail

/// Asynchronously copies \p count T's from the address specified by \p
/// from_ptr to the address specified by \p to_ptr. The return of the function
/// does NOT guarantee the copy is completed.
///
/// \tparam T Datatype to be copied.
/// \param to_ptr Pointer to destination memory address.
/// \param from_ptr Pointer to source memory address.
/// \param count Number of T to be copied.
/// \param q Queue to execute the copy task.
/// \returns no return value.
template <typename T>
static sycl::event memcpy_async(detail::dont_deduce_t<T> *to_ptr,
                                const detail::dont_deduce_t<T> *from_ptr,
                                size_t count,
                                sycl::queue q = get_default_queue()) {
  return detail::memcpy(q, static_cast<void *>(to_ptr),
                        static_cast<const void *>(from_ptr), count * sizeof(T));
}

/// Synchronously copies \p count T's from the address specified by \p from_ptr
/// to the address specified by \p to_ptr. The function will
/// return after the copy is completed.
///
/// \tparam T Datatype to be copied.
/// \param to_ptr Pointer to destination memory address.
/// \param from_ptr Pointer to source memory address.
/// \param count Number of T to be copied.
/// \param q Queue to execute the copy task.
/// \returns no return value.
template <typename T>
static void memcpy(detail::dont_deduce_t<T> *to_ptr,
                   const detail::dont_deduce_t<T> *from_ptr, size_t count,
                   sycl::queue q = get_default_queue()) {
  detail::memcpy(q, static_cast<void *>(to_ptr),
                 static_cast<const void *>(from_ptr), count * sizeof(T))
      .wait();
}

/// Synchronously copies 2D matrix specified by \p x and \p y from the address
/// specified by \p from_ptr to the address specified by \p to_ptr, while \p
/// from_pitch and \p to_pitch are the range of dim x in bytes of the matrix
/// specified by \p from_ptr and \p to_ptr. The function will return after the
/// copy is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param to_pitch Range of dim x in bytes of destination matrix.
/// \param from_ptr Pointer to source memory address.
/// \param from_pitch Range of dim x in bytes of source matrix.
/// \param x Range of dim x of matrix to be copied.
/// \param y Range of dim y of matrix to be copied.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline void memcpy(void *to_ptr, size_t to_pitch, const void *from_ptr,
                          size_t from_pitch, size_t x, size_t y,
                          sycl::queue q = get_default_queue()) {
  sycl::event::wait(
      detail::memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch, x, y));
}

/// Asynchronously copies 2D matrix specified by \p x and \p y from the address
/// specified by \p from_ptr to the address specified by \p to_ptr, while \p
/// \p from_pitch and \p to_pitch are the range of dim x in bytes of the matrix
/// specified by \p from_ptr and \p to_ptr. The return of the function does NOT
/// guarantee the copy is completed.
///
/// \param to_ptr Pointer to destination memory address.
/// \param to_pitch Range of dim x in bytes of destination matrix.
/// \param from_ptr Pointer to source memory address.
/// \param from_pitch Range of dim x in bytes of source matrix.
/// \param x Range of dim x of matrix to be copied.
/// \param y Range of dim y of matrix to be copied.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline sycl::event memcpy_async(void *to_ptr, size_t to_pitch,
                                       const void *from_ptr, size_t from_pitch,
                                       size_t x, size_t y,
                                       sycl::queue q = get_default_queue()) {
  auto events = detail::memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch, x, y);
  return detail::combine_events(events, q);
}

/// Synchronously copies a subset of a 3D matrix specified by \p to to another
/// 3D matrix specified by \p from. The from and to position info are specified
/// by \p from_pos and \p to_pos The copied matrix size is specified by \p size.
// The function will return after the copy is completed.
///
/// \param to Destination matrix info.
/// \param to_pos Position of destination.
/// \param from Source matrix info.
/// \param from_pos Position of destination.
/// \param size Range of the submatrix to be copied.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline void memcpy(pitched_data to, sycl::id<3> to_pos,
                          pitched_data from, sycl::id<3> from_pos,
                          sycl::range<3> size,
                          sycl::queue q = get_default_queue()) {
  sycl::event::wait(detail::memcpy(q, to, to_pos, from, from_pos, size));
}

/// Asynchronously copies a subset of a 3D matrix specified by \p to to another
/// 3D matrix specified by \p from. The from and to position info are specified
/// by \p from_pos and \p to_pos The copied matrix size is specified by \p size.
/// The return of the function does NOT guarantee the copy is completed.
///
/// \param to Destination matrix info.
/// \param to_pos Position of destination.
/// \param from Source matrix info.
/// \param from_pos Position of destination.
/// \param size Range of the submatrix to be copied.
/// \param q Queue to execute the copy task.
/// \returns no return value.
static inline sycl::event memcpy_async(pitched_data to, sycl::id<3> to_pos,
                                       pitched_data from, sycl::id<3> from_pos,
                                       sycl::range<3> size,
                                       sycl::queue q = get_default_queue()) {
  auto events = detail::memcpy(q, to, to_pos, from, from_pos, size);
  return detail::combine_events(events, q);
}

/// Synchronously sets \p pattern to the first \p count elements starting from
/// \p dev_ptr. The function will return after the fill operation is completed.
///
/// \tparam T Datatype of the value to be set.
/// \param dev_ptr Pointer to the device memory address.
/// \param pattern Pattern of type \p T to be set.
/// \param count Number of elements to be set to the patten.
/// \param q The queue in which the operation is done.
/// \returns no return value.
template <class T>
static void inline fill(void *dev_ptr, const T &pattern, size_t count,
                        sycl::queue q = get_default_queue()) {
  detail::fill(q, dev_ptr, pattern, count).wait();
}

/// Asynchronously sets \p pattern to the first \p count elements starting from
/// \p dev_ptr.
/// The return of the function does NOT guarantee the fill operation is
/// completed.
///
/// \tparam T Datatype of the pattern to be set.
/// \param dev_ptr Pointer to the device memory address.
/// \param pattern Pattern of type \p T to be set.
/// \param count Number of elements to be set to the patten.
/// \param q The queue in which the operation is done.
/// \returns no return value.
template <class T>
static sycl::event inline fill_async(void *dev_ptr, const T &pattern,
                                     size_t count,
                                     sycl::queue q = get_default_queue()) {
  return detail::fill(q, dev_ptr, pattern, count);
}

/// Synchronously sets \p value to the first \p size bytes starting from \p
/// dev_ptr. The function will return after the memset operation is completed.
///
/// \param dev_ptr Pointer to the device memory address.
/// \param value Value to be set.
/// \param size Number of bytes to be set to the value.
/// \param q The queue in which the operation is done.
/// \returns no return value.
static void memset(void *dev_ptr, int value, size_t size,
                   sycl::queue q = get_default_queue()) {
  detail::memset(q, dev_ptr, value, size).wait();
}

/// Asynchronously sets \p value to the first \p size bytes starting from \p
/// dev_ptr. The return of the function does NOT guarantee the memset operation
/// is completed.
///
/// \param dev_ptr Pointer to the device memory address.
/// \param value Value to be set.
/// \param size Number of bytes to be set to the value.
/// \returns no return value.
static sycl::event memset_async(void *dev_ptr, int value, size_t size,
                                sycl::queue q = get_default_queue()) {
  return detail::memset(q, dev_ptr, value, size);
}

/// Sets \p value to the 2D memory region pointed by \p ptr in \p q. \p x and
/// \p y specify the setted 2D memory size. \p pitch is the bytes in linear
/// dimension, including padding bytes. The function will return after the
/// memset operation is completed.
///
/// \param ptr Pointer to the device memory region.
/// \param pitch Bytes in linear dimension, including padding bytes.
/// \param value Value to be set.
/// \param x The setted memory size in linear dimension.
/// \param y The setted memory size in second dimension.
/// \param q The queue in which the operation is done.
/// \returns no return value.
static inline void memset(void *ptr, size_t pitch, int val, size_t x, size_t y,
                          sycl::queue q = get_default_queue()) {
  sycl::event::wait(detail::memset(q, ptr, pitch, val, x, y));
}

/// Sets \p value to the 2D memory region pointed by \p ptr in \p q. \p x and
/// \p y specify the setted 2D memory size. \p pitch is the bytes in linear
/// dimension, including padding bytes. The return of the function does NOT
/// guarantee the memset operation is completed.
///
/// \param ptr Pointer to the device memory region.
/// \param pitch Bytes in linear dimension, including padding bytes.
/// \param value Value to be set.
/// \param x The setted memory size in linear dimension.
/// \param y The setted memory size in second dimension.
/// \param q The queue in which the operation is done.
/// \returns no return value.
static inline sycl::event memset_async(void *ptr, size_t pitch, int val,
                                       size_t x, size_t y,
                                       sycl::queue q = get_default_queue()) {
  auto events = detail::memset(q, ptr, pitch, val, x, y);
  return detail::combine_events(events, q);
}

/// Sets \p value to the 3D memory region specified by \p pitch in \p q. \p size
/// specify the setted 3D memory size. The function will return after the
/// memset operation is completed.
///
/// \param pitch Specify the 3D memory region.
/// \param value Value to be set.
/// \param size The setted 3D memory size.
/// \param q The queue in which the operation is done.
/// \returns no return value.
static inline void memset(pitched_data pitch, int val, sycl::range<3> size,
                          sycl::queue q = get_default_queue()) {
  sycl::event::wait(detail::memset(q, pitch, val, size));
}

/// Sets \p value to the 3D memory region specified by \p pitch in \p q. \p size
/// specify the setted 3D memory size. The return of the function does NOT
/// guarantee the memset operation is completed.
///
/// \param pitch Specify the 3D memory region.
/// \param value Value to be set.
/// \param size The setted 3D memory size.
/// \param q The queue in which the operation is done.
/// \returns no return value.
static inline sycl::event memset_async(pitched_data pitch, int val,
                                       sycl::range<3> size,
                                       sycl::queue q = get_default_queue()) {
  auto events = detail::memset(q, pitch, val, size);
  return detail::combine_events(events, q);
}

/// accessor used as device function parameter.
template <class T, memory_region Memory, size_t Dimension> class accessor;
template <class T, memory_region Memory> class accessor<T, Memory, 3> {
public:
  using memory_t = detail::memory_traits<Memory, T>;
  using element_t = typename memory_t::element_t;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<3>;
  accessor(pointer_t data, const sycl::range<3> &in_range)
      : _data(data), _range(in_range) {}
  template <memory_region M = Memory>
  accessor(typename std::enable_if<M != memory_region::local,
                                   const accessor_t>::type &acc)
      : accessor(acc, acc.get_range()) {}
  accessor(const accessor_t &acc, const sycl::range<3> &in_range)
      : accessor(acc.get_pointer(), in_range) {}
  accessor<T, Memory, 2> operator[](size_t index) const {
    sycl::range<2> sub(_range.get(1), _range.get(2));
    return accessor<T, Memory, 2>(_data + index * sub.size(), sub);
  }

  pointer_t get_ptr() const { return _data; }

private:
  pointer_t _data;
  sycl::range<3> _range;
};
template <class T, memory_region Memory> class accessor<T, Memory, 2> {
public:
  using memory_t = detail::memory_traits<Memory, T>;
  using element_t = typename memory_t::element_t;
  using pointer_t = typename memory_t::pointer_t;
  using accessor_t = typename memory_t::template accessor_t<2>;
  accessor(pointer_t data, const sycl::range<2> &in_range)
      : _data(data), _range(in_range) {}
  template <memory_region M = Memory>
  accessor(typename std::enable_if<M != memory_region::local,
                                   const accessor_t>::type &acc)
      : accessor(acc, acc.get_range()) {}
  accessor(const accessor_t &acc, const sycl::range<2> &in_range)
      : accessor(acc.get_pointer(), in_range) {}

  pointer_t operator[](size_t index) const {
    return _data + _range.get(1) * index;
  }

  pointer_t get_ptr() const { return _data; }

private:
  pointer_t _data;
  sycl::range<2> _range;
};

/// Device variable with address space of shared or global.
template <class T, memory_region Memory, size_t Dimension> class device_memory {
public:
  using accessor_t =
      typename detail::memory_traits<Memory, T>::template accessor_t<Dimension>;
  using value_t = typename detail::memory_traits<Memory, T>::value_t;
  using compat_accessor_t = syclcompat::accessor<T, Memory, Dimension>;

  device_memory() : device_memory(sycl::range<Dimension>(1)) {}

  /// Constructor of 1-D array with initializer list
  device_memory(const sycl::range<Dimension> &in_range,
                std::initializer_list<value_t> &&init_list)
      : device_memory(in_range) {
    assert(init_list.size() <= in_range.size());
    _host_ptr = (value_t *)std::malloc(_size);
    std::memset(_host_ptr, 0, _size);
    std::memcpy(_host_ptr, init_list.begin(), init_list.size() * sizeof(T));
  }

  /// Constructor of 2-D array with initializer list
  template <size_t D = Dimension>
  device_memory(
      const typename std::enable_if<D == 2, sycl::range<2>>::type &in_range,
      std::initializer_list<std::initializer_list<value_t>> &&init_list)
      : device_memory(in_range) {
    assert(init_list.size() <= in_range[0]);
    _host_ptr = (value_t *)std::malloc(_size);
    std::memset(_host_ptr, 0, _size);
    auto tmp_data = _host_ptr;
    for (auto sub_list : init_list) {
      assert(sub_list.size() <= in_range[1]);
      std::memcpy(tmp_data, sub_list.begin(), sub_list.size() * sizeof(T));
      tmp_data += in_range[1];
    }
  }

  /// Constructor with range
  device_memory(const sycl::range<Dimension> &range_in)
      : _size(range_in.size() * sizeof(T)), _range(range_in), _reference(false),
        _host_ptr(nullptr), _device_ptr(nullptr) {
    static_assert((Memory == memory_region::global) ||
                      (Memory == memory_region::constant) ||
                      (Memory == memory_region::usm_shared),
                  "device memory region should be global, constant or shared");
    // Make sure that singleton class dev_mgr will destruct later than this.
    detail::dev_mgr::instance();
  }

  /// Constructor with range
  template <class... Args>
  device_memory(Args... Arguments)
      : device_memory(sycl::range<Dimension>(Arguments...)) {}

  ~device_memory() {
    if (_device_ptr && !_reference)
      free(_device_ptr);
    if (_host_ptr)
      std::free(_host_ptr);
  }

  /// Allocate memory with default queue, and init memory if has initial value.
  void init() { init(get_default_queue()); }
  /// Allocate memory with specified queue, and init memory if has initial
  /// value.
  void init(sycl::queue q) {
    if (_device_ptr)
      return;
    if (!_size)
      return;
    allocate_device(q);
    if (_host_ptr)
      detail::memcpy(q, _device_ptr, _host_ptr, _size);
  }

  /// The variable is assigned to a device pointer.
  void assign(value_t *src, size_t size) {
    this->~device_memory();
    new (this) device_memory(src, size);
  }

  /// Get memory pointer of the memory object, which is virtual pointer when
  /// usm is not used, and device pointer when usm is used.
  value_t *get_ptr() { return get_ptr(get_default_queue()); }
  /// Get memory pointer of the memory object, which is virtual pointer when
  /// usm is not used, and device pointer when usm is used.
  value_t *get_ptr(sycl::queue q) {
    init(q);
    return _device_ptr;
  }

  /// Get the device memory object size in bytes.
  size_t get_size() { return _size; }

  template <size_t D = Dimension>
  typename std::enable_if<D == 1, T>::type &operator[](size_t index) {
    init();
    return _device_ptr[index];
  }

  /// Get compat_accessor with dimension info for the device memory object
  /// when usm is used and dimension is greater than 1.
  template <size_t D = Dimension>
  typename std::enable_if<D != 1, compat_accessor_t>::type
  get_access(sycl::handler &cgh) {
    return compat_accessor_t((T *)_device_ptr, _range);
  }

private:
  device_memory(value_t *memory_ptr, size_t size)
      : _size(size), _range(size / sizeof(T)), _reference(true),
        _device_ptr(memory_ptr) {}

  void allocate_device(sycl::queue q) {
    if (Memory == memory_region::usm_shared) {
      _device_ptr = (value_t *)sycl::malloc_shared(_size, q.get_device(),
                                                   q.get_context());
      return;
    }
    _device_ptr = (value_t *)detail::malloc(_size, q);
  }

  size_t _size;
  sycl::range<Dimension> _range;
  bool _reference;
  value_t *_host_ptr;
  value_t *_device_ptr;
};
template <class T, memory_region Memory>
class device_memory<T, Memory, 0> : public device_memory<T, Memory, 1> {
public:
  using base = device_memory<T, Memory, 1>;
  using value_t = typename base::value_t;
  using accessor_t =
      typename detail::memory_traits<Memory, T>::template accessor_t<0>;

  /// Constructor with initial value.
  device_memory(const value_t &val) : base(sycl::range<1>(1), {val}) {}

  /// Default constructor
  device_memory() : base(1) {}
};

template <class T, size_t Dimension>
using global_memory = device_memory<T, memory_region::global, Dimension>;
template <class T, size_t Dimension>
using constant_memory = device_memory<T, memory_region::constant, Dimension>;
template <class T, size_t Dimension>
using shared_memory = device_memory<T, memory_region::usm_shared, Dimension>;

class pointer_attributes {
public:
  void init(const void *ptr, sycl::queue q = get_default_queue()) {
    memory_type = sycl::get_pointer_type(ptr, q.get_context());
    device_pointer = (memory_type != sycl::usm::alloc::unknown) ? ptr : nullptr;
    host_pointer = (memory_type != sycl::usm::alloc::unknown) &&
                           (memory_type != sycl::usm::alloc::device)
                       ? ptr
                       : nullptr;
    sycl::device device_obj = sycl::get_pointer_device(ptr, q.get_context());
    device_id = detail::dev_mgr::instance().get_device_id(device_obj);
  }

  sycl::usm::alloc get_memory_type() { return memory_type; }

  const void *get_device_pointer() { return device_pointer; }

  const void *get_host_pointer() { return host_pointer; }

  bool is_memory_shared() { return memory_type == sycl::usm::alloc::shared; }

  unsigned int get_device_id() { return device_id; }

private:
  sycl::usm::alloc memory_type = sycl::usm::alloc::unknown;
  const void *device_pointer = nullptr;
  const void *host_pointer = nullptr;
  unsigned int device_id = 0;
};

} // namespace syclcompat
