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
 *  atomic.hpp
 *
 *  Description:
 *    Atomic functionality for the SYCL compatibility extension
 **************************************************************************/

// The original source was under the license below:
//==---- atomic.hpp -------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>

#include <sycl/access/access.hpp>
#include <sycl/atomic_ref.hpp>
#include <sycl/memory_enums.hpp>
#include <sycl/multi_ptr.hpp>

namespace syclcompat {

template <typename T> struct arith {
  using type = std::conditional_t<std::is_pointer_v<T>, std::ptrdiff_t, T>;
};
template <typename T> using arith_t = typename arith<T>::type;

/// Atomically add the value operand to the value at the addr and assign the
/// result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to add to the value at \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_add(T *addr, arith_t<T> operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_add(operand);
}

/// Atomically add the value operand to the value at the addr and assign the
/// result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to add to the value at \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_add(T *addr, arith_t<T> operand,
                          sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_add<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_add<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_add<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false &&
           "Invalid memory_order for atomics. Valid memory_order for "
           "atomics are: sycl::memory_order::relaxed, "
           "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

/// Atomically subtract the value operand from the value at the addr and
/// assign the result to the value at addr. \param [in, out] addr The pointer
/// to the data. \param operand The value to subtract from the value at \p
/// addr \param memoryOrder The memory ordering used. \returns The value at
/// the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_sub(T *addr, arith_t<T> operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_sub(operand);
}

/// Atomically subtract the value operand from the value at the addr and
/// assign the result to the value at addr. \param [in, out] addr The pointer
/// to the data. \param operand The value to subtract from the value at \p
/// addr \param memoryOrder The memory ordering used. \returns The value at
/// the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_sub(T *addr, arith_t<T> operand,
                          sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_sub<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_sub<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_sub<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false &&
           "Invalid memory_order for atomics. Valid memory_order for "
           "atomics are: sycl::memory_order::relaxed, "
           "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

/// Atomically perform a bitwise AND between the value operand and the value
/// at the addr and assign the result to the value at addr. \param [in, out]
/// addr The pointer to the data. \param operand The value to use in bitwise
/// AND operation with the value at the \p addr. \param memoryOrder The memory
/// ordering used. \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_and(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_and(operand);
}

/// Atomically perform a bitwise AND between the value operand and the value
/// at the addr and assign the result to the value at addr. \param [in, out]
/// addr The pointer to the data. \param operand The value to use in bitwise
/// AND operation with the value at the \p addr. \param memoryOrder The memory
/// ordering used. \returns The value at the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_and(T *addr, T operand, sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_and<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_and<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_and<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false &&
           "Invalid memory_order for atomics. Valid memory_order for "
           "atomics are: sycl::memory_order::relaxed, "
           "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

/// Atomically or the value at the addr with the value operand, and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise OR operation with the value at
/// the \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_or(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_or(operand);
}

/// Atomically or the value at the addr with the value operand, and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise OR operation with the value at
/// the \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_or(T *addr, T operand, sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_or<T, addressSpace, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_or<T, addressSpace, sycl::memory_order::acq_rel,
                           sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_or<T, addressSpace, sycl::memory_order::seq_cst,
                           sycl::memory_scope::device>(addr, operand);
  default:
    assert(false &&
           "Invalid memory_order for atomics. Valid memory_order for "
           "atomics are: sycl::memory_order::relaxed, "
           "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

/// Atomically xor the value at the addr with the value operand, and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise XOR operation with the value at
/// the \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_xor(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_xor(operand);
}

/// Atomically xor the value at the addr with the value operand, and assign
/// the result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to use in bitwise XOR operation with the value at
/// the \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_xor(T *addr, T operand, sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_xor<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_xor<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_xor<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false &&
           "Invalid memory_order for atomics. Valid memory_order for "
           "atomics are: sycl::memory_order::relaxed, "
           "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

/// Atomically calculate the minimum of the value at addr and the value
/// operand and assign the result to the value at addr. \param [in, out] addr
/// The pointer to the data. \param operand. \param memoryOrder The memory
/// ordering used. \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_min(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_min(operand);
}

/// Atomically calculate the minimum of the value at addr and the value
/// operand and assign the result to the value at addr. \param [in, out] addr
/// The pointer to the data. \param operand. \param memoryOrder The memory
/// ordering used. \returns The value at the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_min(T *addr, T operand, sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_min<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_min<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_min<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false &&
           "Invalid memory_order for atomics. Valid memory_order for "
           "atomics are: sycl::memory_order::relaxed, "
           "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

/// Atomically calculate the maximum of the value at addr and the value
/// operand and assign the result to the value at addr. \param [in, out] addr
/// The pointer to the data. \param operand. \param memoryOrder The memory
/// ordering used. \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_max(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_max(operand);
}

/// Atomically calculate the maximum of the value at addr and the value
/// operand and assign the result to the value at addr. \param [in, out] addr
/// The pointer to the data. \param operand. \param memoryOrder The memory
/// ordering used. \returns The value at the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_fetch_max(T *addr, T operand, sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_max<T, addressSpace, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_max<T, addressSpace, sycl::memory_order::acq_rel,
                            sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_max<T, addressSpace, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device>(addr, operand);
  default:
    assert(false &&
           "Invalid memory_order for atomics. Valid memory_order for "
           "atomics are: sycl::memory_order::relaxed, "
           "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

/// Atomically increment the value stored in \p addr if old value stored in \p
/// addr is less than \p operand, else set 0 to the value stored in \p addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The threshold value.
/// \param memoryOrder The memory ordering used.
/// \returns The old value stored in \p addr.
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline unsigned int atomic_fetch_compare_inc(unsigned int *addr,
                                             unsigned int operand) {
  auto atm =
      sycl::atomic_ref<unsigned int, memoryOrder, memoryScope, addressSpace>(
          addr[0]);
  unsigned int old;
  while (true) {
    old = atm.load();
    if (old >= operand) {
      if (atm.compare_exchange_strong(old, 0))
        break;
    } else if (atm.compare_exchange_strong(old, old + 1))
      break;
  }
  return old;
}

/// Atomically increment the value stored in \p addr if old value stored in \p
/// addr is less than \p operand, else set 0 to the value stored in \p addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The threshold value.
/// \param memoryOrder The memory ordering used.
/// \returns The old value stored in \p addr.
template <sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space>
inline unsigned int atomic_fetch_compare_inc(unsigned int *addr,
                                             unsigned int operand,
                                             sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_fetch_compare_inc<addressSpace, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_fetch_compare_inc<addressSpace, sycl::memory_order::acq_rel,
                                    sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_fetch_compare_inc<addressSpace, sycl::memory_order::seq_cst,
                                    sycl::memory_scope::device>(addr, operand);
  default:
    assert(false &&
           "Invalid memory_order for atomics. Valid memory_order for "
           "atomics are: sycl::memory_order::relaxed, "
           "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

/// Atomically exchange the value at the address addr with the value operand.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to be exchanged with the value pointed by \p
/// addr. \param memoryOrder The memory ordering used. \returns The value at
/// the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_exchange(T *addr, T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.exchange(operand);
}

/// Atomically exchange the value at the address addr with the value operand.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to be exchanged with the value pointed by \p
/// addr. \param memoryOrder The memory ordering used. \returns The value at
/// the \p addr before the call.
template <typename T, sycl::access::address_space addressSpace =
                          sycl::access::address_space::global_space>
inline T atomic_exchange(T *addr, T operand, sycl::memory_order memoryOrder) {
  switch (memoryOrder) {
  case sycl::memory_order::relaxed:
    return atomic_exchange<T, addressSpace, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::acq_rel:
    return atomic_exchange<T, addressSpace, sycl::memory_order::acq_rel,
                           sycl::memory_scope::device>(addr, operand);
  case sycl::memory_order::seq_cst:
    return atomic_exchange<T, addressSpace, sycl::memory_order::seq_cst,
                           sycl::memory_scope::device>(addr, operand);
  default:
    assert(false &&
           "Invalid memory_order for atomics. Valid memory_order for "
           "atomics are: sycl::memory_order::relaxed, "
           "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
  }
}

/// Atomically compare the value at \p addr to the value expected and exchange
/// with the value desired if the value at \p addr is equal to the value
/// expected. Returns the value at the \p addr before the call.
/// \param [in, out] addr Multi_ptr.
/// \param expected The value to compare against the value at \p addr.
/// \param desired The value to assign to \p addr if the value at \p addr
/// is expected.
/// \param success The memory ordering used when comparison succeeds.
/// \param fail The memory ordering used when comparison fails.
/// \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
T atomic_compare_exchange_strong(
    sycl::multi_ptr<T, sycl::access::address_space::global_space> addr,
    T expected, T desired,
    sycl::memory_order success = sycl::memory_order::relaxed,
    sycl::memory_order fail = sycl::memory_order::relaxed) {
  auto atm = sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(*addr);

  atm.compare_exchange_strong(expected, desired, success, fail);
  return expected;
}

/// Atomically compare the value at \p addr to the value expected and exchange
/// with the value desired if the value at \p addr is equal to the value
/// expected. Returns the value at the \p addr before the call.
/// \param [in] addr The pointer to the data.
/// \param expected The value to compare against the value at \p addr.
/// \param desired The value to assign to \p addr if the value at \p addr is
/// expected.
/// \param success The memory ordering used when comparison succeeds.
/// \param fail The memory ordering used when comparison fails.
/// \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
T atomic_compare_exchange_strong(
    T *addr, T expected, T desired,
    sycl::memory_order success = sycl::memory_order::relaxed,
    sycl::memory_order fail = sycl::memory_order::relaxed) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  atm.compare_exchange_strong(expected, desired, success, fail);
  return expected;
}

} // namespace syclcompat
