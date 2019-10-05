//==----------- sub_group.hpp --- SYCL sub-group ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_vars.hpp>
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/intel/functional.hpp>
#include <CL/sycl/range.hpp>
#include <CL/sycl/types.hpp>
#include <type_traits>
#ifdef __SYCL_DEVICE_ONLY__

namespace cl {
namespace sycl {
template <typename T, access::address_space Space> class multi_ptr;

namespace detail {

template <typename> struct is_vec : std::false_type {};
template <typename T, std::size_t N>
struct is_vec<cl::sycl::vec<T, N>> : std::true_type {};

template <typename T, __spv::GroupOperation O>
static typename std::enable_if<
    !detail::is_floating_point<T>::value && std::is_signed<T>::value, T>::type
calc(T x, intel::minimum<T> op) {
  return __spirv_GroupSMin(__spv::Scope::Subgroup, O, x);
}

template <typename T, __spv::GroupOperation O>
static typename std::enable_if<
    !detail::is_floating_point<T>::value && std::is_unsigned<T>::value, T>::type
calc(T x, intel::minimum<T> op) {
  return __spirv_GroupUMin(__spv::Scope::Subgroup, O, x);
}

template <typename T, __spv::GroupOperation O>
static typename std::enable_if<detail::is_floating_point<T>::value, T>::type
calc(T x, intel::minimum<T> op) {
  return __spirv_GroupFMin(__spv::Scope::Subgroup, O, x);
}

template <typename T, __spv::GroupOperation O>
static typename std::enable_if<
    !detail::is_floating_point<T>::value && std::is_signed<T>::value, T>::type
calc(T x, intel::maximum<T> op) {
  return __spirv_GroupSMax(__spv::Scope::Subgroup, O, x);
}

template <typename T, __spv::GroupOperation O>
static typename std::enable_if<
    !detail::is_floating_point<T>::value && std::is_unsigned<T>::value, T>::type
calc(T x, intel::maximum<T> op) {
  return __spirv_GroupUMax(__spv::Scope::Subgroup, O, x);
}

template <typename T, __spv::GroupOperation O>
static typename std::enable_if<detail::is_floating_point<T>::value, T>::type
calc(T x, intel::maximum<T> op) {
  return __spirv_GroupFMax(__spv::Scope::Subgroup, O, x);
}

template <typename T, __spv::GroupOperation O>
static typename std::enable_if<
    !detail::is_floating_point<T>::value && std::is_integral<T>::value, T>::type
calc(T x, intel::plus<T> op) {
  return __spirv_GroupIAdd<T>(__spv::Scope::Subgroup, O, x);
}

template <typename T, __spv::GroupOperation O>
static typename std::enable_if<detail::is_floating_point<T>::value, T>::type
calc(T x, intel::plus<T> op) {
  return __spirv_GroupFAdd<T>(__spv::Scope::Subgroup, O, x);
}

template <typename T, __spv::GroupOperation O,
          template <typename> class BinaryOperation>
static T calc(T x, BinaryOperation<void>) {
  return calc<T, O>(x, BinaryOperation<T>());
}

} // namespace detail

namespace intel {

struct sub_group {
  /* --- common interface members --- */

  id<1> get_local_id() const {
    return __spirv_BuiltInSubgroupLocalInvocationId;
  }
  range<1> get_local_range() const { return __spirv_BuiltInSubgroupSize; }

  range<1> get_max_local_range() const {
    return __spirv_BuiltInSubgroupMaxSize;
  }

  id<1> get_group_id() const { return __spirv_BuiltInSubgroupId; }

  unsigned int get_group_range() const {
    return __spirv_BuiltInNumSubgroups;
  }

  unsigned int get_uniform_group_range() const {
    return __spirv_BuiltInNumEnqueuedSubgroups;
  }

  /* --- vote / ballot functions --- */

  bool any(bool predicate) const {
    return __spirv_GroupAny(__spv::Scope::Subgroup, predicate);
  }

  bool all(bool predicate) const {
    return __spirv_GroupAll(__spv::Scope::Subgroup, predicate);
  }


  template <typename T>
  using EnableIfIsScalarArithmetic = detail::enable_if_t<
    !detail::is_vec<T>::value && detail::is_arithmetic<T>::value, T>;

  /* --- collectives --- */

  template <typename T>
  T broadcast(EnableIfIsScalarArithmetic<T> x, id<1> local_id) const {
    return __spirv_GroupBroadcast<T>(__spv::Scope::Subgroup, x,
                                            local_id.get(0));
  }

  template <typename T, class BinaryOperation>
  EnableIfIsScalarArithmetic<T> reduce(T x, BinaryOperation op) const {
    return detail::calc<T, __spv::GroupOperation::Reduce>(x, op);
  }

  template <typename T, class BinaryOperation>
  EnableIfIsScalarArithmetic<T> reduce(T x, T init, BinaryOperation op) const {
    return op(init, reduce(x, op));
  }

  template <typename T, class BinaryOperation>
  EnableIfIsScalarArithmetic<T> exclusive_scan(T x, BinaryOperation op) const {
    return detail::calc<T, __spv::GroupOperation::ExclusiveScan>(x, op);
  }

  template <typename T, class BinaryOperation>
  EnableIfIsScalarArithmetic<T> exclusive_scan(T x, T init,
                                         BinaryOperation op) const {
    if (get_local_id().get(0) == 0) {
      x = op(init, x);
    }
    T scan = exclusive_scan(x, op);
    if (get_local_id().get(0) == 0) {
      scan = init;
    }
    return scan;
  }

  template <typename T, class BinaryOperation>
  EnableIfIsScalarArithmetic<T> inclusive_scan(T x, BinaryOperation op) const {
    return detail::calc<T, __spv::GroupOperation::InclusiveScan>(x, op);
  }

  template <typename T, class BinaryOperation>
  EnableIfIsScalarArithmetic<T> inclusive_scan(T x, BinaryOperation op,
                                         T init) const {
    if (get_local_id().get(0) == 0) {
      x = op(init, x);
    }
    return inclusive_scan(x, op);
  }

  /* --- one - input shuffles --- */
  /* indices in [0 , sub - group size ) */

  template <typename T>
  EnableIfIsScalarArithmetic<T>
  shuffle(T x, id<1> local_id) const {
    return __spirv_SubgroupShuffleINTEL(x, local_id.get(0));
  }

  template <typename T>
  typename std::enable_if<detail::is_vec<T>::value, T>::type
  shuffle(T x, id<1> local_id) const {
    return __spirv_SubgroupShuffleINTEL((typename T::vector_t)x,
                                               local_id.get(0));
  }

  template <typename T>
  EnableIfIsScalarArithmetic<T>
  shuffle_down(T x, uint32_t delta) const {
    return shuffle_down(x, x, delta);
  }

  template <typename T>
  typename std::enable_if<detail::is_vec<T>::value, T>::type
  shuffle_down(T x, uint32_t delta) const {
    return shuffle_down(x, x, delta);
  }

  template <typename T>
  EnableIfIsScalarArithmetic<T>
  shuffle_up(T x, uint32_t delta) const {
    return shuffle_up(x, x, delta);
  }

  template <typename T>
  typename std::enable_if<detail::is_vec<T>::value, T>::type
  shuffle_up(T x, uint32_t delta) const {
    return shuffle_up(x, x, delta);
  }

  template <typename T>
  EnableIfIsScalarArithmetic<T>
  shuffle_xor(T x, id<1> value) const {
    return __spirv_SubgroupShuffleXorINTEL(x, (uint32_t)value.get(0));
  }

  template <typename T>
  typename std::enable_if<detail::is_vec<T>::value, T>::type
  shuffle_xor(T x, id<1> value) const {
    return __spirv_SubgroupShuffleXorINTEL((typename T::vector_t)x,
                                                  (uint32_t)value.get(0));
  }

  /* --- two - input shuffles --- */
  /* indices in [0 , 2* sub - group size ) */
  template <typename T>
  EnableIfIsScalarArithmetic<T>
  shuffle(T x, T y, id<1> local_id) const {
    return __spirv_SubgroupShuffleDownINTEL(
        x, y, local_id.get(0) - get_local_id().get(0));
  }

  template <typename T>
  typename std::enable_if<detail::is_vec<T>::value, T>::type
  shuffle(T x, T y, id<1> local_id) const {
    return __spirv_SubgroupShuffleDownINTEL(
        (typename T::vector_t)x, (typename T::vector_t)y,
        local_id.get(0) - get_local_id().get(0));
  }

  template <typename T>
  EnableIfIsScalarArithmetic<T>
  shuffle_down(T current, T next, uint32_t delta) const {
    return __spirv_SubgroupShuffleDownINTEL(current, next, delta);
  }

  template <typename T>
  typename std::enable_if<detail::is_vec<T>::value, T>::type
  shuffle_down(T current, T next, uint32_t delta) const {
    return __spirv_SubgroupShuffleDownINTEL(
        (typename T::vector_t)current, (typename T::vector_t)next, delta);
  }

  template <typename T>
  EnableIfIsScalarArithmetic<T>
  shuffle_up(T previous, T current, uint32_t delta) const {
    return __spirv_SubgroupShuffleUpINTEL(previous, current, delta);
  }

  template <typename T>
  typename std::enable_if<detail::is_vec<T>::value, T>::type
  shuffle_up(T previous, T current, uint32_t delta) const {
    return __spirv_SubgroupShuffleUpINTEL(
        (typename T::vector_t)previous, (typename T::vector_t)current, delta);
  }

  /* --- sub - group load / stores --- */
  /* these can map to SIMD or block read / write hardware where available */

  template <typename T, access::address_space Space>
  typename std::enable_if<(sizeof(T) == sizeof(uint32_t) ||
                           sizeof(T) == sizeof(uint16_t)) &&
                              Space == access::address_space::global_space,
                          T>::type
  load(const multi_ptr<T, Space> src) const {
    if (sizeof(T) == sizeof(uint32_t)) {
      uint32_t t = __spirv_SubgroupBlockReadINTEL<uint32_t>(
          (const __attribute__((ocl_global)) uint32_t *)src.get());
      return *((T *)(&t));
    }
    uint16_t t = __spirv_SubgroupBlockReadINTEL<uint16_t>(
        (const __attribute__((ocl_global)) uint16_t *)src.get());
    return *((T *)(&t));
  }

  template <int N, typename T, access::address_space Space>
  vec<typename std::enable_if<(sizeof(T) == sizeof(uint32_t) ||
                               sizeof(T) == sizeof(uint16_t)) &&
                                  Space == access::address_space::global_space,
                              T>::type,
      N>
  load(const multi_ptr<T, Space> src) const {
    if (N == 1) {
      return load<T, Space>(src);
    }
    if (sizeof(T) == sizeof(uint32_t)) {
      typedef uint32_t ocl_t __attribute__((ext_vector_type(N)));

      ocl_t t = __spirv_SubgroupBlockReadINTEL<ocl_t>(
          (const __attribute__((ocl_global)) uint32_t *)src.get());
      return *((typename vec<T, N>::vector_t *)(&t));
    }
    typedef uint16_t ocl_t __attribute__((ext_vector_type(N)));

    ocl_t t = __spirv_SubgroupBlockReadINTEL<ocl_t>(
        (const __attribute__((ocl_global)) uint16_t *)src.get());
    return *((typename vec<T, N>::vector_t *)(&t));
  }

  template <typename T, access::address_space Space>
  void
  store(multi_ptr<T, Space> dst,
        const typename std::enable_if<
            (sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint16_t)) &&
                Space == access::address_space::global_space,
            T>::type &x) const {
    if (sizeof(T) == sizeof(uint32_t)) {
      __spirv_SubgroupBlockWriteINTEL<uint32_t>(
          (__attribute__((ocl_global)) uint32_t *)dst.get(), *((uint32_t *)&x));
    } else {
      __spirv_SubgroupBlockWriteINTEL<uint16_t>(
          (__attribute__((ocl_global)) uint16_t *)dst.get(), *((uint16_t *)&x));
    }
  }

  template <int N, typename T, access::address_space Space>
  void store(multi_ptr<T, Space> dst,
             const vec<typename std::enable_if<N == 1, T>::type, N> &x) const {
    store<T, Space>(dst, x);
  }

  template <int N, typename T, access::address_space Space>
  void store(
      multi_ptr<T, Space> dst,
      const vec<typename std::enable_if<
                    (sizeof(T) == sizeof(uint32_t) ||
                     sizeof(T) == sizeof(uint16_t)) &&
                        N != 1 && Space == access::address_space::global_space,
                    T>::type,
                N> &x) const {
    if (sizeof(T) == sizeof(uint32_t)) {
      typedef uint32_t ocl_t __attribute__((ext_vector_type(N)));
      __spirv_SubgroupBlockWriteINTEL((__attribute__((ocl_global)) uint32_t *)dst.get(),
                                             *((ocl_t *)&x));
    } else {
      typedef uint16_t ocl_t __attribute__((ext_vector_type(N)));
      __spirv_SubgroupBlockWriteINTEL((__attribute__((ocl_global)) uint16_t *)dst.get(),
                                             *((ocl_t *)&x));
    }
  }

  /* --- synchronization functions --- */
  void barrier(access::fence_space accessSpace =
                   access::fence_space::global_and_local) const {
    uint32_t flags = detail::getSPIRVMemorySemanticsMask(accessSpace);
    __spirv_ControlBarrier(__spv::Scope::Subgroup, __spv::Scope::Subgroup,
                           flags);
  }

protected:
  template <int dimensions> friend class cl::sycl::nd_item;
  sub_group() = default;
};
} // namespace intel
} // namespace sycl
} // namespace cl
#else
#include <CL/sycl/intel/sub_group_host.hpp>
#endif
