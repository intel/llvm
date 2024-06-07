// Header with helper macros for testing vec binary operators.

#pragma once

#include <sycl/detail/core.hpp>
#include <sycl/types.hpp>

template <typename T>
using rel_t = std::conditional_t<
    sizeof(T) == 1, int8_t,
    std::conditional_t<
        sizeof(T) == 2, int16_t,
        std::conditional_t<sizeof(T) == 4, int32_t,
                           std::conditional_t<sizeof(T) == 8, int64_t, void>>>>;

template <bool IsRelOp, typename T1, int N, typename T2>
bool CheckResult(sycl::vec<T1, N> V, T2 Ref) {
  if constexpr (IsRelOp) {
    // Check that all elements have the same boolean representation as the
    // scalar.
    for (size_t I = 0; I < N; ++I)
      if (static_cast<bool>(V[I]) != static_cast<bool>(Ref))
        return false;
    return true;
  } else {
    // Check that all elements are equal to the scalar.
    for (size_t I = 0; I < N; ++I)
      if (V[I] != Ref)
        return false;
    return true;
  }
}

#define CHECK(Q, C, T, N, IS_RELOP, OP)                                        \
  {                                                                            \
    using VecT = sycl::vec<T, N>;                                              \
    using ResT = sycl::vec<std::conditional_t<IS_RELOP, rel_t<T>, T>, N>;      \
    constexpr T RefVal = 2;                                                    \
    VecT InVec{static_cast<T>(RefVal)};                                        \
    {                                                                          \
      VecT OutVecsDevice[2];                                                   \
      T OutRefsDevice[2];                                                      \
      {                                                                        \
        sycl::buffer<VecT, 1> OutVecsBuff{OutVecsDevice, 2};                   \
        sycl::buffer<T, 1> OutRefsBuff{OutRefsDevice, 2};                      \
        Q.submit([&](sycl::handler &CGH) {                                     \
          sycl::accessor OutVecsAcc{OutVecsBuff, CGH, sycl::read_write};       \
          sycl::accessor OutRefsAcc{OutRefsBuff, CGH, sycl::read_write};       \
          CGH.single_task([=]() {                                              \
            auto OutVec1 = InVec OP RefVal;                                    \
            auto OutVec2 = RefVal OP InVec;                                    \
            static_assert(std::is_same_v<decltype(OutVec1), ResT>);            \
            static_assert(std::is_same_v<decltype(OutVec2), ResT>);            \
            OutVecsAcc[0] = OutVec1;                                           \
            OutVecsAcc[1] = OutVec2;                                           \
            OutRefsAcc[0] = RefVal OP RefVal;                                  \
            OutRefsAcc[1] = RefVal OP RefVal;                                  \
          });                                                                  \
        });                                                                    \
      }                                                                        \
      if (!CheckResult<IS_RELOP>(OutVecsDevice[0], OutRefsDevice[0])) {        \
        std::cout << ("Check of vector " #OP                                   \
                      " scalar from device failed for " #T " and " #N)         \
                  << std::endl;                                                \
        ++C;                                                                   \
      }                                                                        \
      if (!CheckResult<IS_RELOP>(OutVecsDevice[1], OutRefsDevice[1])) {        \
        std::cout << ("Check of scalar " #OP                                   \
                      " vector from device failed for " #T " and " #N)         \
                  << std::endl;                                                \
        ++C;                                                                   \
      }                                                                        \
    }                                                                          \
    {                                                                          \
      auto OutVec1 = InVec OP RefVal;                                          \
      auto OutVec2 = RefVal OP InVec;                                          \
      static_assert(std::is_same_v<decltype(OutVec1), ResT>);                  \
      static_assert(std::is_same_v<decltype(OutVec2), ResT>);                  \
      if (!CheckResult<IS_RELOP>(OutVec1, RefVal OP RefVal)) {                 \
        std::cout << ("Check of vector " #OP                                   \
                      " scalar from host failed for " #T " and " #N)           \
                  << std::endl;                                                \
        ++C;                                                                   \
      }                                                                        \
      if (!CheckResult<IS_RELOP>(OutVec2, RefVal OP RefVal)) {                 \
        std::cout << ("Check of scalar " #OP                                   \
                      " vector from host failed for " #T " and " #N)           \
                  << std::endl;                                                \
        ++C;                                                                   \
      }                                                                        \
    }                                                                          \
  }

#define CHECK_SIZES(Q, C, T, IS_RELOP, OP)                                     \
  CHECK(Q, C, T, 1, IS_RELOP, OP)                                              \
  CHECK(Q, C, T, 2, IS_RELOP, OP)                                              \
  CHECK(Q, C, T, 4, IS_RELOP, OP)                                              \
  CHECK(Q, C, T, 8, IS_RELOP, OP)                                              \
  CHECK(Q, C, T, 16, IS_RELOP, OP)
