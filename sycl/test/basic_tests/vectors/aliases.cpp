// RUN: %clangxx -fsycl -fsyntax-only -fsycl-device-only %s

#include <sycl/sycl.hpp>

#include <cstdint>
#include <type_traits>

#define CHECK_ALIAS(type, storage_type, elems)                                 \
  static_assert(                                                               \
      std::is_same_v<sycl::type##elems, sycl::vec<storage_type, elems>>);

#define CHECK_ALIASES_FOR_VEC_LENGTH(N)                                        \
  CHECK_ALIAS(schar, std::int8_t, N)                                           \
  CHECK_ALIAS(longlong, std::int64_t, N)                                       \
  CHECK_ALIAS(ulonglong, std::uint64_t, N)                                     \
  CHECK_ALIAS(char, std::int8_t, N)                                            \
  CHECK_ALIAS(uchar, std::uint8_t, N)                                          \
  CHECK_ALIAS(short, std::int16_t, N)                                          \
  CHECK_ALIAS(ushort, std::uint16_t, N)                                        \
  CHECK_ALIAS(int, std::int32_t, N)                                            \
  CHECK_ALIAS(uint, std::uint32_t, N)                                          \
  CHECK_ALIAS(long, std::int64_t, N)                                           \
  CHECK_ALIAS(ulong, std::uint64_t, N)                                         \
  CHECK_ALIAS(half, sycl::half, N)                                             \
  CHECK_ALIAS(float, float, N)                                                 \
  CHECK_ALIAS(double, double, N)

int main() {

  CHECK_ALIASES_FOR_VEC_LENGTH(2)
  CHECK_ALIASES_FOR_VEC_LENGTH(3)
  CHECK_ALIASES_FOR_VEC_LENGTH(4)
  CHECK_ALIASES_FOR_VEC_LENGTH(8)
  CHECK_ALIASES_FOR_VEC_LENGTH(16)

  return 0;
}
