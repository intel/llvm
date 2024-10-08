// RUN: %clangxx -fsycl -fsyntax-only -Wno-deprecated-declarations %s
// RUN: %if preview-breaking-changes-supported %{%clangxx -fsycl -fsyntax-only -fpreview-breaking-changes -Wno-deprecated-declarations %s%}

#include <sycl/sycl.hpp>

#include <cstdint>

#define DEFINE_CONSTEXPR_VECTOR(name, type, size)                              \
  constexpr sycl::vec<type, size> name##_##size{0};

#define DEFINE_CONSTEXPR_VECTOR_INIT_NON_ZERO(name, type, size, init)          \
  constexpr sycl::vec<type, size> name##_##size{init};

#define DEFINE_CONSTEXPR_VECTOR_FOR_TYPE(type)                                 \
  DEFINE_CONSTEXPR_VECTOR(type, type, 1)                                       \
  DEFINE_CONSTEXPR_VECTOR(type, type, 2)                                       \
  DEFINE_CONSTEXPR_VECTOR(type, type, 3)                                       \
  DEFINE_CONSTEXPR_VECTOR(type, type, 4)                                       \
  DEFINE_CONSTEXPR_VECTOR(type, type, 8)                                       \
  DEFINE_CONSTEXPR_VECTOR(type, type, 16)

#define DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(type, name)                     \
  DEFINE_CONSTEXPR_VECTOR(name, type, 1)                                       \
  DEFINE_CONSTEXPR_VECTOR(name, type, 2)                                       \
  DEFINE_CONSTEXPR_VECTOR(name, type, 3)                                       \
  DEFINE_CONSTEXPR_VECTOR(name, type, 4)                                       \
  DEFINE_CONSTEXPR_VECTOR(name, type, 8)                                       \
  DEFINE_CONSTEXPR_VECTOR(name, type, 16)

int main() {

  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(sycl::byte, syclbyte)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(sycl::half, half)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE(bool)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE(char)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(signed char, schar)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(unsigned char, uchar)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(short int, short)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(unsigned short int, ushort)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE(int)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(unsigned int, uint)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(long int, long)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(unsigned long int, ulong)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(long long int, longlong)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(unsigned long long int, ulonglong)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE(float)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE(double)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(std::int8_t, int8)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(std::uint8_t, uint8)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(std::int16_t, int16)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(std::uint16_t, uint16)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(std::int32_t, int32)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(std::uint32_t, uint32)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(std::int64_t, int64)
  DEFINE_CONSTEXPR_VECTOR_FOR_TYPE_NAMED(std::uint64_t, uint64)

  DEFINE_CONSTEXPR_VECTOR_INIT_NON_ZERO(stdbyte, std::byte, 1, std::byte{1});
  DEFINE_CONSTEXPR_VECTOR_INIT_NON_ZERO(stdbyte, std::byte, 2, std::byte{1});
  DEFINE_CONSTEXPR_VECTOR_INIT_NON_ZERO(stdbyte, std::byte, 3, std::byte{1});
  DEFINE_CONSTEXPR_VECTOR_INIT_NON_ZERO(stdbyte, std::byte, 4, std::byte{1});
  DEFINE_CONSTEXPR_VECTOR_INIT_NON_ZERO(stdbyte, std::byte, 8, std::byte{1});
  DEFINE_CONSTEXPR_VECTOR_INIT_NON_ZERO(stdbyte, std::byte, 16, std::byte{1});

  return 0;
}
