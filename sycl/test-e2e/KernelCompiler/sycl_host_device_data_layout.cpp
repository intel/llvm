// RUN: %{build} -o %t.out
// RUN: %if hip %{ env SYCL_JIT_AMDGCN_PTX_TARGET_CPU=%{amd_arch} %} %{run} %t.out

// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20142

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

#include <sycl/aliases.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/half_type.hpp>
#include <sycl/marray.hpp>
#include <sycl/vector.hpp>

#define STRINGIFY(x) #x
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)

// Needs to be duplicated between host/device. @{

// Comma would make preprocessor macro trickier.
using mint3 = sycl::marray<int, 3>;

enum E {
  V0 = 0x12345689,
};
static_assert(sizeof(E) == 4);
enum class ScopedE {
  ScopedV0 = 0x12345689,
};
static_assert(sizeof(ScopedE) == 4);

// }@

namespace syclexp = sycl::ext::oneapi::experimental;
int main() {
  sycl::queue q;
  std::string src = R"""(
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>

#include <sycl/aliases.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/half_type.hpp>
#include <sycl/marray.hpp>
#include <sycl/vector.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

using mint3 = sycl::marray<int, 3>;

enum E {
  V0 = 0x12345689,
};
static_assert(sizeof(E) == 4);
enum class ScopedE {
  ScopedV0 = 0x12345689,
};
static_assert(sizeof(ScopedE) == 4);

extern "C"
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void foo(TYPE *in, TYPE *out, size_t *align_out, size_t *size_out, bool *equal_out) {
  *out = TYPE{INIT};
  *align_out = alignof(TYPE);
  *size_out = sizeof(TYPE);
  auto Equal = [](const auto &lhs, const auto &rhs) {
    using T = std::decay_t<decltype(lhs)>;
    if constexpr (sycl::detail::is_vec_v<T> || sycl::detail::is_marray_v<T>) {
      if (lhs.size() != rhs.size())
        return false;

      for (size_t i = 0; i < lhs.size(); ++i)
        if (lhs[i] != rhs[i])
          return false;

      return true;
    } else {
      return lhs == rhs;
    }
  };
  *equal_out = Equal(*in, *out);
}
)""";
  auto kb_src = syclexp::create_kernel_bundle_from_source(
      q.get_context(), syclexp::source_language::sycl, src);

  auto *align = sycl::malloc_shared<size_t>(1, q);
  auto *size = sycl::malloc_shared<size_t>(1, q);
  auto *equal = sycl::malloc_shared<bool>(1, q);

  auto Test = [&](auto val, auto type_str, auto init_str) {
    using namespace std::literals::string_literals;

    using type = decltype(val);
    auto kb_exe = syclexp::build(
        kb_src,
        syclexp::properties{syclexp::build_options{std::vector<std::string>{
            "-DTYPE="s + type_str, "-DINIT="s + init_str}}});

    sycl::kernel krn = kb_exe.ext_oneapi_get_kernel("foo");
    auto *host = sycl::malloc_shared<type>(1, q);
    *host = val;
    auto *device = sycl::malloc_shared<type>(1, q);

    q.submit([&](sycl::handler &cgh) {
       cgh.set_args(host, device, align, size, equal);
       cgh.single_task(krn);
     }).wait();
    auto Equal = [](const auto &lhs, const auto &rhs) {
      using T = std::decay_t<decltype(lhs)>;
      if constexpr (sycl::detail::is_vec_v<T> || sycl::detail::is_marray_v<T>) {
        if (lhs.size() != rhs.size())
          return false;

        for (size_t i = 0; i < lhs.size(); ++i)
          if (lhs[i] != rhs[i])
            return false;

        return true;
      } else {
        return lhs == rhs;
      }
    };
    assert(Equal(*host, *device));
    assert(*align == alignof(type));
    assert(*size == sizeof(type));
    assert(*equal == true);
    sycl::free(host, q);
    sycl::free(device, q);
  };

#define TEST(TYPE, INIT)                                                       \
  Test(TYPE{INIT}, EXPAND_AND_STRINGIFY(TYPE), EXPAND_AND_STRINGIFY(INIT));
#define TEST2(TYPE, INIT0, INIT1)                                              \
  Test(TYPE{INIT0, INIT1}, EXPAND_AND_STRINGIFY(TYPE),                         \
       EXPAND_AND_STRINGIFY(INIT0) ", " EXPAND_AND_STRINGIFY(INIT1));
#define TEST3(TYPE, INIT0, INIT1, INIT2)                                       \
  Test(TYPE{INIT0, INIT1, INIT2}, EXPAND_AND_STRINGIFY(TYPE),                  \
       EXPAND_AND_STRINGIFY(INIT0) ", " EXPAND_AND_STRINGIFY(                  \
           INIT1) ", " EXPAND_AND_STRINGIFY(INIT2));

  TEST(size_t, 0x1122334455667788)
  TEST(char, 0x12)

  TEST(int8_t, 0x12)
  TEST(int8_t, -0x12)
  TEST(uint8_t, 0x12)

  TEST(int16_t, 0x1234)
  TEST(int16_t, -0x1234)
  TEST(uint16_t, 0x1234)

  TEST(int32_t, 0x12345678)
  TEST(int32_t, -0x12345678)
  TEST(uint32_t, 0x12345678)

  TEST(int64_t, 0x1122334455667788)
  TEST(int64_t, -0x1122334455667788)
  TEST(uint64_t, 0x1122334455667788)

  TEST(size_t, 0x1122334455667788)
  TEST(ptrdiff_t, 0x1122334455667788)

  TEST(float, 42.0f)
  if (q.get_device().has(sycl::aspect::fp64)) {
    TEST(double, 42.0)
  }

  TEST(sycl::half, 42.0f)
  TEST(sycl::ext::oneapi::bfloat16, 42.0f)

  TEST(sycl::range<1>, 0x1122334455667788)
  TEST2(sycl::range<2>, 0x1122334455667788, 0x1223344556677889)
  TEST3(sycl::range<3>, 0x1122334455667788, 0x1223344556677889,
        0x132435465768798A)

  TEST(sycl::id<1>, 0x1122334455667788)
  TEST2(sycl::id<2>, 0x1122334455667788, 0x1223344556677889)
  TEST3(sycl::id<3>, 0x1122334455667788, 0x1223344556677889, 0x132435465768798A)

  // Making these work with macros would be too much work:
  Test(sycl::nd_range<1>{{0x1122334455667788}, {0x1223344556677889}},
       "sycl::nd_range<1>", "{0x1122334455667788}, {0x1223344556677889}");
  Test(sycl::nd_range<2>{{0x1122334455667788, 0x2132435465768798},
                         {0x1223344556677889, 0x2233445586778899}},
       "sycl::nd_range<2>",
       "{0x1122334455667788, 0x2132435465768798}, {0x1223344556677889, "
       "0x2233445586778899}");
  Test(
      sycl::nd_range<3>{
          {0x1122334455667788, 0x2132435465768798, 0x31525364758697A8},
          {0x1223344556677889, 0x2233445586778899, 0x32435465768798A9}},
      "sycl::nd_range<3>",
      "{0x1122334455667788, 0x2132435465768798, 0x31525364758697A8}, "
      "{0x1223344556677889, 0x2233445586778899, 0x32435465768798A9}");

  TEST2(sycl::short2, 0x1234, 0x2345)
  TEST3(sycl::short3, 0x1234, 0x2345, 0x3456)

  TEST3(mint3, 0x1234, 0x2345, 0x3456)

  TEST(E, V0)
  TEST(ScopedE, ScopedE::ScopedV0)

  sycl::free(align, q);
  sycl::free(size, q);
  sycl::free(equal, q);
}
