// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

// Compile-time tests for various annotated USM allocation functions

#include <sycl/sycl.hpp>

#include <complex>

#include "fake_properties.hpp"

// clang-format on

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;
using alloc = sycl::usm::alloc;

namespace sycl {
namespace ext::oneapi::experimental {
namespace detail {

// Make runtime property `foo` and `foz` valid for this test
template <> struct IsRuntimePropertyValid<foo_key> : std::true_type {};
template <> struct IsRuntimePropertyValid<foz_key> : std::true_type {};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace sycl

// Single test instance consisting of (i) calling malloc function `f` (ii) type
// check on returned annotated_ptr (iii) calling free
#define TEST_VOID(f, args...)                                                  \
  {                                                                            \
    auto ap = f(args);                                                         \
    static_assert(                                                             \
        std::is_same_v<decltype(ap), annotated_ptr<void, decltype(OutP)>>);    \
    free(ap, q);                                                               \
  }

#define TEST_T(f, args...)                                                     \
  {                                                                            \
    auto ap = f(args);                                                         \
    static_assert(                                                             \
        std::is_same_v<decltype(ap), annotated_ptr<T, decltype(OutP)>>);       \
    free(ap, q);                                                               \
  }

// Test all the possible use cases for a single allocation API, including:
// 1. specify input properties `InP1` as argument. The template parameter `input
// properties` and `output properties` are automatically inferred. This is the
// most common use case
// 2. specify input properties `InP1` in the template parameters. The template
// parameter `output properties` is automatically inferred. The property list
// argment can be omitted
// 3. fully specify all the template parameters: `input properties` and `output
// properties`. The property list argment can be omitted 4/5/6. Repeat case 1-3
// but with element type T
#define TEST_GROUP(func_name, args...)                                         \
  TEST_VOID((func_name), args, InP1);                                          \
  TEST_VOID((func_name<decltype(InP1)>),                                       \
            args); /*the property list argument is omitted*/                   \
  TEST_VOID((func_name<decltype(InP1)>), args, InP1);                          \
  TEST_VOID((func_name<decltype(InP1), decltype(OutP)>), args, InP1);          \
  TEST_VOID((func_name<decltype(InP1), decltype(OutP)>), args);                \
  TEST_T((func_name<T>), args, InP1);                                          \
  TEST_T((func_name<T, decltype(InP1)>), args);                                \
  TEST_T((func_name<T, decltype(InP1)>), args, InP1);                          \
  TEST_T((func_name<T, decltype(InP1), decltype(OutP)>), args, InP1);          \
  TEST_T((func_name<T, decltype(InP1), decltype(OutP)>), args);

// Test all the possible use cases for a single allocation API where runtime
// property is specified. Note that when runtime property exists, the property
// list argument cannot be omitted
#define TEST_GROUP_WITH_RUNTIME_PROPERTY(func_name, args...)                   \
  TEST_VOID((func_name), args, InP2);                                          \
  TEST_VOID((func_name<decltype(InP2)>), args, InP2);                          \
  TEST_VOID((func_name<decltype(InP2), decltype(OutP)>), args, InP2);          \
  TEST_T((func_name<T>), args, InP2);                                          \
  TEST_T((func_name<T, decltype(InP2)>), args, InP2);                          \
  TEST_T((func_name<T, decltype(InP2), decltype(OutP)>), args, InP2);

constexpr int N = 10;

template <typename T> void testAlloc() {
  sycl::queue q;
  const sycl::context &Ctx = q.get_context();
  auto Dev = q.get_device();

  // Test device allocation
  {
    // Given a property list, all compile-time properties in it appear on
    // the returned annotated_ptr, and runtime properties do not appear on the
    // returned annotated_ptr (e.g. `foo`, `foz`)
    properties InP1{conduit, buffer_location<5>};
    properties InP2{conduit, buffer_location<5>, foo{foo_enum::a}, foz{0.1, 1}};
    properties OutP{conduit, buffer_location<5>, usm_kind_device};

    TEST_GROUP(malloc_device_annotated, N, q);
    TEST_GROUP(malloc_device_annotated, N, Dev, Ctx);
    TEST_GROUP(aligned_alloc_device_annotated, 1, N, q);
    TEST_GROUP(aligned_alloc_device_annotated, 1, N, Dev, Ctx);

    TEST_GROUP_WITH_RUNTIME_PROPERTY(malloc_device_annotated, N, q);
    TEST_GROUP_WITH_RUNTIME_PROPERTY(malloc_device_annotated, N, Dev, Ctx);
    TEST_GROUP_WITH_RUNTIME_PROPERTY(aligned_alloc_device_annotated, 1, N, q);
    TEST_GROUP_WITH_RUNTIME_PROPERTY(aligned_alloc_device_annotated, 1, N, Dev,
                                     Ctx);
  }

  // Test host allocation
  {
    properties InP1{};
    properties InP2{foo{foo_enum::a}, foz{1.0, 1}};
    properties OutP{usm_kind_host};

    TEST_GROUP(malloc_host_annotated, N, q);
    TEST_GROUP(malloc_host_annotated, N, Ctx);
    TEST_GROUP(aligned_alloc_host_annotated, 1, N, q);
    TEST_GROUP(aligned_alloc_host_annotated, 1, N, Ctx);

    TEST_GROUP_WITH_RUNTIME_PROPERTY(malloc_host_annotated, N, q);
    TEST_GROUP_WITH_RUNTIME_PROPERTY(malloc_host_annotated, N, Ctx);
    TEST_GROUP_WITH_RUNTIME_PROPERTY(aligned_alloc_host_annotated, 1, N, q);
    TEST_GROUP_WITH_RUNTIME_PROPERTY(aligned_alloc_host_annotated, 1, N, Ctx);
  }

  // Test shared allocation
  {
    properties InP1{conduit, buffer_location<5>};
    properties InP2{conduit, buffer_location<5>, foo{foo_enum::a}, foz{0.1, 0}};
    properties OutP{conduit, buffer_location<5>, usm_kind_shared};

    TEST_GROUP(malloc_shared_annotated, N, q);
    TEST_GROUP(malloc_shared_annotated, N, Dev, Ctx);
    TEST_GROUP(aligned_alloc_shared_annotated, 1, N, q);
    TEST_GROUP(aligned_alloc_shared_annotated, 1, N, Dev, Ctx);

    TEST_GROUP_WITH_RUNTIME_PROPERTY(malloc_shared_annotated, N, q);
    TEST_GROUP_WITH_RUNTIME_PROPERTY(malloc_shared_annotated, N, Dev, Ctx);
    TEST_GROUP_WITH_RUNTIME_PROPERTY(aligned_alloc_shared_annotated, 1, N, q);
    TEST_GROUP_WITH_RUNTIME_PROPERTY(aligned_alloc_shared_annotated, 1, N, Dev,
                                     Ctx);
  }

  // Test alloc functions with usm_kind argument and no usm_kind compile-time
  // property, usm_kind does not appear on the returned annotated_ptr
  {
    {
      properties InP1{conduit, buffer_location<5>};
      properties InP2{conduit, buffer_location<5>, foo{foo_enum::a},
                      foz{0.1, 1}};
      properties OutP{conduit, buffer_location<5>};

      TEST_GROUP(malloc_annotated, N, q, alloc::device);
      TEST_GROUP(malloc_annotated, N, Dev, Ctx, alloc::device);
      TEST_GROUP(aligned_alloc_annotated, 1, N, q, alloc::device);
      TEST_GROUP(aligned_alloc_annotated, 1, N, Dev, Ctx, alloc::host);

      TEST_GROUP_WITH_RUNTIME_PROPERTY(malloc_annotated, N, q, alloc::device);
      TEST_GROUP_WITH_RUNTIME_PROPERTY(malloc_annotated, N, Dev, Ctx,
                                       alloc::device);
      TEST_GROUP_WITH_RUNTIME_PROPERTY(aligned_alloc_annotated, 1, N, q,
                                       alloc::device);
      TEST_GROUP_WITH_RUNTIME_PROPERTY(aligned_alloc_annotated, 1, N, Dev, Ctx,
                                       alloc::host);
    }

    // Test alloc functions with empty property list
    {
      properties InP1{};
      properties OutP{};
      TEST_GROUP(malloc_annotated, N, q, alloc::device);
      TEST_GROUP(malloc_annotated, N, Dev, Ctx, alloc::device);
      TEST_GROUP(aligned_alloc_annotated, 1, N, q, alloc::host);
      TEST_GROUP(aligned_alloc_annotated, 1, N, Dev, Ctx, alloc::shared);
    }
  }

  // Test alloc functions where usm_kind property is required in the input
  // property list. usm_kind appears on the returned annotated_ptr
  {
    properties InP2{usm_kind_device, foo{foo_enum::a}, foz{0, 0}};
    properties OutP{usm_kind_device};

    TEST_GROUP_WITH_RUNTIME_PROPERTY(malloc_annotated, N, q);
    TEST_GROUP_WITH_RUNTIME_PROPERTY(malloc_annotated, N, Dev, Ctx);
  }
}

int main() {
  testAlloc<double>();
  testAlloc<std::complex<double>>();
  return 0;
}
