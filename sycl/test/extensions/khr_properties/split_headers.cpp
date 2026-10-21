// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -DHEADER=1 %s
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -DHEADER=2 %s
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -DHEADER=3 %s
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -DHEADER=4 %s
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -DHEADER=5 %s
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -DHEADER=6 %s
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -DHEADER=7 %s
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -DHEADER=8 %s
//
// Checks that each header in the sycl_khr_properties header table
// (sec:khr-properties-header-files) provides the APIs listed for it. Each RUN
// line tests one header in isolation. Rows whose APIs take a properties list
// also include <sycl/khr/properties.hpp>, after the header under test.

// expected-no-diagnostics

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#if HEADER == 1
#include <sycl/khr/split_headers/version.hpp>

#if !defined(SYCL_KHR_PROPERTIES) || SYCL_KHR_PROPERTIES != 1
#error "SYCL_KHR_PROPERTIES must be 1"
#endif

#elif HEADER == 2
#include <sycl/khr/properties.hpp>

#include <type_traits>

struct SomeClass {};
using sycl::khr::empty_properties_t;

template <typename List, typename Key, typename = void>
struct can_has_property : std::false_type {};
template <typename List, typename Key>
struct can_has_property<
    List, Key, std::void_t<decltype(List::template has_property<Key>())>>
    : std::true_type {};

static_assert(
    std::is_same_v<empty_properties_t, decltype(sycl::khr::properties{})>);
static_assert(!can_has_property<empty_properties_t, SomeClass>::value);
static_assert(!sycl::khr::is_property_v<SomeClass>);
static_assert(!sycl::khr::is_property<empty_properties_t>::value);
static_assert(!sycl::khr::is_property_key_v<SomeClass>);
static_assert(!sycl::khr::is_property_key<SomeClass>::value);
static_assert(!sycl::khr::is_property_key_compile_time_v<SomeClass>);
static_assert(!sycl::khr::is_property_key_compile_time<SomeClass>::value);
static_assert(!sycl::khr::is_property_for_v<SomeClass, SomeClass>);
static_assert(!sycl::khr::is_property_for<SomeClass, SomeClass>::value);
static_assert(!sycl::khr::is_property_key_for_v<SomeClass, SomeClass>);
static_assert(!sycl::khr::is_property_key_for<SomeClass, SomeClass>::value);
static_assert(sycl::khr::is_property_list_for_v<empty_properties_t, SomeClass>);
static_assert(
    sycl::khr::is_property_list_for<empty_properties_t, SomeClass>::value);

#elif HEADER == 3
#include <sycl/khr/split_headers/context.hpp>

#include <sycl/khr/properties.hpp>

#include <vector>

void test(const sycl::device &Dev, const sycl::platform &Plt,
          const std::vector<sycl::device> &Devs, sycl::async_handler AH) {
  sycl::khr::properties Props{};
  sycl::context C1(Props);
  sycl::context C2(AH, Props);
  sycl::context C3(Dev, Props);
  sycl::context C4(Dev, AH, Props);
  sycl::context C5(Plt, Props);
  sycl::context C6(Plt, AH, Props);
  sycl::context C7(Devs, Props);
  sycl::context C8(Devs, AH, Props);
}

#elif HEADER == 4
#include <sycl/khr/split_headers/queue.hpp>

#include <sycl/khr/properties.hpp>

namespace kp = sycl::khr::property;

static_assert(sycl::khr::is_property_v<kp::enable_profiling>);
static_assert(sycl::khr::is_property_v<kp::in_order>);
static_assert(
    sycl::khr::is_property_key_for_v<kp::key::enable_profiling, sycl::queue>);
static_assert(sycl::khr::is_property_key_for_v<kp::key::in_order, sycl::queue>);

void test(const sycl::device &Dev, const sycl::context &Ctx,
          const sycl::async_handler &AH) {
  auto Sel = sycl::default_selector_v;
  sycl::khr::properties Props{kp::in_order{}, kp::enable_profiling{false}};
  sycl::queue Q1(kp::in_order{});
  sycl::queue Q2(AH, Props);
  sycl::queue Q3(Sel, Props);
  sycl::queue Q4(Sel, AH, Props);
  sycl::queue Q5(Dev, Props);
  sycl::queue Q6(Dev, AH, Props);
  sycl::queue Q7(Ctx, Sel, Props);
  sycl::queue Q8(Ctx, Sel, AH, Props);
  sycl::queue Q9(Ctx, Dev, Props);
  sycl::queue Q10(Ctx, Dev, AH, Props);
  (void)Q1.is_in_order();
  (void)Q1.khr_is_profiling_enabled();
}

#elif HEADER == 5
#include <sycl/khr/split_headers/usm.hpp>

#include <sycl/khr/properties.hpp>

namespace kp = sycl::khr::property;
using sycl::khr::empty_properties_t;

static_assert(
    sycl::khr::is_property_list_for_v<empty_properties_t, kp::tag::usm_device>);
static_assert(
    sycl::khr::is_property_list_for_v<empty_properties_t, kp::tag::usm_host>);
static_assert(
    sycl::khr::is_property_list_for_v<empty_properties_t, kp::tag::usm_shared>);

void test(const sycl::device &Dev, const sycl::context &Ctx,
          const sycl::queue &Q) {
  empty_properties_t P;
  constexpr auto Shared = sycl::usm::alloc::shared;
  sycl::usm_allocator<int, sycl::usm::alloc::host> A1(Ctx, Dev, P);
  sycl::usm_allocator<int, Shared> A2(Q, P);

  (void)sycl::malloc_device(1, Dev, Ctx, P);
  (void)sycl::malloc_device<int>(1, Dev, Ctx, P);
  (void)sycl::malloc_device(1, Q, P);
  (void)sycl::malloc_device<int>(1, Q, P);
  (void)sycl::aligned_alloc_device(4, 1, Dev, Ctx, P);
  (void)sycl::aligned_alloc_device<int>(4, 1, Dev, Ctx, P);
  (void)sycl::aligned_alloc_device(4, 1, Q, P);
  (void)sycl::aligned_alloc_device<int>(4, 1, Q, P);

  (void)sycl::malloc_host(1, Ctx, P);
  (void)sycl::malloc_host<int>(1, Ctx, P);
  (void)sycl::malloc_host(1, Q, P);
  (void)sycl::malloc_host<int>(1, Q, P);
  (void)sycl::aligned_alloc_host(4, 1, Ctx, P);
  (void)sycl::aligned_alloc_host<int>(4, 1, Ctx, P);
  (void)sycl::aligned_alloc_host(4, 1, Q, P);
  (void)sycl::aligned_alloc_host<int>(4, 1, Q, P);

  (void)sycl::malloc_shared(1, Dev, Ctx, P);
  (void)sycl::malloc_shared<int>(1, Dev, Ctx, P);
  (void)sycl::malloc_shared(1, Q, P);
  (void)sycl::malloc_shared<int>(1, Q, P);
  (void)sycl::aligned_alloc_shared(4, 1, Dev, Ctx, P);
  (void)sycl::aligned_alloc_shared<int>(4, 1, Dev, Ctx, P);
  (void)sycl::aligned_alloc_shared(4, 1, Q, P);
  (void)sycl::aligned_alloc_shared<int>(4, 1, Q, P);

  (void)sycl::malloc(1, Dev, Ctx, Shared, P);
  (void)sycl::malloc<int>(1, Dev, Ctx, Shared, P);
  (void)sycl::malloc(1, Q, Shared, P);
  (void)sycl::malloc<int>(1, Q, Shared, P);
  (void)sycl::aligned_alloc(4, 1, Dev, Ctx, Shared, P);
  (void)sycl::aligned_alloc<int>(4, 1, Dev, Ctx, Shared, P);
  (void)sycl::aligned_alloc(4, 1, Q, Shared, P);
  (void)sycl::aligned_alloc<int>(4, 1, Q, Shared, P);
}

#elif HEADER == 6
#include <sycl/khr/split_headers/reduction.hpp>

#include <sycl/khr/properties.hpp>

namespace kp = sycl::khr::property;

static_assert(sycl::khr::is_property_for_v<kp::initialize_to_identity,
                                           kp::tag::reduction>);
static_assert(sycl::khr::is_property_key_for_v<kp::key::initialize_to_identity,
                                               kp::tag::reduction>);

void test(int *Ptr, sycl::span<int, 4> Span) {
  auto Combiner = [](int A, int B) { return A + B; };
  sycl::khr::properties Props{kp::initialize_to_identity{}};
  (void)sycl::reduction(Ptr, Combiner, kp::initialize_to_identity{});
  (void)sycl::reduction(Span, Combiner, Props);
  (void)sycl::reduction(Ptr, 0, Combiner, Props);
  (void)sycl::reduction(Span, 0, Combiner, kp::initialize_to_identity{});
}

#elif HEADER == 7
#include <sycl/khr/split_headers/kernel_bundle.hpp>

#include <sycl/khr/properties.hpp>

#include <vector>

namespace kp = sycl::khr::property;
using sycl::khr::empty_properties_t;

static_assert(
    sycl::khr::is_property_list_for_v<empty_properties_t, kp::tag::compile>);
static_assert(
    sycl::khr::is_property_list_for_v<empty_properties_t, kp::tag::link>);
static_assert(
    sycl::khr::is_property_list_for_v<empty_properties_t, kp::tag::build>);

void test(
    const sycl::kernel_bundle<sycl::bundle_state::input> &In,
    const sycl::kernel_bundle<sycl::bundle_state::object> &Obj,
    const std::vector<sycl::kernel_bundle<sycl::bundle_state::object>> &Objs,
    const std::vector<sycl::device> &Devs) {
  empty_properties_t P;
  (void)sycl::compile(In, Devs, P);
  (void)sycl::compile(In, P);
  (void)sycl::link(Objs, Devs, P);
  (void)sycl::link(Obj, Devs, P);
  (void)sycl::link(Objs, P);
  (void)sycl::link(Obj, P);
  (void)sycl::build(In, Devs, P);
  (void)sycl::build(In, P);
}

#elif HEADER == 8
#include <sycl/khr/split_headers/stream.hpp>

#include <sycl/khr/properties.hpp>

void test(sycl::handler &CGH) {
  sycl::stream S(1024, 256, CGH, sycl::khr::properties{});
}

#else
#error "Unknown HEADER"
#endif
