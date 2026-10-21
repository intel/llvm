// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
//
// Checks that the khr queue constructors reject khr properties that are not
// queue properties, and types that are not exactly properties.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/sycl.hpp>

#include <type_traits>

namespace kp = sycl::khr::property;
using namespace sycl::khr;

struct derived_property : kp::in_order {};
struct two_properties : kp::in_order, kp::enable_profiling {};
struct derived_list : properties<kp::in_order> {};

static_assert(
    !std::is_constructible_v<sycl::queue, kp::initialize_to_identity>);
static_assert(!std::is_constructible_v<sycl::queue, derived_property>);
static_assert(!std::is_constructible_v<sycl::queue, two_properties>);
static_assert(!std::is_constructible_v<sycl::queue, derived_list>);
static_assert(!std::is_constructible_v<sycl::queue, sycl::device,
                                       properties<kp::initialize_to_identity>>);

void foreign_property(const sycl::device &dev, const sycl::context &ctx,
                      sycl::async_handler handler) {
  auto sel = sycl::default_selector_v;
  properties props{kp::in_order{}, kp::initialize_to_identity{}};
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::queue'}}
  sycl::queue q1{kp::initialize_to_identity{}};
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::queue'}}
  sycl::queue q2{props};
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::queue'}}
  sycl::queue q3{handler, kp::initialize_to_identity{}};
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::queue'}}
  sycl::queue q4{sel, props};
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::queue'}}
  sycl::queue q5{dev, handler, props};
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::queue'}}
  sycl::queue q6{ctx, dev, kp::initialize_to_identity{}};
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::queue'}}
  sycl::queue q7{ctx, sel, handler, props};
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::queue'}}
  sycl::queue q8{derived_property{}};
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::queue'}}
  sycl::queue q9{dev, two_properties{}};
}
