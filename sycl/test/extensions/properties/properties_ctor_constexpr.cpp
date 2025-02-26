// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <sycl/sycl.hpp>

#include "mock_compile_time_properties.hpp"

int main() {
  // Empty properties
  constexpr decltype(sycl::ext::oneapi::experimental::properties{})
      EmptyProps1{};
  constexpr auto EmptyProps2 = sycl::ext::oneapi::experimental::properties{};

  // Compile-time value properties
  constexpr decltype(sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::boo<int, bool>}) CTProps1{
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::boo<int, bool>};
  constexpr auto CTProps2 = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::boo<int, bool>};

  // Runtime value properties
  constexpr decltype(sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::foo(42),
      sycl::ext::oneapi::experimental::foz(3.14, false)}) RTProps1{
      sycl::ext::oneapi::experimental::foo(42),
      sycl::ext::oneapi::experimental::foz(3.14, false)};
  constexpr auto RTProps2 = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::foo(42),
      sycl::ext::oneapi::experimental::foz(3.14, false)};

  // Mixed compile-time and runtime value properties
  constexpr decltype(sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::boo<int, bool>,
      sycl::ext::oneapi::experimental::foo(42),
      sycl::ext::oneapi::experimental::foz(3.14, false)}) MixProps1{
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::boo<int, bool>,
      sycl::ext::oneapi::experimental::foo(42),
      sycl::ext::oneapi::experimental::foz(3.14, false)};
  constexpr auto MixProps2 = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::baz<1>,
      sycl::ext::oneapi::experimental::boo<int, bool>,
      sycl::ext::oneapi::experimental::foo(42),
      sycl::ext::oneapi::experimental::foz(3.14, false)};

  // Constructing using a subset of properties in the type.
  constexpr decltype(sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::foo(42),
      sycl::ext::oneapi::experimental::foz(3.14, false)}) SubsetCtorProps1{
      sycl::ext::oneapi::experimental::foo(42),
      sycl::ext::oneapi::experimental::foz(3.14, false)};
  constexpr decltype(sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::foo(42),
      sycl::ext::oneapi::experimental::foz(3.14, false)}) SubsetCtorProps2{
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::foz(3.14, false)};
  constexpr decltype(sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::foo(42),
      sycl::ext::oneapi::experimental::foz(3.14, false)}) SubsetCtorProps3{
      sycl::ext::oneapi::experimental::foz(3.14, false)};
  constexpr decltype(sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::bar,
      sycl::ext::oneapi::experimental::foo(42)}) SubsetCtorProps4{};

  // Runtime value property without constexpr ctors
  // expected-error@+2 {{constexpr variable cannot have non-literal type}}
  constexpr decltype(sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::fir(3.14, false)}) NCRTProps1{
      sycl::ext::oneapi::experimental::fir(3.14, false)};
  // expected-error@+1 {{constexpr variable cannot have non-literal type}}
  constexpr auto NCRTProps2 = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::fir(3.14, false)};
  int RTIntValue = 1;
  // expected-error@+2 {{constexpr variable 'NCRTProps3' must be initialized by a constant expression}}
  constexpr decltype(sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::foo(RTIntValue)}) NCRTProps3{
      sycl::ext::oneapi::experimental::foo(RTIntValue)};
  // expected-error@+1 {{constexpr variable 'NCRTProps4' must be initialized by a constant expressio}}
  constexpr auto NCRTProps4 = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::foo(RTIntValue)};

  return 0;
}
