// RUN: %clangxx -fsycl -ferror-limit=0 -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=warning,note %s

#include <sycl/sycl.hpp>

template <size_t... Is> struct KernelFunctorWithWGSize {
  void operator()() const {}
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::work_group_size<Is...>};
  }
};

template <size_t... Is> struct KernelFunctorWithWGSizeHint {
  void operator()() const {}
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::work_group_size_hint<Is...>};
  }
};

template <uint32_t I> struct KernelFunctorWithSGSize {
  void operator()() const {}
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<I>};
  }
};

void check_work_group_size() {
  // expected-error@+1 {{too few template arguments for variable template 'work_group_size'}}
  auto WGSize0 = sycl::ext::oneapi::experimental::work_group_size<>;

  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size<0>' must be initialized by a constant expression}}
  auto WGSize1 = sycl::ext::oneapi::experimental::work_group_size<0>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size<0, 0>' must be initialized by a constant expression}}
  auto WGSize2 = sycl::ext::oneapi::experimental::work_group_size<0, 0>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size<1, 0>' must be initialized by a constant expression}}
  auto WGSize3 = sycl::ext::oneapi::experimental::work_group_size<1, 0>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size<0, 1>' must be initialized by a constant expression}}
  auto WGSize4 = sycl::ext::oneapi::experimental::work_group_size<0, 1>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size<0, 0, 0>' must be initialized by a constant expression}}
  auto WGSize5 = sycl::ext::oneapi::experimental::work_group_size<0, 0, 0>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size<1, 0, 0>' must be initialized by a constant expression}}
  auto WGSize6 = sycl::ext::oneapi::experimental::work_group_size<1, 0, 0>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size<0, 1, 0>' must be initialized by a constant expression}}
  auto WGSize7 = sycl::ext::oneapi::experimental::work_group_size<0, 1, 0>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size<0, 0, 1>' must be initialized by a constant expression}}
  auto WGSize8 = sycl::ext::oneapi::experimental::work_group_size<0, 0, 1>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size<1, 1, 0>' must be initialized by a constant expression}}
  auto WGSize9 = sycl::ext::oneapi::experimental::work_group_size<1, 1, 0>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size<0, 1, 1>' must be initialized by a constant expression}}
  auto WGSize10 = sycl::ext::oneapi::experimental::work_group_size<0, 1, 1>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size<1, 0, 1>' must be initialized by a constant expression}}
  auto WGSize11 = sycl::ext::oneapi::experimental::work_group_size<1, 0, 1>;

  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size property currently only supports up to three values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size<1, 1, 1, 1>' must be initialized by a constant expression}}
  auto WGSize12 = sycl::ext::oneapi::experimental::work_group_size<1, 1, 1, 1>;

  sycl::queue Q;

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement {{.+}}: Template type is not a property list.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size<1>},
      KernelFunctorWithWGSize<2>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size<1, 1>},
      KernelFunctorWithWGSize<1, 2>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size<1, 1>},
      KernelFunctorWithWGSize<2, 1>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size<1, 1>},
      KernelFunctorWithWGSize<2, 2>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size<1, 1, 1>},
      KernelFunctorWithWGSize<1, 1, 2>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size<1, 1, 1>},
      KernelFunctorWithWGSize<1, 2, 1>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size<1, 1, 1>},
      KernelFunctorWithWGSize<2, 1, 1>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size<1, 1, 1>},
      KernelFunctorWithWGSize<1, 2, 2>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size<1, 1, 1>},
      KernelFunctorWithWGSize<2, 2, 1>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size<1, 1, 1>},
      KernelFunctorWithWGSize<2, 1, 2>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size<1, 1, 1>},
      KernelFunctorWithWGSize<2, 2, 2>{});
}

void check_work_group_size_hint() {
  // expected-error@+1 {{too few template arguments for variable template 'work_group_size_hint'}}
  auto WGSize0 = sycl::ext::oneapi::experimental::work_group_size_hint<>;

  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size_hint property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size_hint<0>' must be initialized by a constant expression}}
  auto WGSize1 = sycl::ext::oneapi::experimental::work_group_size_hint<0>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size_hint property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size_hint<0, 0>' must be initialized by a constant expression}}
  auto WGSize2 = sycl::ext::oneapi::experimental::work_group_size_hint<0, 0>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size_hint property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size_hint<1, 0>' must be initialized by a constant expression}}
  auto WGSize3 = sycl::ext::oneapi::experimental::work_group_size_hint<1, 0>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size_hint property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size_hint<0, 1>' must be initialized by a constant expression}}
  auto WGSize4 = sycl::ext::oneapi::experimental::work_group_size_hint<0, 1>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size_hint property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size_hint<0, 0, 0>' must be initialized by a constant expression}}
  auto WGSize5 = sycl::ext::oneapi::experimental::work_group_size_hint<0, 0, 0>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size_hint property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size_hint<1, 0, 0>' must be initialized by a constant expression}}
  auto WGSize6 = sycl::ext::oneapi::experimental::work_group_size_hint<1, 0, 0>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size_hint property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size_hint<0, 1, 0>' must be initialized by a constant expression}}
  auto WGSize7 = sycl::ext::oneapi::experimental::work_group_size_hint<0, 1, 0>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size_hint property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size_hint<0, 0, 1>' must be initialized by a constant expression}}
  auto WGSize8 = sycl::ext::oneapi::experimental::work_group_size_hint<0, 0, 1>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size_hint property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size_hint<1, 1, 0>' must be initialized by a constant expression}}
  auto WGSize9 = sycl::ext::oneapi::experimental::work_group_size_hint<1, 1, 0>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size_hint property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size_hint<0, 1, 1>' must be initialized by a constant expression}}
  auto WGSize10 =
      sycl::ext::oneapi::experimental::work_group_size_hint<0, 1, 1>;
  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size_hint property must only contain non-zero values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size_hint<1, 0, 1>' must be initialized by a constant expression}}
  auto WGSize11 =
      sycl::ext::oneapi::experimental::work_group_size_hint<1, 0, 1>;

  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: work_group_size_hint property currently only supports up to three values.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'work_group_size_hint<1, 1, 1, 1>' must be initialized by a constant expression}}
  auto WGSize12 =
      sycl::ext::oneapi::experimental::work_group_size_hint<1, 1, 1, 1>;

  sycl::queue Q;

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size_hint<1>},
      KernelFunctorWithWGSizeHint<2>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size_hint<1, 1>},
      KernelFunctorWithWGSizeHint<1, 2>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size_hint<1, 1>},
      KernelFunctorWithWGSizeHint<2, 1>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size_hint<1, 1>},
      KernelFunctorWithWGSizeHint<2, 2>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size_hint<1, 1, 1>},
      KernelFunctorWithWGSizeHint<1, 1, 2>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size_hint<1, 1, 1>},
      KernelFunctorWithWGSizeHint<1, 2, 1>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size_hint<1, 1, 1>},
      KernelFunctorWithWGSizeHint<2, 1, 1>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size_hint<1, 1, 1>},
      KernelFunctorWithWGSizeHint<1, 2, 2>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size_hint<1, 1, 1>},
      KernelFunctorWithWGSizeHint<2, 2, 1>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size_hint<1, 1, 1>},
      KernelFunctorWithWGSizeHint<2, 1, 2>{});

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::work_group_size_hint<1, 1, 1>},
      KernelFunctorWithWGSizeHint<2, 2, 2>{});
}

void check_sub_group_size() {
  // expected-error@+1 {{too few template arguments for variable template 'sub_group_size'}}
  auto WGSize0 = sycl::ext::oneapi::experimental::sub_group_size<>;

  // expected-error-re@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: sub_group_size_key property must contain a non-zero value.}}
  // expected-error@sycl/ext/oneapi/kernel_properties/properties.hpp:* {{constexpr variable 'sub_group_size<0>' must be initialized by a constant expression}}
  auto WGSize1 = sycl::ext::oneapi::experimental::sub_group_size<0>;

  sycl::queue Q;

  // expected-error-re@sycl/ext/oneapi/properties/properties.hpp:* {{static assertion failed due to requirement {{.+}}: Failed to merge property lists due to conflicting properties.}}
  Q.single_task(
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::sub_group_size<1>},
      KernelFunctorWithSGSize<2>{});
}

int main() {
  check_work_group_size();
  check_work_group_size_hint();
  check_sub_group_size();
  return 0;
}
