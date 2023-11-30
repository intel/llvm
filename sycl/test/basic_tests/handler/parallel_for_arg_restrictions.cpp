// RUN: %clangxx -fsycl -fsyntax-only -ferror-limit=0 -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning,error %s
// RUN: %if preview-breaking-changes-supported %{%clangxx -fsycl -fpreview-breaking-changes -fsyntax-only -ferror-limit=0 -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning,error %s -DPREVIEW_BREAKING_CHANGES%}

// Tests the requirements on the first argument in a kernel lambda.
// TODO: Some of the behavior is currently guarded behind the
// -fpreview-breaking-changes option. Remove the use of this option when this
// behavior has been promoted.

#include <sycl/sycl.hpp>

template <int Dims> struct ConvertibleFromItem {
  ConvertibleFromItem(sycl::item<Dims>) {}
};

template <int Dims> struct ConvertibleFromNDItem {
  ConvertibleFromNDItem(sycl::nd_item<Dims>) {}
};

int main() {
  sycl::queue Q;

// TODO: Remove this when the guarded behavior is promoted.
#ifdef PREVIEW_BREAKING_CHANGES
  // ND-range parallel_for with item.
  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{Kernel argument of a sycl::parallel_for with sycl::nd_range must be either sycl::nd_item or be convertible from sycl::nd_item}}
    CGH.parallel_for(sycl::nd_range{sycl::range{1}, sycl::range{1}},
                     [=](sycl::item<1>) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{Kernel argument of a sycl::parallel_for with sycl::nd_range must be either sycl::nd_item or be convertible from sycl::nd_item}}
    CGH.parallel_for(sycl::nd_range{sycl::range{1, 1}, sycl::range{1, 1}},
                     [=](sycl::item<2>) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{Kernel argument of a sycl::parallel_for with sycl::nd_range must be either sycl::nd_item or be convertible from sycl::nd_item}}
    CGH.parallel_for(sycl::nd_range{sycl::range{1, 1, 1}, sycl::range{1, 1, 1}},
                     [=](sycl::item<3>) {});
  });

  // ND-range parallel_for with id.
  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{Kernel argument of a sycl::parallel_for with sycl::nd_range must be either sycl::nd_item or be convertible from sycl::nd_item}}
    CGH.parallel_for(sycl::nd_range{sycl::range{1}, sycl::range{1}},
                     [=](sycl::id<1>) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{Kernel argument of a sycl::parallel_for with sycl::nd_range must be either sycl::nd_item or be convertible from sycl::nd_item}}
    CGH.parallel_for(sycl::nd_range{sycl::range{1, 1}, sycl::range{1, 1}},
                     [=](sycl::id<2>) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{Kernel argument of a sycl::parallel_for with sycl::nd_range must be either sycl::nd_item or be convertible from sycl::nd_item}}
    CGH.parallel_for(sycl::nd_range{sycl::range{1, 1, 1}, sycl::range{1, 1, 1}},
                     [=](sycl::id<3>) {});
  });

  // ND-range parallel_for with argument that is convertible from item.
  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{Kernel argument of a sycl::parallel_for with sycl::nd_range must be either sycl::nd_item or be convertible from sycl::nd_item}}
    CGH.parallel_for(sycl::nd_range{sycl::range{1}, sycl::range{1}},
                     [=](ConvertibleFromItem<1>) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{Kernel argument of a sycl::parallel_for with sycl::nd_range must be either sycl::nd_item or be convertible from sycl::nd_item}}
    CGH.parallel_for(sycl::nd_range{sycl::range{1, 1}, sycl::range{1, 1}},
                     [=](ConvertibleFromItem<2>) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{Kernel argument of a sycl::parallel_for with sycl::nd_range must be either sycl::nd_item or be convertible from sycl::nd_item}}
    CGH.parallel_for(sycl::nd_range{sycl::range{1, 1, 1}, sycl::range{1, 1, 1}},
                     [=](ConvertibleFromItem<3>) {});
  });

  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{sycl::parallel_for(sycl::range) kernel must have the first argument of sycl::item type, or of a type which is implicitly convertible from sycl::item}}
    CGH.parallel_for(sycl::range{1, 1, 1}, [=](float f, sycl::item<3> it) {});
  });

  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{sycl::parallel_for(sycl::range) kernel must have the first argument of sycl::item type, or of a type which is implicitly convertible from sycl::item}}
    CGH.parallel_for(sycl::range{1, 1},
                     [=](kernel_handler kh, sycl::item<3> it) {});
  });

  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{SYCL kernel lambda/functor has an unexpected signature, it should be invocable with sycl::item and optionally sycl::kernel_handler}}
    CGH.parallel_for(sycl::range{1}, [=](sycl::item<1> it, sycl::item<1> it,
                                         kernel_handler kh) {});
  });
#endif // PREVIEW_BREAKING_CHANGES

  // Range parallel_for with nd_item.
  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{Kernel argument cannot have a sycl::nd_item type in sycl::parallel_for with sycl::range}}
    CGH.parallel_for(sycl::range{1}, [=](sycl::nd_item<1>) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{Kernel argument cannot have a sycl::nd_item type in sycl::parallel_for with sycl::range}}
    CGH.parallel_for(sycl::range{1, 1}, [=](sycl::nd_item<2>) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{Kernel argument cannot have a sycl::nd_item type in sycl::parallel_for with sycl::range}}
    CGH.parallel_for(sycl::range{1, 1, 1}, [=](sycl::nd_item<3>) {});
  });

  // Range parallel_for with argument that is convertible from nd_item.
  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{Kernel argument cannot have a sycl::nd_item type in sycl::parallel_for with sycl::range}}
    CGH.parallel_for(sycl::range{1}, [=](ConvertibleFromNDItem<1>) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{Kernel argument cannot have a sycl::nd_item type in sycl::parallel_for with sycl::range}}
    CGH.parallel_for(sycl::range{1, 1}, [=](ConvertibleFromNDItem<2>) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    // expected-error@sycl/handler.hpp:* {{Kernel argument cannot have a sycl::nd_item type in sycl::parallel_for with sycl::range}}
    CGH.parallel_for(sycl::range{1, 1, 1}, [=](ConvertibleFromNDItem<3>) {});
  });
}
