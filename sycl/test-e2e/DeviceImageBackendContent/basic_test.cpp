// RUN: %{build} -fsyntax-only -DTEST_API_VIOLATION=1 -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#ifdef __cpp_lib_span
#include <span>
#endif
#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

class kernel;

void define_kernel(sycl::queue &q) {
  int data;
  sycl::buffer<int> data_buf(&data, 1);
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor data_acc(data_buf, cgh);
    cgh.parallel_for<class kernel>(
        sycl::nd_range{{1}, {1}},
        [=](sycl::nd_item<> it) { data_acc[0] = 42; });
  });
}

int main() {
  sycl::device d;
  sycl::queue q{d};
  sycl::context ctxt = q.get_context();
  sycl::kernel_id id = sycl::get_kernel_id<kernel>();
  auto bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctxt, {id});
  assert(!bundle.empty());
  sycl::backend backend;
  std::vector<std::byte> bytes;
#ifdef __cpp_lib_span
  std::span<std::byte> bytes_view;
#endif
  for (const auto &img : bundle) {
    if (img.has_kernel(id, d)) {
      // Check that all 3 functions of the api compile.
      // Furthermore, check that the backend corresponds to the backend of the
      // bundle Check that the view of the content is indeed equal to the
      // content.
      backend = img.ext_oneapi_get_backend();
      assert(backend == bundle.get_backend());
      bytes = img.ext_oneapi_get_backend_content();
#ifdef __cpp_lib_span
      bytes_view = img.ext_oneapi_get_backend_content_view();
      assert(bytes_view.size() == bytes.size());
      for (size_t i = 0; i < bytes.size(); ++i) {
        assert(bytes[i] == bytes_view[i]);
      }
#endif
    }
  }

#ifdef TEST_API_VIOLATION
  // Check that the ext_oneapi_get_backend_content and the
  // ext_oneapi_get_backend_content_view of the content functions are only
  // available
  // when the image is in the executable state.

  auto input_bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(ctxt, {id});
  // expected-error@+1 {{no matching member function for call to 'ext_oneapi_get_backend_content'}}
  bytes = (*input_bundle.begin()).ext_oneapi_get_backend_content();
#ifdef _cpp_lib_span
  // expected-error@+1 {{no matching member function for call to 'ext_oneapi_get_backend_content_view'}}
  bytes_view = (*input_bundle.begin()).ext_oneapi_get_backend_content_view();
#endif // __cpp_lib_span
#endif // TEST_API_VIOLATION
  return 0;
}
