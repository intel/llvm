// RUN: %clangxx -fsycl -std=c++20 %s -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <type_traits>

int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();
  sycl::buffer<int> buf(sycl::range<1>(1));
  sycl::kernel_id id = sycl::get_kernel_id<class kernel>();
  auto bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctxt, {id});
  assert(!bundle.empty());
  sycl::kernel krn = bundle.get_kernel(id);
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor acc(buf, cgh);
    cgh.single_task<class kernel>(krn, [=]() { acc[0] = 42; });
  });
  sycl::backend backend;
  std::vector<std::byte> bytes;
#ifdef __cpp_lib_span
  std::span<const std::byte> bytes_view;
#endif
  for (const auto &img : bundle) {
    // Check that all 3 functions of the api return correct types and compile.
    // Furthermore, check that the backend corresponds to the backend of the
    // bundle Check that the view of the content is indeed equal to the
    // content.
    static_assert(std::is_same_v<decltype(img.ext_oneapi_get_backend()),
                                 decltype(backend)>);
    static_assert(std::is_same_v<decltype(img.ext_oneapi_get_backend_content()),
                                 decltype(bytes)>);
    backend = img.ext_oneapi_get_backend();
    assert(backend == bundle.get_backend());
    bytes = img.ext_oneapi_get_backend_content();
#ifdef __cpp_lib_span
    static_assert(
        std ::is_same_v<decltype(img.ext_oneapi_get_backend_content_view()),
                        decltype(bytes_view)>);
    bytes_view = img.ext_oneapi_get_backend_content_view();
    assert(bytes_view.size() == bytes.size());
    for (size_t i = 0; i < bytes.size(); ++i) {
      assert(bytes[i] == bytes_view[i]);
    }
#endif
  }
  return 0;
}
