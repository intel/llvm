#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/annotated_usm/alloc_device.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

namespace sycl_ext = sycl::ext::oneapi::experimental;

template <bool Internalize, typename AnnotatedPtrT>
auto get_pointer(AnnotatedPtrT Ptr) {
  if constexpr (Internalize) {
    return Ptr;
  } else {
    return Ptr.get();
  }
}

template <bool Internalize, typename FirstKernelT = class KernelOne,
          typename SecondKernelT = class KernelTwo, typename PropT>
void test_usm_internalization(PropT AccessScope) {
  constexpr size_t dataSize = 512;
  constexpr size_t numBytes = dataSize * sizeof(int);
  constexpr size_t wg_size = 32;

  std::array<int, dataSize> in1;
  std::array<int, dataSize> in2;
  std::array<int, dataSize> in3;
  std::array<int, dataSize> tmp;
  std::array<int, dataSize> out;
  for (size_t i = 0; i < dataSize; ++i) {
    in1[i] = i;
    in2[i] = i;
    in3[i] = i;
    tmp[i] = -1;
  }

  sycl::queue q;

  sycl_ext::command_graph graph{q.get_context(), q.get_device()};

  int *dIn1 = sycl::malloc_device<int>(dataSize, q);
  int *dIn2 = sycl::malloc_device<int>(dataSize, q);
  int *dIn3 = sycl::malloc_device<int>(dataSize, q);
  int *dOut = sycl::malloc_device<int>(dataSize, q);

  sycl_ext::properties P1{sycl_ext::fusion_internal_memory, AccessScope,
                          sycl_ext::fusion_no_init};
  auto annotatedTmp = sycl_ext::malloc_device_annotated<int>(dataSize, q, P1);

  auto copy_in1 = graph.add(
      [&](sycl::handler &cgh) { cgh.memcpy(dIn1, in1.data(), numBytes); });

  auto copy_in2 = graph.add(
      [&](sycl::handler &cgh) { cgh.memcpy(dIn2, in2.data(), numBytes); });

  auto fill = graph.add([&](sycl::handler &cgh) {
    cgh.memcpy(annotatedTmp.get(), tmp.data(), numBytes);
  });
  auto fill2 = graph.add(
      [&](sycl::handler &cgh) { cgh.memcpy(dOut, tmp.data(), numBytes); });

  auto kernel1 = graph.add(
      [&](sycl::handler &cgh) {
        cgh.parallel_for<FirstKernelT>(nd_range{{dataSize}, {wg_size}},
                                          [=](sycl::nd_item<1> it) {
                                            auto i = it.get_global_id(0);
                                            annotatedTmp[i] = dIn1[i] + dIn2[i];
                                          });
      },
      {sycl_ext::property::node::depends_on(copy_in1, copy_in2, fill, fill2)});

  auto copy_in3 = graph.add(
      [&](sycl::handler &cgh) { cgh.memcpy(dIn3, in3.data(), numBytes); });

  auto tmpPtr = get_pointer<Internalize>(annotatedTmp);

  auto kernel2 = graph.add(
      [&](sycl::handler &cgh) {
        cgh.parallel_for<SecondKernelT>(nd_range{{dataSize}, {wg_size}},
                                          [=](sycl::nd_item<1> it) {
                                            auto i = it.get_global_id(0);
                                            dOut[i] = tmpPtr[i] * dIn3[i];
                                          });
      },
      {sycl_ext::property::node::depends_on(copy_in3, kernel1)});

  auto copy_out = graph.add(
      [&](sycl::handler &cgh) { cgh.memcpy(out.data(), dOut, numBytes); },
      {sycl_ext::property::node::depends_on(kernel2)});

  auto copy_tmp = graph.add(
      [&](sycl::handler &cgh) {
        cgh.memcpy(tmp.data(), annotatedTmp.get(), numBytes);
      },
      {sycl_ext::property::node::depends_on(kernel2)});

  auto exec = graph.finalize({sycl_ext::property::graph::require_fusion{}});

  q.ext_oneapi_graph(exec).wait();

  for (size_t i = 0; i < dataSize; ++i) {
    assert(out[i] == (2 * i * i));
    int refTmp = -1;
    if constexpr (!Internalize) {
      refTmp = static_cast<int>(2 * i);
    }
    assert(tmp[i] == refTmp);
  }

  sycl::free(dIn1, q);
  sycl::free(dIn2, q);
  sycl::free(dIn3, q);
  sycl::free(dOut, q);
  sycl::free(annotatedTmp.get(), q);
}
