// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out

#include <sycl/sycl.hpp>

class vector_add;

int main() {
  constexpr size_t dataSize = 1024;

  float a[dataSize], r[dataSize];
  for (int i = 0; i < dataSize; ++i) {
    a[i] = static_cast<float>(i);
    r[i] = 0.0f;
  }

  try {
    auto defaultQueue = sycl::queue{};

    auto bufA = sycl::buffer{a, sycl::range{dataSize}};
    auto bufR = sycl::buffer{r, sycl::range{dataSize}};

    defaultQueue
        .submit([&](sycl::handler &cgh) {
          sycl::accessor accA{bufA, cgh, sycl::read_write};
          sycl::accessor accR{bufR, cgh, sycl::write_only};
          sycl::accessor<const float, 1, sycl::access::mode::read,
                         sycl::target::device>
              accB(accA);
          sycl::accessor<float, 1, sycl::access::mode::read,
                         sycl::target::device>
              accC(accA);

          cgh.parallel_for<vector_add>(
              sycl::range{dataSize},
              [=](sycl::id<1> idx) { accR[idx] = accB[idx] + accC[idx]; });
        })
        .wait();

    defaultQueue.throw_asynchronous();
  } catch (const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  for (int i = 0; i < dataSize; ++i) {
    assert(r[i] == static_cast<float>(i) * 2.0f);
  }
}
