// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
#include <sycl/detail/core.hpp>
#include <sycl/specialization_id.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

constexpr specialization_id<vec<float, 3>> vec_spec_const(1.f, 2.f, 3.f);
constexpr specialization_id<marray<float, 3>> marray_spec_const(1.f, 2.f, 3.f);

int main() {
  queue Q;
  auto *v = malloc_shared<vec<float, 3>>(1, Q);
  auto *m = malloc_shared<marray<float, 3>>(1, Q);
  new (v) vec<float, 3>{0.f, 0.f, 0.f};
  new (m) marray<float, 3>{0.f, 0.f, 0.f};

  Q.single_task([=](kernel_handler h) {
     *v = h.get_specialization_constant<vec_spec_const>();
     *m = h.get_specialization_constant<marray_spec_const>();
   }).wait();

  int nfails = 0;
#define EXPECT_EQ(a, b, ...)                                                   \
  if (a != b) {                                                                \
    nfails++;                                                                  \
    std::cout << "FAIL: " << #a << " != " << #b << " (" << (int)a              \
              << " != " << (int)b << ")\n";                                    \
  }

  // vec
  EXPECT_EQ(v->x(), 1.f);
  EXPECT_EQ(v->y(), 2.f);
  EXPECT_EQ(v->z(), 3.f);

  // marray
  EXPECT_EQ((*m)[0], 1.f);
  EXPECT_EQ((*m)[1], 2.f);
  EXPECT_EQ((*m)[2], 3.f);

  if (nfails == 0) {
    std::cout << "PASS\n";
  } else {
    std::cout << "FAIL\n";
  }

  free(v, Q);
  free(m, Q);
  return nfails;
}
