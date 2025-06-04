// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
#include <sycl/detail/core.hpp>
#include <sycl/specialization_id.hpp>
#include <sycl/usm.hpp>

using namespace sycl;
struct base {
  float a;
  char b = 'b';
  int c;
  struct {
    int x;
    int y;
  } d;
};
struct layer1 : base {};
struct layer2 : layer1 {};
struct foo {
  int e = 3;
  long long f[5] = {5};
  struct {
    int value;
  } g[5] = {1, 2};
  char h[5] = {'a', 'b', 'c'};
};
struct alignas(16) layer3 : layer2, foo {};
struct layer4 : layer3 {};
struct scary : layer4 {};

constexpr scary default_scary{};
constexpr scary zero_scary{base{0, 0, 0, {0, 0}}, foo{0, {}, {}, {}}};
constexpr specialization_id<scary> scary_spec_const(default_scary);

int main() {
  queue Q;
  auto *p = malloc_shared<scary>(1, Q);
  new (p) scary{zero_scary};

  Q.single_task([=](kernel_handler h) {
     *p = h.get_specialization_constant<scary_spec_const>();
   }).wait();

  int nfails = 0;
#define EXPECT_EQ(a, b, ...)                                                   \
  if (a != b) {                                                                \
    nfails++;                                                                  \
    std::cout << "FAIL: " << #a << " != " << #b << " (" << (int)a              \
              << " != " << (int)b << ")\n";                                    \
  }

  // base
  EXPECT_EQ(p->a, 0, );
  EXPECT_EQ(p->b, 'b');
  EXPECT_EQ(p->c, 0);
  EXPECT_EQ(p->d.x, 0);
  EXPECT_EQ(p->d.y, 0);

  // foo
  EXPECT_EQ(p->e, 3);

  EXPECT_EQ(p->f[0], 5);
  EXPECT_EQ(p->f[1], 0);
  EXPECT_EQ(p->f[2], 0);
  EXPECT_EQ(p->f[3], 0);
  EXPECT_EQ(p->f[4], 0);

  EXPECT_EQ(p->g[0].value, 1);
  EXPECT_EQ(p->g[1].value, 2);
  EXPECT_EQ(p->g[2].value, 0);
  EXPECT_EQ(p->g[3].value, 0);
  EXPECT_EQ(p->g[4].value, 0);

  EXPECT_EQ(p->h[0], 'a');
  EXPECT_EQ(p->h[1], 'b');
  EXPECT_EQ(p->h[2], 'c');
  EXPECT_EQ(p->h[3], 0);
  EXPECT_EQ(p->h[4], 0);

  if (nfails == 0) {
    std::cout << "PASS\n";
  } else {
    std::cout << "FAIL\n";
  }

  free(p, Q);
  return nfails;
}
