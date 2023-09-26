// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// https://github.com/intel/llvm/issues/11224
// UNSUPPORTED: windows

#include "common.hpp"

struct test {
  int a;
  int *b;
};

int main() {
  queue Q;

  auto *a = malloc_shared<int>(8, Q);
  auto a_ptr = annotated_ptr{a, properties{alignment<8>}};
  for (int i = 0; i < 8; i++)
    a_ptr[i] = i;

  auto *b = malloc_shared<int>(4, Q);
  auto b_ptr = annotated_ptr{b, properties{alignment<16>}};

  auto *c = malloc_shared<test>(1, Q);
  c->b = malloc_shared<int>(1, Q);
  auto c_ptr = annotated_ptr{c, properties{alignment<16>}};
  c->a = 0;
  c->b[0] = 0;

  auto *d = malloc_shared<int>(4, Q);
  auto d_ptr = annotated_ptr{d};
  for (int i = 0; i < 4; i++)
    d_ptr[i] = i;
  Q.single_task([=]() {
     a_ptr[0] += 1;
     a_ptr[0] -= 2;

     a_ptr[1] *= 4;
     a_ptr[1] /= 2;

     a_ptr[2]++;
     ++a_ptr[2];
     a_ptr[3]--;
     --a_ptr[3];

     a_ptr[4] = -a_ptr[4];
     a_ptr[5] = !a_ptr[5];

     b_ptr[0] = a_ptr[6] < a_ptr[7];
     b_ptr[1] = a_ptr[6] == a_ptr[7];
     b_ptr[2] = a_ptr[6] != a_ptr[7];
     b_ptr[3] = 11;

     auto *c = c_ptr.get();
     c->a++;
     c->a += 1;
     *c->b = 5;

     auto func = [=](const int &a, const int b, const int &c) {
       return a + b - c;
     };

     d_ptr[3] = func(d_ptr[0], d_ptr[1], d_ptr[2]);
   }).wait();

  assert(a_ptr[0] == -1 && "a_ptr[0] value does not match.");
  assert(a_ptr[1] == 2 && "a_ptr[1] value does not match.");
  assert(a_ptr[2] == 4 && "a_ptr[2] value does not match.");
  assert(a_ptr[3] == 1 && "a_ptr[3] value does not match.");
  assert(a_ptr[4] == -4 && "a_ptr[4] value does not match.");
  assert(a_ptr[5] == 0 && "a_ptr[5] value does not match.");
  assert(b_ptr[0] == 1 && "b_ptr[0] value does not match.");
  assert(b_ptr[1] == 0 && "b_ptr[1] value does not match.");
  assert(b_ptr[2] == 1 && "b_ptr[2] value does not match.");
  assert(b_ptr[3] == 11 && "b_ptr[3] value does not match.");

  assert(c->a == 2 && "c_ptr[0] value does not match.");
  assert(*c->b == 5 && "c_ptr[0].b value does not match.");

  assert(d_ptr[3] == -1 && "d_ptr[3] value does not match.");

  free(a, Q);
  free(b, Q);
  free(c->b, Q);
  free(c, Q);
  free(d, Q);

  return 0;
}
