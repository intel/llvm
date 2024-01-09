// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//

#include "common.hpp"

struct test {
  int a;
  int *b;
};

template <typename T, typename = void>
struct sfinae_test_unary : std::false_type {};

template <typename T>
struct sfinae_test_unary<T, std::void_t<decltype(~std::declval<T>())>>
    : std::true_type {};

template <typename T, typename = void>
struct sfinae_test_binary : std::false_type {};

template <typename T>
struct sfinae_test_binary<
    T, std::void_t<decltype(std::declval<T>() % std::declval<T>())>>
    : std::true_type {};

// SFIANE test for operator forward
static_assert(!sfinae_test_unary<annotated_ref<float>>::value);
static_assert(sfinae_test_unary<annotated_ref<int>>::value);
static_assert(!sfinae_test_binary<annotated_ref<float>>::value);
static_assert(sfinae_test_binary<annotated_ref<int>>::value);

template <typename T> struct MyStruct;
struct MySecondStruct {
  int data;
  MySecondStruct(){};
  MySecondStruct(int data) : data(data){};

  operator MyStruct<int>() const;
};
template <typename T> struct MyStruct {
  T data;
  MyStruct(){};
  MyStruct(T data) : data(data){};

  template <typename O>
  auto operator+(const O &rhs) -> decltype(std::declval<T>() + rhs) const {
    return data + rhs;
  }

  auto operator+(const MySecondStruct &rhs) const { return data + rhs.data; }

  friend class MySecondStruct;
};

template <typename T> struct MyThirdStruct {
  T data;

  operator int() const { return (int)data; }

  int operator+(const T &rhs) const { return 0; }
  double operator+(T &rhs) const { return 0; }

  T &operator-(T &rhs) const { return rhs; }
  T &&operator-(T &&rhs) const { return std::move(rhs); }

  float operator==(const T &rhs) const { return data == rhs.data ? 3.0 : 1.0; }
};

template <typename T> struct MyFourthStruct {
  T p;

  template <typename T2> MyFourthStruct(const T2 &p_) : p(p_) {}

  template <typename T2> void operator=(const T2 &p_) {}

  int operator+(const int &rhs) const { return 0; }
  int operator+=(const int &rhs) const { return 0; }
};

MySecondStruct::operator MyStruct<int>() const { return MyStruct<int>(0); }

#define BINARY_OP(op)                                                          \
  template <typename T>                                                        \
  MyStruct<T> operator op(const MyStruct<T> &lhs, const MyStruct<T> &rhs) {    \
    return MyStruct<T>(lhs.data op rhs.data);                                  \
  }
BINARY_OP(+)
BINARY_OP(-)
BINARY_OP(/)
BINARY_OP(*)
BINARY_OP(%)
BINARY_OP(<<)
BINARY_OP(>>)

#define LOGICAL_OP(op)                                                         \
  template <typename T>                                                        \
  MyStruct<bool> operator op(const MyStruct<T> &lhs, const MyStruct<T> &rhs) { \
    return MyStruct<bool>(lhs.data op rhs.data);                               \
  }
LOGICAL_OP(&)
LOGICAL_OP(|)
LOGICAL_OP(^)
LOGICAL_OP(==)
LOGICAL_OP(!=)
LOGICAL_OP(<)
LOGICAL_OP(>)
LOGICAL_OP(<=)
LOGICAL_OP(>=)

#define UNARY_OP(op)                                                           \
  template <typename T> MyStruct<T> operator op(const MyStruct<T> &rhs) {      \
    return MyStruct<T>(op rhs.data);                                           \
  }
UNARY_OP(+)
UNARY_OP(-)
UNARY_OP(!)

int main() {
  queue Q;

  auto *a = malloc_shared<int>(8, Q);
  auto a_ptr = annotated_ptr{a};
  for (int i = 0; i < 8; i++)
    a_ptr[i] = i;

  auto *b = malloc_shared<int>(4, Q);
  auto b_ptr = annotated_ptr{b};

  auto *c = malloc_shared<test>(1, Q);
  c->b = malloc_shared<int>(1, Q);
  auto c_ptr = annotated_ptr{c};
  c->a = 0;
  c->b[0] = 0;

  auto *d = malloc_shared<int>(4, Q);
  auto d_ptr = annotated_ptr{d};

  // testing arithmetic overloaded operators
  auto *ee = malloc_shared<MyStruct<int>>(1, Q);
  *ee = MyStruct(5);
  annotated_ptr<MyStruct<int>> e(ee);
  auto *ff = malloc_shared<MyStruct<int>>(1, Q);
  *ff = MyStruct(6);
  annotated_ptr<MyStruct<int>> f(ff);
  auto *gg = malloc_shared<MyStruct<int>>(1, Q);
  *gg = MyStruct(3);
  annotated_ptr<MyStruct<int>> g(gg);
  auto *hh = malloc_shared<MyStruct<int>>(1, Q);
  *hh = MyStruct(2);
  annotated_ptr<MyStruct<int>> h(hh);

  auto *r1 = malloc_shared<MyStruct<int>>(8, Q);
  auto *r2 = malloc_shared<MyStruct<int>>(5, Q);
  auto *r3 = malloc_shared<MyStruct<int>>(5, Q);

  // testing logical/compare overloaded operators
  auto *mm = malloc_shared<MyStruct<bool>>(1, Q);
  *mm = MyStruct(true);
  annotated_ptr<MyStruct<bool>> m(mm);
  auto *nn = malloc_shared<MyStruct<bool>>(1, Q);
  *nn = MyStruct(false);
  annotated_ptr<MyStruct<bool>> n(nn);

  auto *r4 = malloc_shared<MyStruct<bool>>(9, Q);
  auto *r5 = malloc_shared<MyStruct<bool>>(9, Q);
  auto *r6 = malloc_shared<MyStruct<bool>>(9, Q);

  // testing bit shift overloaded operators
  auto *xx = malloc_shared<MyStruct<int>>(1, Q);
  *xx = MyStruct(1);
  annotated_ptr<MyStruct<int>> x(xx);
  auto *yy = malloc_shared<MyStruct<int>>(1, Q);
  *yy = MyStruct(2);
  annotated_ptr<MyStruct<int>> y(yy);
  auto *zz = malloc_shared<MyStruct<int>>(1, Q);
  *zz = MyStruct(4);
  annotated_ptr<MyStruct<int>> z(zz);

  auto *r7 = malloc_shared<MyStruct<int>>(2, Q);
  auto *r8 = malloc_shared<MyStruct<int>>(2, Q);
  auto *r9 = malloc_shared<MyStruct<int>>(2, Q);

  // testing conversion sequence of overloaded operators
  auto *oo = malloc_shared<MyStruct<int>>(1, Q);
  *oo = MyStruct(1);
  annotated_ptr<MyStruct<int>> o(oo);

  auto *pp = malloc_shared<MySecondStruct>(1, Q);
  *pp = MySecondStruct(1);
  annotated_ptr<MySecondStruct> p(pp);

  auto *r10 = malloc_shared<int>(1, Q);

  auto *r11 = malloc_shared<MyFourthStruct<int>>(1, Q);
  annotated_ptr r11_ptr{r11};
  auto r11_add = *r11_ptr + 1;
  *r11_ptr += 1;

  // testing return type of operators
  int o1 = 0;
  float o2 = 1.5;
  double o3 = 3.2;
  auto t1 = *o + o1;
  static_assert(std::is_same_v<decltype(t1), decltype(std::declval<int>() +
                                                      std::declval<int>())>);
  auto t2 = *o + o2;
  static_assert(std::is_same_v<decltype(t2), decltype(std::declval<int>() +
                                                      std::declval<float>())>);
  auto t3 = *o + o3;
  static_assert(std::is_same_v<decltype(t3), decltype(std::declval<int>() +
                                                      std::declval<double>())>);

  MyThirdStruct<int> th;
  annotated_ptr l{&th};
  static_assert(std::is_same_v<int, decltype(*l + 1)>);
  static_assert(std::is_same_v<int, decltype((*l + 1))>);
  static_assert(std::is_same_v<int, decltype(*l + std::forward<int>(1))>);
  int int_var = 1;
  static_assert(std::is_same_v<double, decltype(*l + int_var)>);
  static_assert(std::is_same_v<int, decltype(*l + std::forward<int>(int_var))>);
  static_assert(
      std::is_same_v<double, decltype(*l + std::forward<int &>(int_var))>);
  static_assert(std::is_same_v<float, decltype(*l == 1)>);
  static_assert(std::is_same_v<int &, decltype(*l - std::declval<int &>())>);
  static_assert(std::is_same_v<int &&, decltype(*l - std::declval<int &&>())>);

  // testing volatile and const volatile
  auto *e1 = malloc_shared<int>(1, Q);
  *e1 = 0;
  volatile int *e1_vol = e1;
  auto e1_ptr = annotated_ptr{e1_vol};

  auto *e2 = malloc_shared<int>(1, Q);
  *e2 = 5;
  const volatile int *e2_vol = e2;
  auto e2_ptr = annotated_ptr{e2_vol};

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

     e1_ptr[0] = e2_ptr[0];

     r1[0] = *e + *h;
     r1[1] = *e - *g;
     r1[2] = *g * *h;
     r1[3] = *f / *h;
     r1[4] = *e % *g;
     r1[5] = +*e;
     r1[6] = -*e;
     r1[7] = !*e;

     r2[0] = *e + MyStruct(3);
     r2[1] = *f - MyStruct(5);
     r2[2] = *g * MyStruct(2);
     r2[3] = *f / MyStruct(3);
     r2[4] = *f % MyStruct(4);

     r3[0] = MyStruct(3) + *e;
     r3[1] = MyStruct(7) - *f;
     r3[2] = MyStruct(2) * *g;
     r3[3] = MyStruct(9) / *g;
     r3[4] = MyStruct(9) % *f;

     r4[0] = *m & *n;
     r4[1] = *m | *n;
     r4[2] = *m ^ *n;
     r4[3] = *e == *f;
     r4[4] = *e != *f;
     r4[5] = *e < *f;
     r4[6] = *e > *f;
     r4[7] = *e <= *f;
     r4[8] = *e >= *f;

     r5[0] = *m & MyStruct(true);
     r5[1] = *n | MyStruct(false);
     r5[2] = *m ^ MyStruct(true);
     r5[3] = *e == MyStruct(1);
     r5[4] = *e != MyStruct(1);
     r5[5] = *e < MyStruct(1);
     r5[6] = *e > MyStruct(1);
     r5[7] = *e <= MyStruct(1);
     r5[8] = *e >= MyStruct(1);

     r6[0] = MyStruct(false) & *n;
     r6[1] = MyStruct(false) | *m;
     r6[2] = MyStruct(true) ^ *n;
     r6[3] = MyStruct(1) == *e;
     r6[4] = MyStruct(1) != *e;
     r6[5] = MyStruct(1) < *e;
     r6[6] = MyStruct(1) > *e;
     r6[7] = MyStruct(1) <= *e;
     r6[8] = MyStruct(1) >= *e;

     r7[0] = *z >> *y;
     r7[1] = *y << *x;

     r8[0] = *z >> MyStruct(1);
     r8[1] = *x << MyStruct(3);

     r9[0] = MyStruct(8) >> *y;
     r9[1] = MyStruct(2) << *x;

     r10[0] = *o + *p;
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

  assert(r1[0].data == 7 && "r1[0] value does not match.");
  assert(r1[1].data == 2 && "r1[1] value does not match.");
  assert(r1[2].data == 6 && "r1[2] value does not match.");
  assert(r1[3].data == 3 && "r1[3] value does not match.");
  assert(r1[4].data == 2 && "r1[4] value does not match.");
  assert(r1[5].data == 5 && "r1[5] value does not match.");
  assert(r1[6].data == -5 && "r1[6] value does not match.");
  assert(r1[7].data == 0 && "r1[7] value does not match.");

  assert(r2[0].data == 8 && "r2[0] value does not match.");
  assert(r2[1].data == 1 && "r2[1] value does not match.");
  assert(r2[2].data == 6 && "r2[2] value does not match.");
  assert(r2[3].data == 2 && "r2[3] value does not match.");
  assert(r2[4].data == 2 && "r2[4] value does not match.");

  assert(r3[0].data == 8 && "r3[0] value does not match.");
  assert(r3[1].data == 1 && "r3[1] value does not match.");
  assert(r3[2].data == 6 && "r3[2] value does not match.");
  assert(r3[3].data == 3 && "r3[3] value does not match.");
  assert(r3[4].data == 3 && "r3[4] value does not match.");

  assert(r4[0].data == false && "r4[0] value does not match.");
  assert(r4[1].data == true && "r4[1] value does not match.");
  assert(r4[2].data == true && "r4[2] value does not match.");
  assert(r4[3].data == false && "r4[3] value does not match.");
  assert(r4[4].data == true && "r4[4] value does not match.");
  assert(r4[5].data == true && "r4[5] value does not match.");
  assert(r4[6].data == false && "r4[6] value does not match.");
  assert(r4[7].data == true && "r4[7] value does not match.");
  assert(r4[8].data == false && "r4[8] value does not match.");

  assert(r5[0].data == true && "r5[0] value does not match.");
  assert(r5[1].data == false && "r5[1] value does not match.");
  assert(r5[2].data == false && "r5[2] value does not match.");
  assert(r5[3].data == false && "r5[3] value does not match.");
  assert(r5[4].data == true && "r5[4] value does not match.");
  assert(r5[5].data == false && "r5[5] value does not match.");
  assert(r5[6].data == true && "r5[6] value does not match.");
  assert(r5[7].data == false && "r5[7] value does not match.");
  assert(r5[8].data == true && "r5[8] value does not match.");

  assert(r6[0].data == false && "r6[0] value does not match.");
  assert(r6[1].data == true && "r6[1] value does not match.");
  assert(r6[2].data == true && "r6[2] value does not match.");
  assert(r6[3].data == false && "r6[3] value does not match.");
  assert(r6[4].data == true && "r6[4] value does not match.");
  assert(r6[5].data == true && "r6[5] value does not match.");
  assert(r6[6].data == false && "r6[6] value does not match.");
  assert(r6[7].data == true && "r6[7] value does not match.");
  assert(r6[8].data == false && "r6[8] value does not match.");

  assert(r7[0].data == 1 && "r7[0] value does not match.");
  assert(r7[1].data == 4 && "r7[1] value does not match.");

  assert(r8[0].data == 2 && "r8[0] value does not match.");
  assert(r8[1].data == 8 && "r8[1] value does not match.");

  assert(r9[0].data == 2 && "r9[0] value does not match.");
  assert(r9[1].data == 4 && "r9[1] value does not match.");

  assert(r10[0] == 2 && "r10[0] value does not match.");

  assert(e1_ptr[0] == 5 && "e_ptr[0] value does not match.");

  free(a, Q);
  free(b, Q);
  free(c->b, Q);
  free(c, Q);
  free(d, Q);
  free(e1, Q);
  free(e2, Q);

  free(e, Q);
  free(f, Q);
  free(g, Q);
  free(m, Q);
  free(n, Q);
  free(o, Q);
  free(p, Q);
  free(x, Q);
  free(y, Q);
  free(z, Q);
  free(r1, Q);
  free(r2, Q);
  free(r3, Q);
  free(r4, Q);
  free(r5, Q);
  free(r6, Q);
  free(r7, Q);
  free(r8, Q);
  free(r9, Q);
  free(r10, Q);

  return 0;
}
