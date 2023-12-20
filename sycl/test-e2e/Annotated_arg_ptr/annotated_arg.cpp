// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//

#include "common.hpp"

struct test {
  int a;
  int *b;
};

class device_copyable_class {
  int a;
  device_copyable_class(const device_copyable_class &other) : a(other.a){};
};

template <>
struct is_device_copyable<device_copyable_class> : std::true_type {};

template <typename T> struct MyStruct {
  T data;
  MyStruct(){};
  MyStruct(T data) : data(data){};
};

template <typename T>
MyStruct<T> operator+(const MyStruct<T> &lhs, const MyStruct<T> &rhs) {
  return MyStruct<T>(lhs.data + rhs.data);
}

template <typename T>
MyStruct<T> operator-(const MyStruct<T> &lhs, const MyStruct<T> &rhs) {
  return MyStruct<T>(lhs.data - rhs.data);
}

template <typename T>
MyStruct<T> operator*(const MyStruct<T> &lhs, const MyStruct<T> &rhs) {
  return MyStruct<T>(lhs.data * rhs.data);
}

template <typename T>
MyStruct<T> operator/(const MyStruct<T> &lhs, const MyStruct<T> &rhs) {
  return MyStruct<T>(lhs.data / rhs.data);
}

template <typename T>
MyStruct<T> operator%(const MyStruct<T> &lhs, const MyStruct<T> &rhs) {
  return MyStruct<T>(lhs.data % rhs.data);
}

template <typename T>
MyStruct<T> operator&(const MyStruct<T> &lhs, const MyStruct<T> &rhs) {
  return MyStruct<T>(lhs.data & rhs.data);
}

template <typename T>
MyStruct<T> operator|(const MyStruct<T> &lhs, const MyStruct<T> &rhs) {
  return MyStruct<T>(lhs.data | rhs.data);
}

template <typename T>
MyStruct<T> operator^(const MyStruct<T> &lhs, const MyStruct<T> &rhs) {
  return MyStruct<T>(lhs.data ^ rhs.data);
}

template <typename T>
MyStruct<T> operator>>(const MyStruct<T> &lhs, const MyStruct<T> &rhs) {
  return MyStruct<T>(lhs.data >> rhs.data);
}

template <typename T>
MyStruct<T> operator<<(const MyStruct<T> &lhs, const MyStruct<T> &rhs) {
  return MyStruct<T>(lhs.data << rhs.data);
}

int main() {
  queue Q;

  auto *a = malloc_shared<int>(8, Q);
  auto a_ptr = annotated_arg{a};
  for (int i = 0; i < 8; i++)
    a_ptr[i] = i;

  auto *b = malloc_shared<int>(4, Q);
  auto b_ptr = annotated_arg{b};

  auto *c = malloc_shared<test>(1, Q);
  c->b = malloc_shared<int>(1, Q);
  auto c_ptr = annotated_arg{c};
  c_ptr->a = 0;
  c_ptr->b[0] = 0;

  auto *d = malloc_shared<int>(4, Q);
  auto d_ptr = annotated_arg{d};
  for (int i = 0; i < 4; i++)
    d_ptr[i] = i;

  // testing arithmetic overloaded operators
  annotated_arg<MyStruct<int>> e = MyStruct(5);
  annotated_arg<MyStruct<int>> f = MyStruct(6);
  annotated_arg<MyStruct<int>> g = MyStruct(3);
  annotated_arg<MyStruct<int>> h = MyStruct(2);

  auto *r1 = malloc_shared<MyStruct<int>>(5, Q);
  auto *r2 = malloc_shared<MyStruct<int>>(5, Q);
  auto *r3 = malloc_shared<MyStruct<int>>(5, Q);

  // testing logical overloaded operators
  annotated_arg<MyStruct<bool>> m = MyStruct(true);
  annotated_arg<MyStruct<bool>> n = MyStruct(false);

  auto *r4 = malloc_shared<MyStruct<bool>>(3, Q);
  auto *r5 = malloc_shared<MyStruct<bool>>(3, Q);
  auto *r6 = malloc_shared<MyStruct<bool>>(3, Q);

  // testing bit shift overloaded operators
  annotated_arg<MyStruct<int>> x = MyStruct(1);
  annotated_arg<MyStruct<int>> y = MyStruct(2);
  annotated_arg<MyStruct<int>> z = MyStruct(4);

  auto *r7 = malloc_shared<MyStruct<int>>(2, Q);
  auto *r8 = malloc_shared<MyStruct<int>>(2, Q);
  auto *r9 = malloc_shared<MyStruct<int>>(2, Q);

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

     c_ptr->a++;
     (*c_ptr).a += 1;
     *c_ptr->b = 5;

     auto func = [=](int &a, const int b, const int &c, int *d) {
       *d = a + b - c;
     };

     func(d_ptr[0], d_ptr[1], d_ptr[2], &d_ptr[3]);

     r1[0] = e + h;
     r1[1] = e - g;
     r1[2] = g * h;
     r1[3] = f / h;
     r1[4] = e % g;

     r2[0] = e + MyStruct(3);
     r2[1] = f - MyStruct(5);
     r2[2] = g * MyStruct(2);
     r2[3] = f / MyStruct(3);
     r2[4] = f % MyStruct(4);

     r3[0] = MyStruct(3) + e;
     r3[1] = MyStruct(7) - f;
     r3[2] = MyStruct(2) * g;
     r3[3] = MyStruct(9) / g;
     r3[4] = MyStruct(9) % f;

     r4[0] = m & n;
     r4[1] = m | n;
     r4[2] = m ^ n;

     r5[0] = m & MyStruct(true);
     r5[1] = n | MyStruct(false);
     r5[2] = m ^ MyStruct(true);

     r6[0] = MyStruct(false) & n;
     r6[1] = MyStruct(false) | m;
     r6[2] = MyStruct(true) ^ n;

     r7[0] = z >> y;
     r7[1] = y << x;

     r8[0] = z >> MyStruct(1);
     r8[1] = x << MyStruct(3);

     r9[0] = MyStruct(8) >> y;
     r9[1] = MyStruct(2) << x;
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

  assert(c_ptr->a == 2 && "c_ptr[0] value does not match.");
  assert(*c_ptr->b == 5 && "c_ptr[0].b value does not match.");

  assert(d_ptr[3] == -1 && "d_ptr[3] value does not match.");

  assert(r1[0].data == 7 && "r1[0] value does not match.");
  assert(r1[1].data == 2 && "r1[1] value does not match.");
  assert(r1[2].data == 6 && "r1[2] value does not match.");
  assert(r1[3].data == 3 && "r1[3] value does not match.");
  assert(r1[4].data == 2 && "r1[4] value does not match.");

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

  assert(r5[0].data == true && "r5[0] value does not match.");
  assert(r5[1].data == false && "r5[1] value does not match.");
  assert(r5[2].data == false && "r5[2] value does not match.");

  assert(r6[0].data == false && "r6[0] value does not match.");
  assert(r6[1].data == true && "r6[1] value does not match.");
  assert(r6[2].data == true && "r6[2] value does not match.");

  assert(r7[0].data == 1 && "r7[0] value does not match.");
  assert(r7[1].data == 4 && "r7[1] value does not match.");

  assert(r8[0].data == 2 && "r8[0] value does not match.");
  assert(r8[1].data == 8 && "r8[1] value does not match.");

  assert(r9[0].data == 2 && "r9[0] value does not match.");
  assert(r9[1].data == 4 && "r9[1] value does not match.");

  assert(!std::is_trivially_copyable<device_copyable_class>::value &&
         "device_copyable_class must not be trivially_copyable.");
  assert(is_device_copyable<device_copyable_class>::value &&
         "device_copyable_class is not device copyable.");
  using device_copyable_annotated_arg = annotated_arg<device_copyable_class>;
  assert(is_device_copyable<device_copyable_annotated_arg>::value &&
         "annotated_arg<device_copyable_class> is not device copyable.");
  using device_copyable_annotated_arg_with_properties =
      annotated_arg<device_copyable_class>;
  assert(is_device_copyable<
             device_copyable_annotated_arg_with_properties>::value &&
         "annotated_arg<device_copyable_class, properties> is not device "
         "copyable.");

  free(a, Q);
  free(b, Q);
  free(c->b, Q);
  free(c, Q);
  free(d, Q);
  free(r1, Q);
  free(r2, Q);
  free(r3, Q);
  free(r4, Q);
  free(r5, Q);
  free(r6, Q);
  free(r7, Q);
  free(r8, Q);
  free(r9, Q);

  return 0;
}
