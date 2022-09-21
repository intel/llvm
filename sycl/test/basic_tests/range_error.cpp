// RUN: %clangxx %s %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning
#include <cassert>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace std;
int main() {
  sycl::range<1> one_dim_range(64);
  sycl::range<2> two_dim_range(64, 1);
  sycl::range<3> three_dim_range(64, 1, 2);
  assert(one_dim_range.size() ==64);
  assert(one_dim_range.get(0) ==64);
  assert(one_dim_range[0] ==64);
  cout << "one_dim_range passed " << endl;
  assert(two_dim_range.size() ==64);
  assert(two_dim_range.get(0) ==64);
  assert(two_dim_range[0] ==64);
  assert(two_dim_range.get(1) ==1);
  assert(two_dim_range[1] ==1);
  cout << "two_dim_range passed " << endl;
  assert(three_dim_range.size() ==128);
  assert(three_dim_range.get(0) ==64);
  assert(three_dim_range[0] ==64);
  assert(three_dim_range.get(1) ==1);
  assert(three_dim_range[1] ==1);
  assert(three_dim_range.get(2) ==2);
  assert(three_dim_range[2] ==2);
  cout << "three_dim_range passed " << endl;
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::range<1>'}}
  sycl::range<1> one_dim_range_f1(64, 2, 4);
  // expected-error@+1 {{no matching constructor for initialization of 'sycl::range<2>'}}
  sycl::range<2> two_dim_range_f1(64);
}
