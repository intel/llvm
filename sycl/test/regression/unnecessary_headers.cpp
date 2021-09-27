// RUN: %clangxx %fsycl-host-only -fsyntax-only -std=c++17 -Xclang -verify -Xclang -verify-ignore-unexpected=note %s -o %t.out

#include <sycl/sycl.hpp>

// Make sure iostream, fstream, stringstream and cmath are not included
int main() {
  // expected-error@+1 {{no member named 'cout' in namespace 'std'}}
  std::cout << "Hello, world!";
  // expected-error@+1 {{no member named 'round' in namespace 'std'; did you mean 'sycl::round'}}
  std::round(1.5f);
  // expected-error@+1{{implicit instantiation of undefined template 'std::basic_stringstream<char>'}}
  std::stringstream stream;
  // expected-error@+1{{implicit instantiation of undefined template 'std::basic_ofstream<char>'}}
  std::ofstream of{"file.txt"};
}
