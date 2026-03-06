// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -aux-triple x86_64-pc-windows-msvc -fsyntax-only -verify %s

// expected-no-diagnostics

#include "Inputs/sycl.hpp"
