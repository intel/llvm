// RUN: cgeist %s -w -O0 --function=* -S | FileCheck %s
// XFAIL: *

void lambda_capture(bool x, bool &y) {
  const auto f = [x, &y]() { y = !x; };
  f();
}
