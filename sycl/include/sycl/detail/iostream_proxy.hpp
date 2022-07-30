#pragma once
#include <istream>
#include <ostream>

namespace std {
extern istream cin;  /// Linked to standard input
extern ostream cout; /// Linked to standard output
extern ostream cerr; /// Linked to standard error (unbuffered)
extern ostream clog; /// Linked to standard error (buffered)
} // namespace std