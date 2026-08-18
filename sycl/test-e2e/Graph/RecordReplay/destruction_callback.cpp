// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/destruction_callback.cpp"
