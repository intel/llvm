// RUN: %{build} -ffp-contract=fast -o %t.fp-contract.out
// RUN: %{run} %t.fp-contract.out
// RUN: %{build} -ffp-contract=on -o %t.fp-contract-on.out
// RUN: %{run} %t.fp-contract-on.out
// RUN: %{build} -ffp-contract=fast-honor-pragmas -o %t.fp-contract-fast-honor-pragmas.out
// RUN: %{run} %t.fp-contract-fast-honor-pragmas.out

#include "common.hpp"
