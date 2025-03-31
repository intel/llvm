// RUN: %clangxx -fsycl -fsycl-targets=spir64 %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: %if hip || cuda %{ not %} %{run} %t.out
