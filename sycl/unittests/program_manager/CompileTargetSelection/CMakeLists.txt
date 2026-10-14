# Uses its own test binary: the registered images decide which devices have a
# compatible image, so other tests' images must not be visible.
add_sycl_unittest(CompileTargetSelectionTests OBJECT
  CompileTargetSelection.cpp
)
