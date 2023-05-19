// RUN: polygeist-opt %s --split-input-file -verify-diagnostics

// expected-error @+1 {{opaque struct type must have a name}}
func.func @test_struct(%arg0: !polygeist.struct<isOpaque=true>) {
  return
}
