// This test will check that --sym-prop-bc-files can be used to supply symbols and properties
// for creating a wrapped BC file.
//
// This test will generate two wrapped BC files.
//
// The first one (%t1.bc) will be generated with code, properties, and symbols coming through a
// "standard" table file.
//
// The second one (%t2.bc) will be generated with only code (which has been modified
// from the code used to generate %t1.bc) coming through a table file.
// The symbols and properties will come through the --sym-prop-bc-files option.
// The file (%t1.bc) will be listed in the file specified by the --sym-prop-bc-files option.
// Thus %t2.bc will be generated with the same symbols and properties as in %t1.bc.
// The two wrapped BC files, %t1.bc and %t2.bc, will be identical except for their code sections.
//
// TEST1: will check %t1.bc and %t2.bc are identical except for the expected differences in their
//        code sections.
// TEST2: will check that the Code sections have the expected contents.

///////////////////////////////////////////////////////////////////////////////////////////
// Generate wrapped BC file where code, properties and symbols come through a table file.
// This is the "standard" way to make a wrapped BC file.
///////////////////////////////////////////////////////////////////////////////////////////

// Generate fake AOCX files
// RUN: echo 'pseudo-aocx-0' > %t0.aocx
// RUN: echo 'pseudo-aocx-1' > %t1.aocx
// RUN: echo 'pseudo-aocx-2' > %t2.aocx

// Generate property files
// RUN: echo '[SYCL/devicelib req mask]'                                     >  %t0.prop
// RUN: echo 'DeviceLibReqMask=1|0'                                          >> %t0.prop
// RUN: echo '[SYCL/device requirements]'                                    >> %t0.prop
// RUN: echo 'aspects=2|AAAAAAAAAAA'                                         >> %t0.prop
// RUN: echo 'reqd_work_group_size=2|AAAAAAAAAAA'                            >> %t0.prop
// RUN: echo '[SYCL/kernel param opt]'                                       >> %t0.prop
// RUN: echo '_ZTSZ4add5N4sycl3_V15queueEPiiiE10add5_dummy=2|DAAAAAAAAAAA'   >> %t0.prop
// RUN: echo '_ZZ4add5N4sycl3_V15queueEPiiiENKUlvE_clEv=2|BAAAAAAAAAAA'      >> %t0.prop
// RUN: echo '[SYCL/misc properties]'                                        >> %t0.prop
// RUN: echo 'optLevel=1|2'                                                  >> %t0.prop

// RUN: echo '[SYCL/devicelib req mask]'                                     >  %t1.prop
// RUN: echo 'DeviceLibReqMask=1|0'                                          >> %t1.prop
// RUN: echo '[SYCL/device requirements]'                                    >> %t1.prop
// RUN: echo 'aspects=2|AAAAAAAAAAA'                                         >> %t1.prop
// RUN: echo 'reqd_work_group_size=2|AAAAAAAAAAA'                            >> %t1.prop
// RUN: echo '[SYCL/kernel param opt]'                                       >> %t1.prop
// RUN: echo '_ZZ4add4N4sycl3_V15queueEPiiiENKUlvE_clEv=2|BAAAAAAAAAAA'      >> %t1.prop
// RUN: echo '_ZTSZ4add4N4sycl3_V15queueEPiiiE10add4_dummy=2|DAAAAAAAAAAA'   >> %t1.prop
// RUN: echo '[SYCL/misc properties]'                                        >> %t1.prop
// RUN: echo 'optLevel=1|2'                                                  >> %t1.prop

// RUN: echo '[SYCL/devicelib req mask]'                                     >  %t2.prop
// RUN: echo 'DeviceLibReqMask=1|0'                                          >> %t2.prop
// RUN: echo '[SYCL/device requirements]'                                    >> %t2.prop
// RUN: echo 'aspects=2|AAAAAAAAAAA'                                         >> %t2.prop
// RUN: echo 'reqd_work_group_size=2|AAAAAAAAAAA'                            >> %t2.prop
// RUN: echo '[SYCL/kernel param opt]'                                       >> %t2.prop
// RUN: echo '_ZTSZ4add3N4sycl3_V15queueEPiiiE10add3_dummy=2|DAAAAAAAAAAA'   >> %t2.prop
// RUN: echo '_ZZ4add3N4sycl3_V15queueEPiiiENKUlvE_clEv=2|BAAAAAAAAAAA'      >> %t2.prop
// RUN: echo '[SYCL/misc properties]'                                        >> %t2.prop
// RUN: echo 'optLevel=1|2'                                                  >> %t2.prop

// Generate sym files
// RUN: echo '_ZTSZ4add5N4sycl3_V15queueEPiiiE10add5_dummy'                  >  %t0.sym
// RUN: echo '_ZTSZ4add4N4sycl3_V15queueEPiiiE10add4_dummy'                  >  %t1.sym
// RUN: echo '_ZTSZ4add3N4sycl3_V15queueEPiiiE10add3_dummy'                  >  %t2.sym

// Generate table file
// RUN: echo '[Code|Properties|Symbols]'                                     >  %t.table
// RUN: echo '%t0.aocx|%t0.prop|%t0.sym'                                     >> %t.table
// RUN: echo '%t1.aocx|%t1.prop|%t1.sym'                                     >> %t.table
// RUN: echo '%t2.aocx|%t2.prop|%t2.sym'                                     >> %t.table

// Generate BC file with Code, Properties and Symbols by the standard method
// RUN: clang-offload-wrapper "-o=%t1.bc" "-host=x86_64-unknown-linux-gnu" "--emit-reg-funcs=0" "-target=fpga_aocx-intel-unknown" "-kind=sycl" "-batch"  %t.table

///////////////////////////////////////////////////////////////////////////////////////////
// Generate wrapped BC file where only code comes through the standard table file.
// And symbols and properties come through symbols and properties files
// specified by --sym-prop-bc-files=<table of sym/prop files>
// sym/prop file(s) are previously wrapped BC files.
///////////////////////////////////////////////////////////////////////////////////////////

// Update fake AOCX files (i.e. modify the code)
// RUN: echo 'fake-aocx-0-updated'        > %t0.aocx
// RUN: echo 'fake-aocx-1-enhanced'       > %t1.aocx
// RUN: echo 'fake-aocx-2-different-form' > %t2.aocx

// Create a table with only Code entries.
// RUN: file-table-tform  --extract=Code  %t.table  -o %t_code_only.table

// Create "table of sym/prop files"
// The file "%t1.bc" listed in the "table of sym/prop files"
// is the previously wrapped BC file generated by the "standard" method.
// RUN: echo "[SymAndProps]" > %t_sym_prop_files.txt
// RUN: echo %t1.bc >> %t_sym_prop_files.txt

// Generate BC file with only Code coming from the table but symbols and properties coming through --sym-prop-bc-files
// RUN: clang-offload-wrapper "-o=%t2.bc" "-host=x86_64-unknown-linux-gnu" "--emit-reg-funcs=0" "-target=fpga_aocx-intel-unknown" "-kind=sycl" "-batch"  %t_code_only.table --sym-prop-bc-files=%t_sym_prop_files.txt

// The two wrapped BC files, %t1.bc (generated by the standard table method) and
// %t2.bc (generated with symbols and properties coming from the --sym-prop-bc-files option),
// should be the same except for their Code.
// RUN: llvm-dis %t1.bc
// RUN: llvm-dis %t2.bc

// Filter out expected differences
//   ModuleID:                      filename is different
//   sycl_offloading.?.data:        fake AOCX were updated
//   sycl_offloading.device_images: pointer type to offloading data includes the length which is different
// The grep -v commands will filter out the expected differences.
// RUN: grep -v '\(ModuleID\|^@\.sycl_offloading\.[0-9]\.data\|^@\.sycl_offloading\.device_images\)' %t1.ll > %tfiltered1.txt
// RUN: grep -v '\(ModuleID\|^@\.sycl_offloading\.[0-9]\.data\|^@\.sycl_offloading\.device_images\)' %t2.ll > %tfiltered2.txt

// TEST1
// Verify disassembly is identical except for filterered differences
// RUN: cmp %tfiltered1.txt %tfiltered2.txt

// RUN: FileCheck --check-prefix=CHECK_LL1 < %t1.ll %s
// RUN: FileCheck --check-prefix=CHECK_LL2 < %t2.ll %s

// TEST2
// Check that expected code is found
// CHECK_LL1: pseudo-aocx-0
// CHECK_LL1: pseudo-aocx-1
// CHECK_LL1: pseudo-aocx-2

// CHECK_LL2: fake-aocx-0-updated
// CHECK_LL2: fake-aocx-1-enhanced
// CHECK_LL2: fake-aocx-2-different-form
