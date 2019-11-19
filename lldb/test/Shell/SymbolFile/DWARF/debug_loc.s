# Test debug_loc parsing, including the cases of invalid input. The exact
# behavior in the invalid cases is not particularly important, but it should be
# "reasonable".

# REQUIRES: x86

# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s > %t
# RUN: %lldb %t -o "image lookup -v -a 0" -o "image lookup -v -a 2" -o exit \
# RUN:   | FileCheck %s

# CHECK-LABEL: image lookup -v -a 0
# CHECK: Variable: {{.*}}, name = "x0", type = "int", location = DW_OP_reg5 RDI,
# CHECK: Variable: {{.*}}, name = "x1", type = "int", location = ,
# CHECK: Variable: {{.*}}, name = "x2", type = "int", location = ,

# CHECK-LABEL: image lookup -v -a 2
# CHECK: Variable: {{.*}}, name = "x0", type = "int", location = DW_OP_reg0 RAX,
# CHECK: Variable: {{.*}}, name = "x1", type = "int", location = ,
# CHECK: Variable: {{.*}}, name = "x2", type = "int", location = ,
# CHECK: Variable: {{.*}}, name = "x3", type = "int", location = DW_OP_reg1 RDX,

        .type   f,@function
f:                                      # @f
.Lfunc_begin0:
        nop
.Ltmp0:
        nop
.Ltmp1:
        nop
.Ltmp2:
        nop
.Ltmp3:
        nop
.Ltmp4:
        nop
.Lfunc_end0:
        .size   f, .Lfunc_end0-f

        .section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "Hand-written DWARF"
.Linfo_string3:
        .asciz  "f"
.Linfo_string4:
        .asciz  "int"

        .section        .debug_loc,"",@progbits
.Ldebug_loc0:
        .quad   .Lfunc_begin0-.Lfunc_begin0
        .quad   .Ltmp0-.Lfunc_begin0
        .short  1                       # Loc expr size
        .byte   85                      # super-register DW_OP_reg5
        .quad   .Ltmp0-.Lfunc_begin0
        .quad   .Lfunc_end0-.Lfunc_begin0
        .short  1                       # Loc expr size
        .byte   80                      # super-register DW_OP_reg0
        .quad   0
        .quad   0

.Ldebug_loc3:
        .quad   -1                      # Select base address
        .quad   .Ltmp1
        .quad   .Ltmp1-.Ltmp1
        .quad   .Ltmp2-.Ltmp1
        .short  1                       # Loc expr size
        .byte   81                      # super-register DW_OP_reg1
        .quad   0
        .quad   0

.Ldebug_loc2:
        .quad   .Lfunc_begin0-.Lfunc_begin0
        .quad   .Lfunc_end0-.Lfunc_begin0
        .short  0xdead                  # Loc expr size

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   14                      # DW_FORM_strp
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   46                      # DW_TAG_subprogram
        .byte   1                       # DW_CHILDREN_yes
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   5                       # DW_TAG_formal_parameter
        .byte   0                       # DW_CHILDREN_no
        .byte   2                       # DW_AT_location
        .byte   23                      # DW_FORM_sec_offset
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   4                       # Abbreviation Code
        .byte   36                      # DW_TAG_base_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   62                      # DW_AT_encoding
        .byte   11                      # DW_FORM_data1
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0x50 DW_TAG_compile_unit
        .long   .Linfo_string0          # DW_AT_producer
        .short  12                      # DW_AT_language
        .byte   2                       # Abbrev [2] 0x2a:0x29 DW_TAG_subprogram
        .quad   .Lfunc_begin0           # DW_AT_low_pc
        .long   .Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
        .long   .Linfo_string3          # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .long   .Ldebug_loc0            # DW_AT_location
        .asciz  "x0"                    # DW_AT_name
        .long   .Lint-.Lcu_begin0       # DW_AT_type
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .long   0xdeadbeef              # DW_AT_location
        .asciz  "x1"                    # DW_AT_name
        .long   .Lint-.Lcu_begin0       # DW_AT_type
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .long   .Ldebug_loc2            # DW_AT_location
        .asciz  "x2"                    # DW_AT_name
        .long   .Lint-.Lcu_begin0       # DW_AT_type
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .long   .Ldebug_loc3            # DW_AT_location
        .asciz  "x3"                    # DW_AT_name
        .long   .Lint-.Lcu_begin0       # DW_AT_type
        .byte   0                       # End Of Children Mark
.Lint:
        .byte   4                       # Abbrev [4] 0x53:0x7 DW_TAG_base_type
        .long   .Linfo_string4          # DW_AT_name
        .byte   5                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:
