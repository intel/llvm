#!/usr/bin/perl

# Creates C Pre-Processed Version
# Additional arguments to the script are all passed to gcc.
# At least the include path of polybench.h must be added.
#
# Written by Tomofumi Yuki, 01/14 2015
#

if ($#ARGV == -1) {
   printf("usage perl create-cpped-version.pl filename [cflags]\n");
   exit(1);
}

my @CFLAGS = @ARGV;
my $TARGET = shift @CFLAGS;

my $TEMP_T = '.__poly_top.c';
my $TEMP_B = '.__poly_bottom.c';
my $TEMP_P = '.__poly_bottom.pp.c';

$TARGET =~ /([^\.\/]+)\.c$/;
my $KERNEL = $1;
$TARGET_DIR = substr $TARGET, 0, -length($KERNEL)-2;
$TARGET_DIR = '.' if $TARGET_DIR eq '';

open FILE, $TARGET or die "Error opening $TARGET";

my $top;
my $bottom;
my $current = \$top;
while (<FILE>) {
   my $line = $_;
   if ($line =~ /polybench\.h/) {
     $current = \$bottom;
   }
   $$current .= $line;
}
close FILE;

&writeToFile($TEMP_T, $top);
&writeToFile($TEMP_B, $bottom);

my $ignoreLibs = "-D_STDLIB_H_ -D_STDIO_H_ -D_MATH_H_ -D_STRING_H_ -D_UNISTD_H_";

my $command = 'gcc -E '.$ignoreLibs.' '.$TEMP_B.' -I '.$TARGET_DIR.' '.join(" ", @CFLAGS).' 2>/dev/null > '.$TEMP_P;
system($command);

my $OUTFILE = $TARGET_DIR.'/'.$KERNEL.'.preproc.c';
system('cat '.$TEMP_T.' > '.$OUTFILE);
system('echo "#include<polybench.h>" >> '.$OUTFILE);
$command = 'sed -e "s~'.$TEMP_B.'~'.$KERNEL.'.c~g" '.$TEMP_P.' >> '.$OUTFILE;
system($command);

unlink $TEMP_P;
unlink $TEMP_B;
unlink $TEMP_T;

sub writeToFile() {
   my $file = $_[0];
   my $content = $_[1];

   open FILE, ">$file" or die "Error writing to $file";

   print FILE $content;

   close FILE;
}
