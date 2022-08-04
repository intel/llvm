#!/usr/bin/perl

# Visits every directory, calls make clean, and then removes the Makefile
#
# Written by Tomofumi Yuki, 11/21 2014
#

my $TARGET_DIR = ".";

if ($#ARGV != 0) {
   printf("usage perl clean.pl target-dir\n");
   exit(1);
}



if ($#ARGV == 0) {
   $TARGET_DIR = $ARGV[0];
}


my @categories = ('linear-algebra/blas',
                  'linear-algebra/kernels',
                  'linear-algebra/solvers',
                  'datamining',
                  'stencils',
                  'medley');


foreach $cat (@categories) {
   my $target = $TARGET_DIR.'/'.$cat;
   opendir DIR, $target or die "directory $target not found.\n";
   while (my $dir = readdir DIR) {
        next if ($dir=~'^\..*');
        next if (!(-d $target.'/'.$dir));

        my $targetDir = $target.'/'.$dir;
        my $command = "cd $targetDir; make clean; rm -f Makefile";
        print($command."\n");
        system($command);
   }

   closedir DIR;
}

my $cfgFile = $TARGET_DIR.'/'.'config.mk';
if (-e $cfgFile) {
  unlink $cfgFile;
}

