#!/usr/bin/env perl

use JSON::XS;
use List::Util qw(max);
use feature say;

my $total_res = 0;
my @vs;
my @iterations;
while (<>) {
    $total_res += 1;
    $data = decode_json $_;
    push @vs, -$data->{obj_info}->{value};
    if (exists $data->{iteration}) {
    push @iterations, $data->{iteration};
  } else {
    push @iterations, $.;
  }

}
my $num = max @vs;
my @argmax = grep $vs[$_]==$num, 0..$#vs;
say "max is: $num (in $total_res resuls)";
say "argmax: @iterations[@argmax]";
