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
my $num_s = sprintf("%.3f",$num);
my $total_res_s = sprintf("%-3d", $total_res);
my @argmax = grep $vs[$_]==$num, 0..$#vs;
say " max is: $num_s (in $total_res_s resuls), (argmax: @iterations[@argmax])";
