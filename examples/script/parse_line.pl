#!/usr/bin/env perl

use JSON::XS;
use List::Util qw(max);
use feature say;

my $total_res=0;
my @vs;
while (<>) {
        $total_res += 1;
        $data = decode_json $_;
        push @vs, -$data->{obj_info}->{value};
}
my $num = max @vs;
say "max is: $num (in $total_res resuls)";
