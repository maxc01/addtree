#!/usr/bin/env perl

use Mojo::JSON qw(decode_json encode_json to_json from_json j);
use Data::Dumper;
use feature 'say';


sub extract_info {
  my @res;
  my $fh = shift;
  while (<$fh>) {
    my $text = $_;
    say $text;
    if ($text =~ /Finishing BO (\d+) iteration/) {
      push @res, $1;
    }
    if ($text =~ /(\{.*\})/) {
      my $new = $1 =~ s/\'/\"/gr;
      push @res, $new;
    }
  }

  return @res;
}


# foreach (@ARGV) {
#   say $_;
#   my @res = extract_info($_);
#   say "@res\n";
# }


my @res;
while (<>) {
    my $text = $_;
    if ($text =~ /Finishing BO (\d+) iteration/) {
      # push @res, $1;
      say $1;
    }
    if ($text =~ /(\{.*\})/) {
      my $new = $1 =~ s/\'/\"/gr;
      # push @res, $new;
      say $new;
    }
}





# while(<>) {

#     my $res = {};

#     if ($text =~ /Finishing BO (\d+) iteration/) {
#         $res->{iteration} = $1;
#     }

#     # my $text = "2020-04-11 02:01:51,421 - ModelCompression - INFO -
#     # {'conv1': { 'prune_method': 'ln', 'amount': 0.5904995264593453},
#     # 'conv2': { 'prune_method': 'l1', 'amount': 0.22352045270027143}, 'fc1':
#     # {' prune_method': 'l1', 'amount': 0.9676065194571231}}";
#     if ($text =~ /(\{.*\})/) {
#         my $new = $1 =~ s/\'/\"/gr;
#         my $data = from_json $new;
#         $res->{param} = $data;
#     }

#     # my $text = "blabla {'top1': 0.9908, 'sparsity': 0.9234571092831962,
#     # 'value': -1.9142571092831964, 'value_sigma': 1e-05}";
#     # if ($text =~ /(\{.*\})/) {
#     #     my $new = $1 =~ s/\'/\"/gr;
#     #     my $data = from_json $new;
#     #     $res->{obj_info} = $data;
#     # }



# }


