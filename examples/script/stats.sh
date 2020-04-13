#!/usr/bin/env bash

echo SMAC info
perl ./script/parse_line.pl ./checkpoints_smac/smac_iter_hists.txt

echo ==========
echo RANDOM info
perl ./script/parse_line.pl ./checkpoints_random/random_iter_*.txt

echo ==========
echo ADDTREE info
perl ./script/parse_line.pl ./checkpoints_addtree/addtree_iter_*.json

