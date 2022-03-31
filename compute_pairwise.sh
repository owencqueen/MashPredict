#!/bin/bash

touch pairwise_dist.txt

for sketch in *.msh; do

	mash dist "$sketch" *.msh >> pairwise_dst.txt
done
