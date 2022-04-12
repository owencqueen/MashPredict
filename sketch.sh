#!/bin/bash

# Genome size with scaffolds: 434134862
# Parallelizing:
#	Make diff tmp dir for each run
#	First batch: ls | head -n 333
#	Second batch: ls | head -n 666 | tail -n 333
#	Third batch: ls | head -n 999 | tail -n 333
#	Fourth batch: ls | tail -n 332

mkdir tmp1

for geno in $(ls /mnt/data/core_data/reads/ | head -n  333)
do

	cat /mnt/data/core_data/reads/"$geno"/*.fastq.gz > tmp1/reads.fastq.gz
	mash sketch -p 10 -m 2 -s 2000 -g 434134862 -I "$geno" -o "$geno" tmp1/reads.fastq.gz

done
