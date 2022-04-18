#!/bin/bash

for f in *.msh; do

	out=$(basename "$f" .msh)
	mash info -d "$f" > json/"$out".json

done
