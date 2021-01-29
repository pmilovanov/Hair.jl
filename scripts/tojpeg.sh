#!/usr/bin/bash

#INDIR=$1
OUTDIR=$1



mkdir -p $OUTDIR

ls ~/drives/pico/Pico/DINA-downloads/Hermitage\ HiRes\ TIFF/*.tif |
	while read line; do
	    OUTFILE=$(basename "$line" | sed 's/ /_/g' | sed 's/.tif/.jpg/')
		echo $line
		convert "$line" "$OUTDIR/$OUTFILE"
	done





