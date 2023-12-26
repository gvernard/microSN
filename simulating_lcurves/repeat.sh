#!/bin/bash

plot_dir="/home/giorgos/Desktop/My Papers/unpublished/2024_microSN/MPDs"
pwd=`pwd`

for (( i=32; i<100; i++ )); do
    jq --argjson i "$i" '.locations.Nside = $i' input.json > tmp.json
    mv tmp.json input.json
    integral=$(./bin/microlc | tail -1)
    echo $i $integral
    cd "$plot_dir"
    python plot_mpd.py 2> /dev/null
    mv mpd.png mpd_${i}.png
    cd $pwd
done
