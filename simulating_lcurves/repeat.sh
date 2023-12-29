#!/bin/bash

plot_dir="/home/giorgos/Desktop/My Papers/unpublished/2024_microSN/MPDs"
pwd=`pwd`


# for (( i=32; i<100; i++ )); do
#     jq --argjson i "$i" '.locations.Nside = $i' grid_input.json > tmp.json
#     mv tmp.json grid_input.json
#     integral=$(./bin/microlc grid_input.json | tail -1)
#     echo $i $integral
#     cd "$plot_dir"
#     python plot_mpd.py 2> /dev/null
#     mv mpd.png mpd_grid_${i}.png
#     cd $pwd
# done



for (( i=32; i<100; i=i+8 )); do
    j=$(( i*i ))
    jq --argjson j "$j" '.locations.Nloc = $j' rand_input.json > tmp1.json

    for (( s=1; s<6; s++ )); do
	jq --argjson s "$s" '.locations.seed = $s' tmp1.json > tmp2.json
	mv tmp2.json rand_input.json
	integral=$(./bin/microlc rand_input.json | tail -1)
	echo $i $s $integral
	cd "$plot_dir"
	python plot_mpd.py 2> /dev/null
	mv mpd.png mpd_rand_${i}_${s}.png
	cd $pwd
    done
    
done
rm tmp1.json
