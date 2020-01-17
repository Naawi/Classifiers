#!/bin/bash
# A shell script to print each number five times.
for (( i = 0; i < 30; i++ ))      ### Outer for loop ###
do

    for (( j = i + 1 ; j < 30; j++ )) ### Inner for loop ###
    do
          echo "Plotting graph for features $i $j"
          py classify-breast-cancer.py --f1 $i --f2 $j
    done

  echo "Done!" 
done