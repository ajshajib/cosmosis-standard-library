#!/bin/bash

# Array of modifiers to append to the run names
modifiers=("" "_lim" "_lcdm")
# Associative array to specify which runs to execute
declare -A run_specs=(
    ["wl"]=1
    ["bao_cmb"]=1
    ["cmb"]=1
    ["sl"]=1
    ["sn_bao"]=1
    ["sn"]=1
    ["bao"]=1
)
# Loop through each modifier and run specification to execute cosmosis with the corresponding ini file
for mod in "${modifiers[@]}"; do
    for run in "${!run_specs[@]}"; do
        export RUN_NAME="${run}${mod}"
        echo "##################################################"
        echo "Running ${RUN_NAME}"
        echo "##################################################"
        if [ -f "inis/${run}.ini" ]; then
            cosmosis "inis/${run}.ini"
        else
            echo "##################################################"
            echo "File inis/${run}.ini does not exist."
            echo "##################################################"
        fi
    done
done