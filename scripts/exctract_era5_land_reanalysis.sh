#!/bin/sh

# you can change this to your absolute path
dir="./ERA5-Land_Reanalysis_Global"

find "$dir" -name '*.7z.001' -exec 7za x {} -o"$dir" \;