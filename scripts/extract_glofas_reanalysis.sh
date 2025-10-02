#!/bin/sh

# you can change this to your absolute path
dir="./GloFAS_Reanalysis_Global"

find "$dir" -name '*.7z' -exec 7za x {} -o"$dir" \;