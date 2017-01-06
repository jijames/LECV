#!/bin/bash
# This script uses VLC to extract frames from video and
# save them to a folder

# Arguments: Full path to a video file
VIDEO=$1
$(cvlc -h > /dev/null)
if [ $? -eq 0 ]; then
	echo "I: VLC is installed"
else
	echo "E: VLC is NOT installed!"
	echo "E: Please install VLC (cvlc) first"
	exit 127
fi

if [ "$VIDEO" != "" ] && [ -f "$VIDEO" ]; then
	echo "I: Video selected: $VIDEO"
else
	echo "E: No video file was selected!"
	echo ""
	echo "Usage: $0 video.file"
	exit 127
fi

# if we made it here, we have a video file and vlc is installed
OUT="$(mktemp -d)"
echo "I: Temp out is ${OUT}"
echo "I: Running..."
# Change the rate (speed) and the scene-ration (how many frames between) to change
# how often a screenshot is taken
$(cvlc "$VIDEO" --rate=25 --video-filter=scene --vout=dummy --aout=dummy \
	--scene-format=jpg --scene-ratio=100 --scene-prefix=snap --scene-path="$OUT" vlc://quit)

echo "I: Moving temp folder to current directory..."
$(mv ${OUT} ${PWD}/)
