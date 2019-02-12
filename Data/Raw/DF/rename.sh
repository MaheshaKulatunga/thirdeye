# Remove Spaces
for oldname in *
do
  ffprobe -i $oldname -show_entries format=duration -v quiet -of csv="p=0"
done
# New file names
# count=0
# for f in ./*.mp4;
#  do mv $f ./$count.mp4;
#  (( count++ ))
# done
