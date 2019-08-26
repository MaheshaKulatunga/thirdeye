cd $1
echo $1
pwd
for file in $1/*.mp4                                      
do mv "$file" "${file//[ ()@$]/}"
done
