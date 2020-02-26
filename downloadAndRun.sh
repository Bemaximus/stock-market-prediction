if [ -n $1 ]; then
	cd data
	sh get-yahoo-quotes.sh $1
	cd ..
	echo $1 > options.txt
fi

matlab -nodesktop -nosplash -nojvm < main.m > output/output.txt