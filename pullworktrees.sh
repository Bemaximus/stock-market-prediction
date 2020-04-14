for branch in $(cat branches.txt)
do
    echo "Pulling $branch"
    cd strategies
    cd $branch
    echo $(pwd)
    git pull
    cd ../..
done
