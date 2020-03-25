for branch in $(cat branches.txt)
do
    echo $branch
    git checkout master
    cd strategies
    git worktree add -B $branch $branch/ origin/$branch
    cd ..
done
