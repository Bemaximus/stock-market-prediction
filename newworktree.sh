git checkout --orphan $1
git reset --hard
git commit --allow-empty -m "Initial commit"
git push origin $1
git checkout master
echo $1 >> branches.txt
git add branches.txt
git commit -m "Add $1 to branches.txt"
cd strategies
git worktree add -B $1 $1/ origin/$1
cd ..
