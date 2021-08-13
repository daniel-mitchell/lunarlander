if [ $# -gt 1 ]; then
    MIN=$1;
    MAX=$2
elif [ $# -gt 0 ]; then
    MAX=$1;
    MIN=0
else
    MAX=99;
    MIN=0;
fi

FILE_PREFIX="testAgent"
DIR=$(dirname $(realpath $0))
for i in $(seq $MIN $MAX); do
    $DIR/build/main --episodes 1000 --agent-seed $i --init-seed 0 --agent-type 2 >> $DIR/data/$FILE_PREFIX$i;
done
