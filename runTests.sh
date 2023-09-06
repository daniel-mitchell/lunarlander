if [ $# -gt 1 ]; then
    MIN=$1;
    MAX=$2
elif [ $# -gt 0 ]; then
    MAX=$1;
    MIN=0
else
    MAX=29;
    MIN=0;
fi

AGENT_TYPE=0
MAX_TIME=200
ALPHA_U=0.005
ALPHA_V=0.005
ALPHA_UU=0.0005
ALPHA_VV=0.01
ALPHA_R=0.01
LAMBDA=0.0
LAMBDA_C=0.0

FILE_PREFIX="multiGauss4"
DIR=$(dirname $(realpath $0))
for i in $(seq $MIN $MAX); do
    $DIR/build/main --episodes 200 --agent-seed $i --init-seed $i --agent-type $AGENT_TYPE --trunc-normal\
        --alpha-u $ALPHA_U --alpha-v $ALPHA_V --alpha-uu $ALPHA_UU --alpha-vv $ALPHA_VV --alpha-r $ALPHA_R\
        --lambda $LAMBDA --lambda-c $LAMBDA_C --subspaces 0,1,2\
        --epsilon 0 --no-print-rewards --print-timesteps --max-time $MAX_TIME --continuing\
            >> $DIR/data/$FILE_PREFIX$i;
done
