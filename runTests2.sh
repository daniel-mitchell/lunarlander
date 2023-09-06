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

ALPHA_U=0.1 #(0.1) #(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1)
ALPHA_V=0.01 #(0.1) #(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1)
ALPHA_UU=0.1 #(0.1) #(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1)
ALPHA_VV=0.01 #(0.01) #(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1)
ALPHA_R=0.0001 #(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1)
LAMBDA=0.75 #(0.0 0.5 0.75 0.875 0.9375 0.96875)
LAMBDA_C=0.9375 #(0.0 0.5 0.75 0.875 0.9375 0.96875)
EPSILON=0.05 #(0 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1)

AGENT_TYPE=2
MAX_TIME=1000000
EPISODES=1

FILE_PREFIX="mgTest"
DIR=$(dirname $(realpath $0))
mkdir $DIR/data/$FILE_PREFIX/
for i in $(seq $MIN $MAX); do
    $DIR/build/main --agent-seed $i --init-seed $i --episodes $EPISODES --agent-type $AGENT_TYPE $TRUNC_NORMAL\
      --alpha-u $ALPHA_U --alpha-v  $ALPHA_V --alpha-uu $ALPHA_UU --alpha-vv $ALPHA_VV --alpha-r $ALPHA_R\
      --lambda $LAMBDA --epsilon $EPSILON --lambda-c $LAMBDA_C --max-time $MAX_TIME\
       >> $DIR/data/$FILE_PREFIX/$FILE_PREFIX$i;
done
