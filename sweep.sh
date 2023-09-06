ALPHA_U_LIST=(0.001) #(0.0001) #(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1)
ALPHA_V_LIST=(0.001) #(0.001) #(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1)
ALPHA_UU_LIST=(0.1) #(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1)
ALPHA_VV_LIST=(0.01) #(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1)
ALPHA_R_LIST=(0.05) #(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1)
#ALPHA_DECAY_RATE=(2 1.8 1.6 1.4 1.2 1 0.8 0.6 0.4 0.2 0)
LAMBDA_LIST=(0.75) #(0.0 0.5 0.75 0.875 0.9375 0.96875)
LAMBDA_C_LIST=(0.9375) #(0.0 0.5 0.75 0.875 0.9375 0.96875)
EPSILON_LIST=(0, 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1)
GAMMA_LIST=(1) #(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
#INAC
#S
TRUNC_NORMAL_LIST=(--no-trunc-normal) #(--trunc-normal --no-trunc-normal)

NUM_RUNS=30
AGENT_TYPE=2
MAX_TIME=200
EPISODES=200
TEMPFILE="tempResults"
SCRIPTPATH=$(dirname $(realpath $0))

OUTPUT_FILE=$1

if [ $# -ne 1 ]; then
    1>&2 echo "Must specify output file.";
    exit 1;
fi
if [ -s "$TEMPFILE" ]; then
    1>&2 read -p "File $TEMPFILE will be overwritten. Continue? " -n 1 -r
    echo    # (optional) move to a new line
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        exit 1
    fi
    > "$TEMPFILE"
fi
if [ -s "$OUTPUT_FILE" ]; then
    1>&2 read -p "File $OUTPUT_FILE will be overwritten. Continue? " -n 1 -r
    echo    # (optional) move to a new line
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        exit 1
    fi
fi
for ALPHA_U in ${ALPHA_U_LIST[@]}; do
    for ALPHA_V in ${ALPHA_V_LIST[@]}; do
for ALPHA_UU in ${ALPHA_UU_LIST[@]}; do
    for ALPHA_VV in ${ALPHA_VV_LIST[@]}; do
        for ALPHA_R in ${ALPHA_R_LIST[@]}; do
            for LAMBDA in ${LAMBDA_LIST[@]}; do
            for LAMBDA_C in ${LAMBDA_C_LIST[@]}; do
                for EPSILON in ${EPSILON_LIST[@]}; do
                    for GAMMA in ${GAMMA_LIST[@]}; do
                        for TRUNC_NORMAL in ${TRUNC_NORMAL_LIST[@]}; do
                            TOTAL_RETURN=0;
                            echo -n Running $ALPHA_U $ALPHA_V $ALPHA_UU $ALPHA_VV $ALPHA_R $LAMBDA $EPSILON $GAMMA $TRUNC_NORMAL "   "
                            for I in $(seq -f "%02g" 0 $(expr $NUM_RUNS - 1)); do
                                echo -en \\b\\b$I
                                RETURN=$($SCRIPTPATH/build/main --agent-seed $I --init-seed $I --episodes $EPISODES\
                                    --agent-type $AGENT_TYPE $TRUNC_NORMAL --alpha-u $ALPHA_U --alpha-v  $ALPHA_V\
                                    --alpha-uu $ALPHA_UU --alpha-vv $ALPHA_VV --alpha-r $ALPHA_R --lambda $LAMBDA\
                                    --subspaces 0,1,2 --epsilon $EPSILON --lambda-c $LAMBDA_C\
                                    --gamma $GAMMA --no-print-episodes --max-time $MAX_TIME --continuing);
                                if [ $? -eq 1 ]; then
                                    TOTAL_RETURN="Error";
                                    break;
                                fi
                                TOTAL_RETURN=$(awk -v tr=$TOTAL_RETURN -v r=$RETURN 'BEGIN { print tr + r }')
                                echo $RETURN >> $OUTPUT_FILE"_"$ALPHA_U"_"$ALPHA_V"_"$ALPHA_UU"_"$ALPHA_VV"_"$ALPHA_R"_"$LAMBDA"_"$LAMBDA_C"_"$EPSILON"_"$TRUNC_NORMAL
                            done;
                            echo
                            echo $TOTAL_RETURN,$ALPHA_U,$ALPHA_V,$ALPHA_UU,$ALPHA_VV,$ALPHA_R,$LAMBDA,$LAMBDA_C,$EPSILON,$TRUNC_NORMAL >> $TEMPFILE;
                        done;
                    done;
                done;
            done;
            done;
        done;
    done;
done;
    done;
done;
sort $TEMPFILE -nr -k1 > $OUTPUT_FILE
rm $TEMPFILE