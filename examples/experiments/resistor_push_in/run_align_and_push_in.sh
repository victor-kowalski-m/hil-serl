for i in $(seq 1 10);
do
    echo $i
    cd ../resistor_align/
    bash run_actor.sh || true
    # bash run_reset.sh || true
    cd ../resistor_push_in/ || true
    bash run_actor.sh
done

