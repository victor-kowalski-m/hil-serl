export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=resistor_insertion \
    --checkpoint_path=first_run \
    --actor \
    --ip=pc81-182 \
    # for eval vvv
    # --eval_checkpoint_step=15000 \ 
    # --eval_n_trajs=20 

