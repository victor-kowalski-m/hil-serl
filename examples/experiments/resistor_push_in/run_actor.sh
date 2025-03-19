export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=resistor_push_in \
    --checkpoint_path=first_run \
    --actor \
    --ip=pc81-182 \
    --eval_checkpoint_step=5000 \
    --eval_n_trajs=1

