export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../multi_actor_eval.py "$@" \
    --exp_names=resistor_align,resistor_push_in \
    --checkpoint_paths=../resistor_align/first_run,first_run \
    --eval_checkpoint_steps=5000,5000 \
    --eval_n_trajs=10

