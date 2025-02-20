export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=cable_route \
    --checkpoint_path=first_run \
    --demo_path=./demo_data/cable_route_20_demos_2025-01-27_18-00-19.pkl \
    --learner \
