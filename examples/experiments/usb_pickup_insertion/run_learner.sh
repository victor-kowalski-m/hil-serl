export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=usb_pickup_insertion \
    --checkpoint_path=../../experiments/usb_pickup_insertion/debug \
    --demo_path=../../demo_data/usb_pickup_insertion_20_demos_2025-02-18_17-28-50.pkl \
    --learner \
