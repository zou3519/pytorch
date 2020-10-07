
# Run test_autograd
pytest test/test_autograd.py -v -k "not test_sparse_ctor_getter" --result-log=test_autograd.txt
python analysis/read_test_logs.py test_autograd.txt

# Run test_nn
pytest test/test_nn.py -v -k "not Conv1d_zero_batch and not Conv2d_zero_batch and not Conv3d_zero_batch and not interpolate_bicubic_2d_zero_dim and not interpolate and not pdist" --result-log=test_nn.txt
