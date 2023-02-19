#! /usr/bin/bash
CUDA_VISIBLE_DEVICES=0 python hurr_u_net_2d0.0.py &
CUDA_VISIBLE_DEVICES=1 python hurr_u_net_2d0.1.py &
CUDA_VISIBLE_DEVICES=2 python hurr_u_net_2d0.2.py &
