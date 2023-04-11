import argparse
import os
import pandas as pd 
from qllm.utils import ModelMemEstimator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter parameters for the model.')
    parser.add_argument('--h2', type=int, default=512, help='First dimension of fc2')
    parser.add_argument('--h1', type=int, default=256, help='Model hidden space')
    parser.add_argument('--b', type=int, default=32, help='Token batch size')
    parser.add_argument('--i', type=int, default=128, help='past length')
    parser.add_argument('--l', type=int, default=1, help='Number of layers')
    parser.add_argument('--unit', type=str, default='MB', help='Unit of the output (MB, GB, TB)')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--store', action='store_true', help='Store the result in a csv file (results.csv)')
    parser.add_argument('--bits', type=int, default=16, help='bit')
    parser.add_argument('--FLOPS', type=int, default=312e12)
    parser.add_argument('--MOPS',  type=int, default=1.5e12)

    args = parser.parse_args()
    
    h1 = args.h1
    h2 = args.h2
    b = args.b
    i = args.i
    l = args.l
    bit = args.bits


    FLOPS, MOPS = args.FLOPS, args.MOPS
    print(i, h1, h2, b, l, FLOPS, MOPS)


    # Self-Attention and FFN FLOPs / MOPs
    GEMM_FLOPs = (4 * (2 * b * h1**2) + 2 * 2 * b * i * h1 + 2 * 2 * b * h1 * h2)
    GEMM_MOPs = (4 * bit/32) * (4 * (2 * b*h1 + h1**2) + 2 * (b*h1 + b*h1*i + b*i) + 2 * (b*h1 + b*h2 + h1*h2))
    print("size", GEMM_FLOPs, GEMM_MOPs)

    overall_compute_intensity = GEMM_FLOPs / GEMM_MOPs
    print(f"Overall compute intensity: {overall_compute_intensity}")

    GEMM_FLOPs = GEMM_FLOPs / FLOPS
    GEMM_MOPs = GEMM_MOPs / MOPS

    # estimate DELTA FLOPs and MOPs on A100
    delta_flops = (4*b * h1) / FLOPS
    delta_mops =(bit/16) * (2 * b*h1 + b*h1*i + b*i) / MOPS

    print(f"Self-Attention and FFN FLOPs: {GEMM_FLOPs} , MOPs: {GEMM_MOPs}")
    print(f"DELTA FLOPs: {delta_flops} , MOPs: {delta_mops}")



    # store the result in a csv file
    if args.store:
        if not os.path.exists('results_perfs.csv'):
            df = pd.DataFrame(columns=['model', 'GEMM_FLOPs', 'GEMM_MOPs', 'delta_FLOPs', 'delta_MOPs', 'overall_compute_intensity'])
            df.to_csv('results_perfs.csv', index=False)

        df = pd.read_csv('results_perfs.csv')
        df = df._append({'model': args.model, 'GEMM_FLOPs': GEMM_FLOPs, 'GEMM_MOPs': GEMM_MOPs, 'delta_FLOPs': delta_flops, 'delta_MOPs': delta_mops, 'overall_compute_intensity': overall_compute_intensity}, ignore_index=True)
        df.to_csv('results_perfs.csv', index=False)
