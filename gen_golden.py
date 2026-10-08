#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
axis_cnn_top 完整 CNN 推理链路黄金向量生成脚本
Conv(3x3, 1->6ch) -> BN -> Pool(2x2) -> ReLU -> FC(54->10)
与 RTL 计算方式完全一致
"""
import random
import os

SEED = 42
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 网络参数
IMG_W = 8
CONV_OUT_W = 6
POOL_OUT_W = 3
CONV_OUT_CH = 6
CONV_KERNEL = 3
FC_IN_WIDTH = POOL_OUT_W * POOL_OUT_W * CONV_OUT_CH  # 54
FC_OUT_WIDTH = 10
SHIFT_BITS = 4

def signed_to_hex(val, width=8):
    """有符号整数转十六进制字符串"""
    if val < 0:
        val = val + (1 << width)
    return f"{val:0{width//4}X}"

def saturate(val, out_w=8):
    """饱和截断到有符号 out_w bit"""
    max_val = (1 << (out_w - 1)) - 1
    min_val = -(1 << (out_w - 1))
    if val > max_val:
        return max_val
    elif val < min_val:
        return min_val
    else:
        return val

def conv2d(input_img, weights, biases, in_ch=1, out_ch=6, kernel=3, img_w=8, out_w=6):
    """2D卷积，窗口优先输出"""
    output = []
    for row in range(out_w):
        for col in range(out_w):
            for ch in range(out_ch):
                acc = biases[ch]
                for kr in range(kernel):
                    for kc in range(kernel):
                        in_val = input_img[row + kr][col + kc]
                        w_val = weights[ch * kernel * kernel + kr * kernel + kc]
                        acc += in_val * w_val
                result = saturate(acc >> SHIFT_BITS, 8)
                output.append(result)
    return output

def batch_norm(input_data, alphas, betas, channels=6):
    """批量归一化（推理阶段简化为逐通道线性变换）"""
    output = []
    for i in range(len(input_data)):
        ch = i % channels
        acc = input_data[i] * alphas[ch] + betas[ch]
        result = saturate(acc >> SHIFT_BITS, 8)
        output.append(result)
    return output

def maxpool2d(input_data, channels=6, img_w=6, out_w=3):
    """2x2最大池化，窗口优先输出"""
    img = []
    idx = 0
    for row in range(img_w):
        row_data = []
        for col in range(img_w):
            ch_data = input_data[idx:idx+channels]
            row_data.append(ch_data)
            idx += channels
        img.append(row_data)

    output = []
    for row in range(out_w):
        for col in range(out_w):
            for ch in range(channels):
                v00 = img[row*2][col*2][ch]
                v01 = img[row*2][col*2+1][ch]
                v10 = img[row*2+1][col*2][ch]
                v11 = img[row*2+1][col*2+1][ch]
                max_val = max(v00, v01, v10, v11)
                output.append(max_val)
    return output

def relu(input_data):
    """ReLU激活函数"""
    return [max(0, x) for x in input_data]

def fully_connected(input_data, weights, biases, in_width=54, out_width=10):
    """全连接层"""
    output = []
    for out_idx in range(out_width):
        acc = biases[out_idx]
        for in_idx in range(in_width):
            in_val = input_data[in_idx]
            w_val = weights[out_idx * in_width + in_idx]
            acc += in_val * w_val
        result = saturate(acc >> SHIFT_BITS, 8)
        output.append(result)
    return output

def main():
    random.seed(SEED)

    #==================== 生成随机数据 ====================
    # 输入：64个，8bit有符号，reshape成8x8
    inputs_flat = [random.randint(-128, 127) for _ in range(IMG_W * IMG_W)]
    input_img = [inputs_flat[row*IMG_W:(row+1)*IMG_W] for row in range(IMG_W)]

    # 卷积权重：54个（6通道×9）
    conv_weights = [random.randint(-128, 127) for _ in range(CONV_OUT_CH * CONV_KERNEL * CONV_KERNEL)]

    # 卷积偏置：6个，16bit
    conv_biases = [random.randint(-32768, 32767) for _ in range(CONV_OUT_CH)]

    # BN alpha：6个，8bit
    bn_alphas = [random.randint(-64, 64) for _ in range(CONV_OUT_CH)]

    # BN beta：6个，16bit
    bn_betas = [random.randint(-1000, 1000) for _ in range(CONV_OUT_CH)]

    # FC权重：540个（10通道×54）
    fc_weights = [random.randint(-128, 127) for _ in range(FC_IN_WIDTH * FC_OUT_WIDTH)]

    # FC偏置：10个，16bit
    fc_biases = [random.randint(-32768, 32767) for _ in range(FC_OUT_WIDTH)]

    #==================== 五级计算 ====================
    print("[STEP 1/5] Computing Conv (3x3, 1->6ch)...")
    conv_out = conv2d(input_img, conv_weights, conv_biases)
    print(f"  Conv output: {len(conv_out)} values")

    print("[STEP 2/5] Computing Batch Normalization...")
    bn_out = batch_norm(conv_out, bn_alphas, bn_betas)
    print(f"  BN output: {len(bn_out)} values")

    print("[STEP 3/5] Computing MaxPool (2x2)...")
    pool_out = maxpool2d(bn_out)
    print(f"  Pool output: {len(pool_out)} values")

    print("[STEP 4/5] Computing ReLU...")
    relu_out = relu(pool_out)
    print(f"  ReLU output: {len(relu_out)} values")

    print("[STEP 5/5] Computing FC (54->10)...")
    fc_out = fully_connected(relu_out, fc_weights, fc_biases)
    print(f"  FC output: {len(fc_out)} values")

    #==================== 写入 hex 文件 ====================
    with open(os.path.join(OUT_DIR, "input.hex"), "w") as f:
        for val in inputs_flat:
            f.write(signed_to_hex(val, 8) + "\n")
    print(f"[INFO] Wrote {len(inputs_flat)} inputs")

    with open(os.path.join(OUT_DIR, "conv_weights.hex"), "w") as f:
        for val in conv_weights:
            f.write(signed_to_hex(val, 8) + "\n")
    print(f"[INFO] Wrote {len(conv_weights)} conv weights")

    with open(os.path.join(OUT_DIR, "conv_biases.hex"), "w") as f:
        for val in conv_biases:
            f.write(signed_to_hex(val, 16) + "\n")
    print(f"[INFO] Wrote {len(conv_biases)} conv biases")

    with open(os.path.join(OUT_DIR, "bn_alpha.hex"), "w") as f:
        for val in bn_alphas:
            f.write(signed_to_hex(val, 8) + "\n")
    print(f"[INFO] Wrote {len(bn_alphas)} BN alphas")

    with open(os.path.join(OUT_DIR, "bn_beta.hex"), "w") as f:
        for val in bn_betas:
            f.write(signed_to_hex(val, 16) + "\n")
    print(f"[INFO] Wrote {len(bn_betas)} BN betas")

    with open(os.path.join(OUT_DIR, "fc_weights.hex"), "w") as f:
        for val in fc_weights:
            f.write(signed_to_hex(val, 8) + "\n")
    print(f"[INFO] Wrote {len(fc_weights)} FC weights")

    with open(os.path.join(OUT_DIR, "fc_biases.hex"), "w") as f:
        for val in fc_biases:
            f.write(signed_to_hex(val, 16) + "\n")
    print(f"[INFO] Wrote {len(fc_biases)} FC biases")

    with open(os.path.join(OUT_DIR, "golden_output.hex"), "w") as f:
        for val in fc_out:
            f.write(signed_to_hex(val, 8) + "\n")
    print(f"[INFO] Wrote {len(fc_out)} golden outputs")

    #==================== 打印结果 ====================
    print("\n" + "="*50)
    print("Final FC outputs (golden):")
    print(f"{'idx':>4} {'value':>8}")
    for i in range(FC_OUT_WIDTH):
        print(f"{i:>4} {fc_out[i]:>8}")

    # 统计
    pos_cnt = sum(1 for x in fc_out if x > 0)
    neg_cnt = sum(1 for x in fc_out if x < 0)
    zero_cnt = sum(1 for x in fc_out if x == 0)
    print(f"\n[STAT] positive={pos_cnt}, negative={neg_cnt}, zero={zero_cnt}")

    # 中间结果统计
    conv_pos = sum(1 for x in conv_out if x > 0)
    bn_pos = sum(1 for x in bn_out if x > 0)
    relu_nonzero = sum(1 for x in relu_out if x > 0)
    print(f"[STAT] Conv positive: {conv_pos}/{len(conv_out)}")
    print(f"[STAT] BN positive: {bn_pos}/{len(bn_out)}")
    print(f"[STAT] ReLU non-zero: {relu_nonzero}/{len(relu_out)}")

if __name__ == "__main__":
    main()
