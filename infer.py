import os
import threading
from threading import Lock

import numpy as np
import torch
from tqdm import tqdm
import warnings
import argparse
import time
import math
from queue import Queue
from models.vfi import VFI
from models.utils.tools import *

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Interpolation a video with ReconDeduplicate')
    parser.add_argument('-i', '--video', dest='video', type=str, required=True, help='absolute path of input video')
    parser.add_argument('-o', '--video_output', dest='video_output', required=True, type=str, default='output',
                        help='absolute path of output video')
    parser.add_argument('-mrl', '--max_recon_len', dest='max_recon_len', type=int, default=4,
                        help='the value of parameter max_recon_len')
    parser.add_argument('-fps', '--target_fps', dest='target_fps', type=float, default=60, help='interpolate to ? fps')
    parser.add_argument('-t', '--times', dest='times', type=int, default=-1, help='interpolate to ?x fps')
    parser.add_argument('-m', '--model_type', dest='model_type', type=str, default='gmfss',
                        help='the interpolation model to use (gmfss/rife/gimm)')
    parser.add_argument('-s', '--enable_scdet', dest='enable_scdet', action='store_true', default=False,
                        help='enable scene change detection')
    parser.add_argument('-st', '--scdet_threshold', dest='scdet_threshold', type=float, default=0.3,
                        help='ssim scene detection threshold')
    parser.add_argument('-dr', '--dup_res', dest='dup_res', type=float, default=0.95,
                        help='ssim duplicate frame detection threshold')
    parser.add_argument('-scale', '--scale', dest='scale', type=float, default=1.0,
                        help='flow scale, generally use 1.0 with 1080P and 0.5 with 4K resolution')
    parser.add_argument('-hw', '--hwaccel', dest='hwaccel', action='store_true', default=True,
                        help='enable hardware acceleration encode(require nvidia graph card)')
    return parser.parse_args()


def load_model():
    model = VFI(
        model_type=model_type,
        weights='weights',
        scale=scale,
        device=device
    )

    return model


def linear_recon(recon_list, value_list):
    """
    将非线性的帧序列重建为从头到尾线性的帧序列(选择性的使用中间 1 ~ n-1 帧的信息而不是只和头尾的0,n帧有关)
    :param recon_list: 存放待重建帧的列表, 如[I0, I1, I2, I3]
    :param value_list: 存放帧差, 如[0.997, 0.992, 0.998]
    :return: 重建后的帧序列, 如[I0, nI1, nI2, I3], 重建后预期帧差[0.9957, 0.9957, 0.9957]
    """
    assert len(recon_list) == len(value_list) + 1

    # ssim时应该先做反转
    for i in range(len(value_list)):
        value_list[i] = 1 - value_list[i]

    acc_value_list = np.cumsum([0] + value_list)  # 累加帧差
    ap = np.linspace(0, acc_value_list[-1], len(value_list) + 1)  # 等差帧差
    ap = ap[1:-1]  # 掐头去尾, 保证头尾不被重建

    if ap.size == 0:  # 无法重建
        return recon_list

    after_recon_list = [recon_list[0]]  # 头尾不被重建

    for i in range(len(ap)):
        for j in range(len(acc_value_list) - 1):
            if acc_value_list[j] <= ap[i] <= acc_value_list[j + 1]:
                t = (ap[i] - acc_value_list[j]) / (acc_value_list[j + 1] - acc_value_list[j])
                after_recon_list.append(model.gen_ts_frame(recon_list[j], recon_list[j + 1], [t])[0])
                break  # 避免重复补偿

    after_recon_list.append(recon_list[-1])   # 头尾不被重建

    return after_recon_list


def run():
    inp_queue = video_io.read_buffer
    out_queue = video_io.write_buffer

    i0 = inp_queue.get()
    if i0 is None:
        raise ValueError(f"video doesn't contains enough frames for interpolation")

    size = get_valid_net_inp_size(i0, model.scale, div=model.pad_size)
    src_size, dst_size = size['src_size'], size['dst_size']

    idx = 0
    I0 = to_inp(i0, dst_size)
    recon_list = [I0]
    value_list = list()
    out_queue.put(i0)
    while True:
        i1 = inp_queue.get()
        if i1 is None:
            ts = mapper.get_range_timestamps(idx, idx + 1, lclose=True, rclose=True, normalize=True)
            if 1 in ts:
                out_queue.put(i0)
            idx += 1

            out_queue.put(None)  # end sign
            pbar.update(1)
            break

        I1 = to_inp(i1, dst_size)

        scene_change, value = check_scene(I0, I1, scdet_threshold=scdet_threshold, return_value=True)

        if scene_change and enable_scdet:
            if len(recon_list) < 2:
                ts = mapper.get_range_timestamps(idx, idx + 1, lclose=True, rclose=False, normalize=True)
                for _ in ts:
                    out_queue.put(i0)  # 转场处理
                recon_list = [I1]
                value_list.clear()
                idx += 1
            else:
                # 先重建列表中的内容
                recon_list = linear_recon(recon_list, value_list)
                for i in range(len(recon_list) - 1):
                    ts = mapper.get_range_timestamps(idx, idx + 1, lclose=True, rclose=False, normalize=True)
                    output = model.gen_ts_frame(recon_list[i], recon_list[i + 1], ts)
                    for item in output:
                        out_queue.put(to_out(item, src_size))

                    idx += 1

                # 再处理转场帧
                ts = mapper.get_range_timestamps(idx, idx + 1, lclose=True, rclose=False, normalize=True)
                for _ in ts:
                    out_queue.put(to_out(recon_list[-1], src_size))  # 转场处理

                idx += 1
                recon_list = [I1]
                value_list.clear()
        else:
            # 这里是ssim，如果使用absdiff则用value >= dup_res
            if len(recon_list) + 1 > max_recon_len or value <= dup_res:
                if len(recon_list) < 2:  # 说明不需要重建, 直接补
                    ts = mapper.get_range_timestamps(idx, idx + 1, lclose=True, rclose=False, normalize=True)
                    output = model.gen_ts_frame(I0, I1, ts)
                    for item in output:
                        out_queue.put(to_out(item, src_size))
                    idx += 1

                    recon_list = [I1]
                    value_list.clear()
                else:
                    # 先重建
                    recon_list.append(I1)
                    value_list.append(value)
                    recon_list = linear_recon(recon_list, value_list)
                    # print(f"{idx} 先重建后正常补帧 {len(recon_list)}")

                    # 后正常补帧
                    for i in range(len(recon_list) - 1):
                        ts = mapper.get_range_timestamps(idx, idx + 1, lclose=True, rclose=False, normalize=True)
                        output = model.gen_ts_frame(recon_list[i], recon_list[i + 1], ts)
                        for item in output:
                            out_queue.put(to_out(item, src_size))
                        idx += 1

                    recon_list = [I1]
                    value_list.clear()
            else:
                recon_list.append(I1)
                value_list.append(value)

        i0 = i1
        I0 = I1
        pbar.update(1)


if __name__ == '__main__':
    args = parse_args()
    model_type = args.model_type
    max_recon_len = args.max_recon_len
    dup_res = args.dup_res
    target_fps = args.target_fps
    times = args.times  # interpolation ratio >= 2
    enable_scdet = args.enable_scdet  # enable scene change detection
    scdet_threshold = args.scdet_threshold  # scene change detection threshold
    video = args.video  # input video path
    video_output = args.video_output  # output img dir
    scale = args.scale  # flow scale
    hwaccel = args.hwaccel  # Use hardware acceleration video encoder

    assert model_type in ['gmfss', 'rife', 'gimm'], f"not implement the model {model_type}"

    model = load_model()

    if not os.path.exists(video):
        raise FileNotFoundError(f"can't find the file {video}")

    video_io = VideoFI_IO(video, video_output, dst_fps=target_fps, times=times, hwaccel=hwaccel)

    src_fps = video_io.src_fps
    if target_fps <= src_fps:
        raise ValueError(f'dst fps should be greater than src fps, but got tar_fps={target_fps} and src_fps={src_fps}')

    pbar = tqdm(total=video_io.total_frames_count)
    mapper = TMapper(src_fps, target_fps, times)

    run()

    print('Wait for all frames to be exported...')
    while not video_io.finish_writing():
        time.sleep(0.1)

    print('Done!')
