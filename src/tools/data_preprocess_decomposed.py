import os
import os.path as osp
from os.path import join as pjoin
import argparse
import random
from tqdm import tqdm
import numpy as np
import re
import math

import codecs as cs


from multiprocessing import Pool, Queue, Manager

def safe_float(x):
    try:
        v = float(x)
        return 0.0 if math.isnan(v) else v
    except ValueError:
        return 0.0

def worker(name, root_dir, min_motion_length, data_queue):
    data_dict = {}

    try:
        motion = np.load(pjoin(root_dir, "new_joint_vecs", f"{name}.npy"), allow_pickle=True)
        if len(motion) < min_motion_length or len(motion) >= 200:
            print(f"skip {name}, len {len(motion)}")
            return

        with cs.open(pjoin(root_dir, "texts", f"{name}.txt"), encoding="utf-8") as f, \
             cs.open(pjoin(root_dir, "texts_decomposed", f"{name}.txt"), encoding="utf-8") as g:
            base_lines = [ln.strip() for ln in f if ln.strip()]

            raw = g.read().strip()
            block_strs = [blk for blk in re.split(r"\n\s*\n", raw) if blk.strip()]
            blocks = [blk.strip().splitlines() for blk in block_strs]

        if len(base_lines) != len(blocks):
            print(f"[{name}] line/block length mismatch → {len(base_lines)} vs {len(blocks)}")
            return

        for base_line, sub_lines in zip(base_lines, blocks):
            cap, tag_str, f_tag, to_tag = base_line.split("#")
            tokens = tag_str.split()
            f_tag, to_tag = float(f_tag or 0), float(to_tag or 0)

            f_tag = safe_float(f_tag)
            to_tag = safe_float(to_tag)

            text_dict = {"caption": cap, "tokens": tokens}

            decomposed = []
            for sub in sub_lines:
                s_cap, s_tag_str, *_ = sub.split("#")
                decomposed.append({"caption": s_cap,
                                   "tokens": s_tag_str.split()})

            text_dict["decomposed"] = decomposed

            if f_tag == 0.0 and to_tag == 0.0:
                key = name
                data_dict.setdefault(key, {"motion": motion,
                                           "length": len(motion),
                                           "text": []})["text"].append(text_dict)
            else:
                idx_from, idx_to = int(f_tag*20), int(to_tag*20)
                n_motion = motion[idx_from:idx_to]
                if len(n_motion) < min_motion_length or len(n_motion) >= 200:
                    continue

                key = f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVW')}_{name}"
                while key in data_dict:
                    key = f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVW')}_{name}"

                data_dict[key] = {"motion": n_motion,
                                  "length": len(n_motion),
                                  "text": [text_dict]}

    except Exception as e:
        print(f"error! {name} → {e}")

    data_queue.put(data_dict)

def process(data_dir, split, min_motion_length, args):
    print(f"processing {split}...")
    with cs.open(osp.join(data_dir, split+".txt"), "r") as f:
        id_list = [line.strip() for line in f.readlines()]
    results = {}

    with Manager() as manager:
        output_queue = manager.Queue()
        pool = Pool(processes=args.num_workers)

        with tqdm(total=len(id_list)) as pbar:
            for x in id_list:
                pool.apply_async(worker,
                                 (x, data_dir, min_motion_length, output_queue),
                                 callback=lambda _: pbar.update(1))

            pool.close()
            pool.join()

        while not output_queue.empty():
            ret = output_queue.get()
            results.update(ret)
    np.save(osp.join(data_dir, "data_"+split+".npy"), results)

def main(args):
    if args.dataset == "hml3d":
        data_dir = "./data/HumanML3D/"
        min_motion_length = 40
    elif args.dataset == "kit":
        data_dir = "./data/KIT-ML-Condition/"
        min_motion_length = 24
    else:
        raise ValueError(f"{args.dataset} not supported!")

    process(data_dir, "train", min_motion_length, args)
    process(data_dir, "val", min_motion_length, args)
    process(data_dir, "test", min_motion_length, args)

    print(f"dataset {args.dataset} preparation done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hml3d", required=True)
    parser.add_argument("--num_workers", type=int, default=64)
    args = parser.parse_args()
    main(args)