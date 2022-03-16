import torch
import numpy as np

def weight_compute(w):
    gain = 1.       # fixed
    lr_mul = 1.     # fixed

    # fan caluate
    if (w.ndim == 5):
        w = w[0]  # fan 계산을 위해 4차원으로 변환
    num_input_fmaps = w.size(1)
    num_output_fmaps = w.size(0)
    receptive_field_size = 1
    if w.dim() > 2:
        receptive_field_size = w[0][0].numel()  # kernel_h * kernel_w
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    # 'mode = fan_in' use
    w = w * torch.tensor(gain, device=w.device) * torch.sqrt(
        torch.tensor(1. / fan_in, device=w.device)) * lr_mul

    return w

def extract_weight(CHECK_POINT, WEIGHTS_PATH):
    weight_list = [(key, value) for (key, value) in CHECK_POINT['state_dict'].items()]

    up_conv = [(key, value) for i in range(0, 12, 2) for (key, value) in CHECK_POINT['state_dict'].items() if
               "generator.convs.%d." % i in key]
    conv = [(key, value) for i in range(1, 12, 2) for (key, value) in CHECK_POINT['state_dict'].items() if
            "generator.convs.%d." % i in key]
    fusion_out = [(key, value) for (key, value) in CHECK_POINT['state_dict'].items() if "generator.fusion_out." in key]
    fusion_skip = [(key, value) for (key, value) in CHECK_POINT['state_dict'].items() if
                   "generator.fusion_skip." in key]
    to_rgbs = [(key, value) for (key, value) in CHECK_POINT['state_dict'].items() if "generator.to_rgbs." in key]

    loop_list = []
    for loop in range(6):
        if (loop < 4):
            for i in range(2):  # fusion_out
                loop_list += [fusion_out[2 * loop + i]]

            for i in range(2):  # fusion_skip
                loop_list += [fusion_skip[2 * loop + i]]

        for i in range(6):  # up_conv
            loop_list += [up_conv[6 * loop + i]]

        for i in range(5):  # conv
            loop_list += [conv[5 * loop + i]]

        for i in range(5):  # to_rgbs
            loop_list += [to_rgbs[5 * loop + i]]

    with open(WEIGHTS_PATH, 'wb') as f:
        dummy = np.array([0] * 10, dtype=np.float32)
        f.write(dummy)  # dummy 10 line

        for idx in range(13):  # injected_noise
            key, value = weight_list[idx]
            w = value.cpu().data.numpy()
            w.tofile(f)
            print(0, idx, key, w.shape)

        for idx in range(135, 847):  # encoder
            key, w = weight_list[idx]
            if (key in 'num_batches_tracked'):
                print(idx, "--------------------")
                continue

            # compute_weight
            if ("_orig" in key):
                w = weight_compute(w)

            if len(w.shape) == 2:
                w = w.transpose(1, 0)
                w = w.cpu().data.numpy()
            else:
                w = w.cpu().data.numpy()

            w.tofile(f)
            print(0, idx, key, w.shape)

        for idx in range(29, 39):  # constant_input, conv1, to_rgb1
            key, w = weight_list[idx]
            if (key in 'num_batches_tracked'):
                print(idx, "--------------------")
                continue

            # compute_weight
            if ("_orig" in key):
                w = weight_compute(w)

            if len(w.shape) == 2:
                w = w.transpose(1, 0)
                w = w.cpu().data.numpy()
            else:
                w = w.cpu().data.numpy()

            w.tofile(f)
            print(0, idx, key, w.shape)

        for idx in range(len(loop_list)):  # gernerator loop
            key, w = loop_list[idx]
            if (key in 'num_batches_tracked'):
                print(idx, "--------------------")
                continue

            # compute_weight
            if ("_orig" in key):
                w = weight_compute(w)

            if len(w.shape) == 2:
                w = w.transpose(1, 0)
                w = w.cpu().data.numpy()
            else:
                w = w.cpu().data.numpy()

            w.tofile(f)
            print(0, idx, key, w.shape)

        for idx in range(863, 873):  # decoder
            key, w = weight_list[idx]
            if (key in 'num_batches_tracked'):
                print(idx, "--------------------")
                continue

            # compute_weight
            if ("_orig" in key):
                w = weight_compute(w)

            if len(w.shape) == 2:
                w = w.transpose(1, 0)
                w = w.cpu().data.numpy()
            else:
                w = w.cpu().data.numpy()

            w.tofile(f)
            print(0, idx, key, w.shape)

    print("============== Weight Extract Doen! ==============")
    
    
if __name__ == '__main__':
    # load model path
    MODEL_PATH = './weights/glean_cat_8x.pth'
    CHECK_POINT = torch.load(MODEL_PATH)

    # extracted weight path
    WEIGHTS_PATH = '../mgmt/weights/glean.weights'  # Don't modify it.
    
    extract_weight(CHECK_POINT, WEIGHTS_PATH)
    


