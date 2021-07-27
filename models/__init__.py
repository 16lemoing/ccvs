import torch
import os
from glob import glob

def save_network(net, label, which_iter, opt, latest=False, best=False):
    if net is None:
        return

    assert not (latest and best), "Either 'latest' or 'best' but both were given"

    if latest:
        save_path = os.path.join(opt.checkpoint_path, f'{label}_latest_net_{which_iter}.pth')
        old_paths = glob(os.path.join(opt.checkpoint_path, f"{label}_latest_net_*.pth"))
    elif best:
        save_path = os.path.join(opt.checkpoint_path, f'{label}_best_net_{which_iter}.pth')
        old_paths = glob(os.path.join(opt.checkpoint_path, f"{label}_best_net_*.pth"))
    else:
        save_path = os.path.join(opt.checkpoint_path, f'{label}_net_{which_iter}.pth')
        old_paths = None

    torch.save(net.state_dict(), save_path)

    if old_paths is not None:
        for old_path in old_paths:
            open(old_path, 'w').close()
            os.unlink(old_path)

def load_state_dict(net, state_dict, strict=True, block_delta=None):
    if block_delta is not None:
        new_dict = {}
        for key in state_dict:
            if "blocks" in key:
                pre, post = key.split("blocks")
                post_list = post.split('.')
                post_list[1] = str(int(post_list[1]) + block_delta)
                post = '.'.join(post_list)
                new_key = pre + "blocks" + post
                print(f"Renaming {key} to {new_key}")
            else:
                new_key = key
            new_dict[new_key] = state_dict[key]
        state_dict = new_dict

    if not strict:
        model_dict = net.state_dict()
        pop_list = []
        # remove the keys which don't match in size
        for key in state_dict:
            if key in model_dict:
                if model_dict[key].shape != state_dict[key].shape:
                    pop_list.append(key)
                    print(f"Size mismatch for {key}")
            else:
                pop_list.append(key)
                print(f"Key missing in model for {key}")
        for key in pop_list:
            state_dict.pop(key)
        model_dict.update(state_dict)
        net.load_state_dict(model_dict)
    else:
        net.load_state_dict(state_dict)

def load_network(net, label, opt, override_iter=None, override_load_path=None, required=True, head_to_n=0, block_delta=None):
    if net is None:
        return None

    which_iter = override_iter if override_iter is not None else opt.which_iter
    load_path = override_load_path if override_load_path is not None else opt.load_path

    if load_path is not None and which_iter != "0":
        if which_iter == "latest":
            load_paths = glob(os.path.join(load_path, f"{label}_latest_net_*.pth"))
            assert len(load_paths) > 0, f"Did not find any checkpoint for {label} net at latest iter and path {load_path}"
            assert len(load_paths) == 1
            load_path = load_paths[0]
        elif which_iter == "best":
            load_paths = glob(os.path.join(load_path, f"{label}_best_net_*.pth"))
            assert len(load_paths) > 0, f"Did not find any checkpoint for {label} net at best iter and path {load_path}"
            assert len(load_paths) == 1
            load_path = load_paths[0]
        else:
            latest_load_path = os.path.join(load_path, f'{label}_latest_net_{which_iter}.pth')
            best_load_path = os.path.join(load_path, f'{label}_best_net_{which_iter}.pth')
            load_path = os.path.join(load_path, f'{label}_net_{which_iter}.pth')
            if not os.path.exists(load_path):
                if not os.path.exists(latest_load_path):
                    if not os.path.exists(best_load_path):
                        if required:
                            raise ValueError(f"No checkpoint for {label} net at iter {which_iter} and path {load_path}")
                        else:
                            print(f"No checkpoint for {label} net at iter {which_iter} and path {load_path}")
                            print(f"Loading untrained {label} net")
                            return net
                    else:
                        load_path = best_load_path
                else:
                    load_path = latest_load_path
        state_dict = torch.load(load_path)
        if head_to_n != 0:
            print(f"Copying head weights, from 1 to {head_to_n} proposals with randomly initialized associated logits")
            h_weight = state_dict["head.weight"]
            s0, s1 = h_weight.shape
            mask = torch.ones((s0 + 1) * head_to_n).to(h_weight.get_device())
            mask[::s0 + 1] = 0
            new_h_weight = torch.rand(head_to_n * (s0 + 1), s1).to(h_weight.get_device())
            new_h_weight[mask.bool()] = h_weight.unsqueeze(0).repeat(head_to_n, 1, 1).view(head_to_n * s0, s1)
            state_dict["head.weight"] = new_h_weight
        load_state_dict(net, state_dict, strict=not opt.not_strict, block_delta=block_delta)
        print(f"Loading checkpoint for {label} net from {load_path}")

    elif opt.cont_train and which_iter != "0":
        assert which_iter not in ["latest", "best"], "If load_path is not specified, which_iter should be an int"
        load_paths = glob(os.path.join(opt.save_path, "checkpoints", f"*-{opt.name}", f"{label}_latest_net_{which_iter}.pth"))
        load_paths += glob(os.path.join(opt.save_path, "checkpoints", f"*-{opt.name}", f"{label}_best_net_{which_iter}.pth"))
        load_paths += glob(os.path.join(opt.save_path, "checkpoints", f"*-{opt.name}", f"{label}_net_{which_iter}.pth"))
        assert len(load_paths) > 0, f"Did not find any checkpoint for {label} net at iter {which_iter} and name {opt.name}"
        assert len(load_paths) == 1, f"Too many checkpoint candidates for {label} net at iter {which_iter} and name {opt.name}:\n{load_paths}"
        load_path = load_paths[0]
        load_state_dict(net, torch.load(load_path), strict=not opt.not_strict)
        print(f"Loading checkpoint for {label} net from {load_path}")

    else:
        print(f"Loading untrained {label} net")

    return net

def print_network(net):
    if net is not None:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)