import torch_pruning as tp

from models.common import *
from models.yolo import Detect
from utils.torch_utils import select_device
from models.yolov5 import DecoupledDetect


def layer_pruning(weights):
    device = select_device('cpu')
    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)

    for para in model.parameters():
        para.requires_grad = True

    example_inputs = torch.randn(1, 3, 640, 640).to(device)
    # 选择评估方法
    imp = tp.importance.MagnitudeImportance(p=2) # L2 norm pruning

    # 忽略无需剪枝的层
    ignored_layers = []
    for layer_id, layer in enumerate(model.model.model):
        if layer_id <= 9:
            ignored_layers.append(layer)
            continue
        if isinstance(layer, (Detect, DecoupledDetect)):
            ignored_layers.append(layer)
            continue
    
    # 初始化剪枝器
    iterative_steps = 1 # 迭代式剪枝，重复多少次次Pruning-Finetuning的循环完成剪枝。
    pruner = tp.pruner.MagnitudePruner(
        model.model,
        example_inputs, # 用于分析依赖的伪输入
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=0.5, # 目标稀疏性，移除多少的通道
        ignored_layers=ignored_layers,
    )

    # Pruning-Finetuning的循环
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for g in pruner.step(interactive=True):
        # print(g)
        g.prune()

    pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    # print(model)
    print("Before Pruning: MACs=%f G, #Params=%f G"%(base_macs/1e9, base_nparams/1e9))
    print("After Pruning: MACs=%f G, #Params=%f G"%(pruned_macs/1e9, pruned_nparams/1e9))
    ####################################################################################
    save_path = Path(weights).absolute()
    stem = save_path.stem
    save_path = save_path.with_name(stem+"_pruning.pt")
    
    model_ = {
        'model': model.model.half(),
        'ema': model.model,
    }
    print("save_path: ", save_path)
    torch.save(model_, str(save_path))

if __name__ == "__main__":
    layer_pruning('/home/mqr/Desktop/person3/test/yolov5n.pt')
