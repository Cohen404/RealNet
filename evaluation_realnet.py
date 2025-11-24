import warnings
import argparse
import torch
from datasets.data_builder import build_dataloader
from easydict import EasyDict
import yaml
import os
from utils.misc_helper import set_seed
from models.model_helper import ModelHelper
from utils.eval_helper import performances
from sklearn.metrics import precision_recall_curve
import numpy as np
from utils.visualize import export_segment_images
from utils.eval_helper import Report
from train_realnet import update_config
from utils.categories import Categories


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description="evaluation RealNet")
parser.add_argument("--config", default="experiments/{}/realnet.yaml")
parser.add_argument("--dataset", default="MVTec-AD",choices=['MVTec-AD','VisA','MPDD','BTAD'])
parser.add_argument("--class_name", default="bottle",choices=[
        # mvtec-ad
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
        # visa
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
        #mpdd
        "bracket_black",
        "bracket_brown",
        "bracket_white",
        "connector",
        "metal_plate",
        "tubes",
        # btad
         "01",
         "02",
         "03",
        ] )


def main():
    args = parser.parse_args()

    class_name_list=Categories[args.dataset]

    assert args.class_name in class_name_list

    args.config=args.config.format(args.dataset)

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.exp_path = os.path.dirname(args.config)

    args.checkpoints_folder = os.path.join(config.exp_path, config.saver.checkpoints_dir,args.class_name)

    args.model_path=os.path.join(args.checkpoints_folder,"ckpt_best.pth.tar")

    config=update_config(config,args)
    set_seed(config.random_seed)

    config.evaluator.metrics['auc'].append({'name':'pro'})

    config.vis_path = os.path.join(config.exp_path, config.saver.vis_dir)
    os.makedirs(config.vis_path, exist_ok=True)

    _, val_loader = build_dataloader(config.dataset,distributed=False)

    model = ModelHelper(config.net)
    model.cuda()

    state_dict=torch.load(args.model_path)
    model.load_state_dict(state_dict['state_dict'],strict=False)

    ret_metrics = validate(config,val_loader, model,args.class_name)
    print_metrics(ret_metrics, config.evaluator.metrics, args.class_name)


def print_metrics(ret_metrics, config, class_name):
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = list(clsnames - set(["mean"]))
    clsnames.sort()

    if config.get("auc", None):
        auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
        evalnames = list(set([k.rsplit("_", 2)[1] for k in auc_keys]))
        evalnames.sort()

        record = Report(["clsname"] + evalnames)

        for clsname in clsnames:
            clsvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for evalname in evalnames
            ]
            record.add_one_record([clsname] + clsvalues)

        print(f"\n{record}")



def validate(config,val_loader, model,class_name):

    model.eval()

    fileinfos = []
    preds = []
    masks = []

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            # forward
            outputs = model(input,train=False)

            for j in range(len(outputs['filename'])):
                fileinfos.append(
                    {
                        "filename": str(outputs["filename"][j]),
                        "height": int(outputs["height"][j]),
                        "width": int(outputs["width"][j]),
                        "clsname": str(outputs["clsname"][j]),
                    }
                )
            preds.append(outputs["anomaly_score"].cpu().numpy())
            masks.append(outputs["mask"].cpu().numpy())

    # Handle variable-sized arrays by checking shapes before concatenation
    try:
        preds = np.squeeze(np.concatenate(np.asarray(preds), axis=0),axis=1)  # N x H x W
        masks = np.squeeze(np.concatenate(np.asarray(masks), axis=0),axis=1)  # N x H x W
    except ValueError:
        # If concatenation fails due to shape mismatch, process each batch separately
        processed_preds = []
        processed_masks = []
        
        for pred_batch, mask_batch in zip(preds, masks):
            # Ensure each batch has consistent shape
            if len(pred_batch.shape) == 3:  # Batch x H x W
                pred_batch = np.squeeze(pred_batch, axis=0) if pred_batch.shape[0] == 1 else pred_batch
            if len(mask_batch.shape) == 3:  # Batch x H x W
                mask_batch = np.squeeze(mask_batch, axis=0) if mask_batch.shape[0] == 1 else mask_batch
                
            processed_preds.append(pred_batch)
            processed_masks.append(mask_batch)
        
        # Now try to concatenate the processed arrays
        preds = np.concatenate(processed_preds, axis=0)
        masks = np.concatenate(processed_masks, axis=0)

    ret_metrics = performances(class_name, preds, masks, config.evaluator.metrics)

    preds_cls = []
    masks_cls = []
    image_paths = []

    for fileinfo, pred, mask in zip(fileinfos, preds, masks):
        # Ensure pred and mask have consistent shapes
        if len(pred.shape) == 2:  # H x W
            pred = pred[None, ...]  # Add batch dimension
        if len(mask.shape) == 2:  # H x W
            mask = mask[None, ...]  # Add batch dimension
            
        preds_cls.append(pred)
        masks_cls.append(mask)
        image_paths.append(fileinfo['filename'])

    try:
        preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)  # N x H x W
        masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)  # N x H x W
    except ValueError:
        # If concatenation still fails, try to handle shape mismatches
        print("Warning: Shape mismatch in preds_cls or masks_cls, attempting to fix...")
        # Find the maximum dimensions
        max_h = max(p.shape[1] for p in preds_cls)
        max_w = max(p.shape[2] for p in preds_cls)
        
        # Pad all arrays to the same size
        padded_preds = []
        padded_masks = []
        
        for pred, mask in zip(preds_cls, masks_cls):
            # Pad prediction
            if pred.shape[1] < max_h or pred.shape[2] < max_w:
                padded_pred = np.zeros((pred.shape[0], max_h, max_w))
                padded_pred[:, :pred.shape[1], :pred.shape[2]] = pred
                padded_preds.append(padded_pred)
            else:
                padded_preds.append(pred)
                
            # Pad mask
            if mask.shape[1] < max_h or mask.shape[2] < max_w:
                padded_mask = np.zeros((mask.shape[0], max_h, max_w))
                padded_mask[:, :mask.shape[1], :mask.shape[2]] = mask
                padded_masks.append(padded_mask)
            else:
                padded_masks.append(mask)
        
        preds_cls = np.concatenate(padded_preds, axis=0)
        masks_cls = np.concatenate(padded_masks, axis=0)
    masks_cls[masks_cls != 0.0] = 1.0

    precision, recall, thresholds = precision_recall_curve(masks_cls.flatten(), preds_cls.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    seg_threshold = thresholds[np.argmax(f1)]
    export_segment_images(config, image_paths, masks_cls, preds_cls, seg_threshold, class_name)
    return ret_metrics


if __name__ == "__main__":
    main()
