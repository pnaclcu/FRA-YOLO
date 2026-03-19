import os
from ultralytics import YOLO
import contextlib
import io
import re
from collections import defaultdict, OrderedDict


def get_model_info(model_yolo):

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        #model_yolo.fuse() # convert Conv-BN_Act three layers into cascaded one Conv_BN_Act to reduce params and flops.
        #NOTE, the Params and FLOPs are reported without fusion in our manuscript.
        model_info = model_yolo.info()  #  (layers, parameters, gradients, flops)
    layers, params, grads, flops = model_info
    #param>M，FLOPs>G
    params_m = params / 1e6
    gflops = flops / 1.0
    return f"{params_m:.2f}M", f"{gflops:,.3f}"





def format_results_table(result):



    grouped = defaultdict(dict)
    size_order = {'n': 0, 's': 1, 'm': 2, 'l': 3, 'x': 4}

    for full_name, metrics in result.items():
        #  yolov8[n/s/m/l/x]
        match = re.match(r'(.+)-(yolov8[nsmlx])', full_name)
        if match:
            dataset_name = match.group(1)
            model_type = match.group(2)  # yolov8n
            size = model_type[-1]  # n/s/m
        else:
            parts = full_name.rsplit('-', 1)
            dataset_name = parts[0]
            model_type = parts[1] if len(parts) > 1 else full_name
            size = model_type[-1] if model_type[-1] in size_order else 'z'

        grouped[dataset_name][size] = {
            'model': model_type,
            'full_name': full_name,
            **metrics
        }


    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    for dataset in sorted(grouped.keys()):
        models = grouped[dataset]
        print(f"\n【dataset: {dataset}】")
        print("-" * 100)
        print(
            f"{'Model':<10} {'Params':<10} {'GFLOPs':<10} {'P':<8} {'R':<8} {'mAP50':<10} {'mAP75':<10} {'mAP50:95':<10}")
        print("-" * 100)

        # 按 n,s,m 顺序输出
        for size in ['n', 's', 'm', 'l', 'x']:
            if size in models:
                m = models[size]

                # remove np.float64
                def fmt(val):
                    if hasattr(val, 'item'):  # numpy scalar
                        return f"{val.item():.3f}"
                    return f"{val:.3f}" if isinstance(val, float) else str(val)

                print(f"{m['model']:<10} {m['Params']:<10} {m['GFLOPs']:<10} "
                      f"{fmt(m['P']):<8} {fmt(m['R']):<8} {fmt(m['mAP50']):<10} "
                      f"{fmt(m['mAP75']):<10} {fmt(m['mAP50:95']):<10}")
        print("-" * 100)






if __name__ == '__main__':

    #####################Hyper-Params########################

    imgz = 640
    batch = 4
    conf = 0.001
    nms_thres_iou = 0.5
    device = '0'
    save = False
    show_boxes = True
    split = 'val' # optional 'test' for testset. Note that the CARPK-dataset does not include a test set
    verbose = True # False if you do not need logs
    dataset_results = {}
    #####################Hyper-Params########################

    base_dir = './ckpts'
    datasets = os.listdir(base_dir)
    for dataset in datasets:
        if not os.path.isdir(os.path.join(base_dir, dataset)): # README.md continue
            continue
        # if 'drone' not in dataset : # quick pass using visdrone
        #     continue
        if split == 'test' and dataset == 'CARPK_dataset': # CARPK-dataset does not include a test set, continue to the next dataset
            continue
        dataset_name = dataset.split('_')[0]
        dataset_yaml = '{}.yaml'.format(dataset_name)
        dataset_path = os.path.join(base_dir, dataset)
        models = os.listdir(dataset_path)

        for model in sorted(models):
            model_name ='{}-yolov8{}'.format(dataset, model)
            model_path = os.path.join(dataset_path, model,'weights/best.pt')

            model_yolo = YOLO(model_path)
            params_str, gflops_str = get_model_info(model_yolo)
            metrics = model_yolo.val(data=dataset_yaml, imgsz=imgz, batch=batch, conf=conf, iou=nms_thres_iou, device=device, save=save,
                                show_boxes=show_boxes, split=split,verbose=verbose)

            dataset_results[model_name] = {
                'Params': params_str,
                'GFLOPs': gflops_str,
                'P': round(metrics.box.mp, 3),
                'R': round(metrics.box.mr, 3),
                'mAP50': round(metrics.box.map50, 3),
                'mAP75': round(metrics.box.map75, 3),
                'mAP50:95': round(metrics.box.map, 3)
            }
    format_results_table(dataset_results)














































