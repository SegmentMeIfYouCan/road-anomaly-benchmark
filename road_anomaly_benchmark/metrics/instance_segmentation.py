
import dataclasses

import numpy as np

from easydict import EasyDict
from scipy.ndimage.measurements import label
from pathlib import Path

from .base import EvaluationMetric, MetricRegistry
from ..evaluation import DIR_OUTPUTS
from ..datasets.dataset_io import hdf5_write_hierarchy_to_file, hdf5_read_hierarchy_from_file
from ..jupyter_show_image import adapt_img_data, imwrite, imread


def default_instancer(anomaly_p: np.ndarray, label_pixel_gt: np.ndarray, thresh_p: float,
                      thresh_segsize: int, thresh_instsize: int = 0):

    """segmentation from pixel-wise anoamly scores"""
    segmentation = np.copy(anomaly_p)
    segmentation[anomaly_p > thresh_p] = 1
    segmentation[anomaly_p <= thresh_p] = 0

    anomaly_gt = np.zeros(label_pixel_gt.shape)
    anomaly_gt[label_pixel_gt == 1] = 1
    anomaly_pred = np.zeros(label_pixel_gt.shape)
    anomaly_pred[segmentation == 1] = 1
    anomaly_pred[label_pixel_gt == 255] = 0

    """connected components"""
    structure = np.ones((3, 3), dtype=np.uint8)
    anomaly_instances, n_anomaly = label(anomaly_gt, structure)
    anomaly_seg_pred, n_seg_pred = label(anomaly_pred, structure)

    """remove connected components below size threshold"""
    if thresh_segsize is not None:
        minimum_cc_sum  = thresh_segsize
        labeled_mask = np.copy(anomaly_seg_pred)
        for comp in range(n_seg_pred+1):
            if len(anomaly_seg_pred[labeled_mask == comp]) < minimum_cc_sum:
                anomaly_seg_pred[labeled_mask == comp] = 0
    labeled_mask = np.copy(anomaly_instances)
    label_pixel_gt = label_pixel_gt.copy() # copy for editing
    for comp in range(n_anomaly + 1):
        if len(anomaly_instances[labeled_mask == comp]) < thresh_instsize:
            label_pixel_gt[labeled_mask == comp] = 255

    """restrict to region of interest"""
    mask_roi = label_pixel_gt < 255
    segmentation_filtered = np.copy(anomaly_seg_pred).astype("uint8")
    segmentation_filtered[anomaly_seg_pred>0] = 1
    segmentation_filtered[mask_roi==255] = 0

    return anomaly_instances[mask_roi], anomaly_seg_pred[mask_roi], segmentation_filtered


def anomaly_instances_from_mask(segmentation: np.ndarray, label_pixel_gt: np.ndarray, thresh_instsize: int = 0):
    anomaly_gt = np.zeros(label_pixel_gt.shape)
    anomaly_gt[label_pixel_gt == 1] = 1
    anomaly_pred = np.zeros(label_pixel_gt.shape)
    anomaly_pred[segmentation == 1] = 1
    anomaly_pred[label_pixel_gt == 255] = 0

    """connected components"""
    structure = np.ones((3, 3), dtype=np.uint8)
    anomaly_instances, n_anomaly = label(anomaly_gt, structure)
    anomaly_seg_pred, n_seg_pred = label(anomaly_pred, structure)

    """remove ground truth connected components below size threshold"""
    labeled_mask = np.copy(anomaly_instances)
    label_pixel_gt = label_pixel_gt.copy() # copy for editing
    for comp in range(n_anomaly + 1):
        if len(anomaly_instances[labeled_mask == comp]) < thresh_instsize:
            label_pixel_gt[labeled_mask == comp] = 255

    """restrict to region of interest"""
    mask_roi = label_pixel_gt < 255
    return anomaly_instances[mask_roi], anomaly_seg_pred[mask_roi]


def save_anomaly_mask(anomaly_p: np.ndarray, thresh_p: float, save_path: str):

    """segmentation from pixel-wise anomaly scores"""
    segmentation = np.copy(anomaly_p)
    segmentation[anomaly_p > thresh_p] = 1
    segmentation[anomaly_p <= thresh_p] = 0

    # Image.fromarray(segmentation.astype('uint8')).save(save_path)
    imwrite(save_path, segmentation.astype('uint8'))
    print("Saved:", save_path)


def segment_metrics(anomaly_instances, anomaly_seg_pred, iou_thresholds=np.linspace(0.25, 0.75, 11, endpoint=True)):
    """
    function that computes the segments metrics based on the adjusted IoU
    anomaly_instances: (numpy array) anomaly instance annoation
    anomaly_seg_pred: (numpy array) anomaly instance prediction
    iou_threshold: (float) threshold for true positive
    """

    """Loop over ground truth instances"""
    sIoU_gt = []
    size_gt = []

    for i in np.unique(anomaly_instances[anomaly_instances>0]):
        tp_loc = anomaly_seg_pred[anomaly_instances == i]
        seg_ind = np.unique(tp_loc[tp_loc != 0])

        """calc area of intersection"""
        intersection = len(tp_loc[np.isin(tp_loc, seg_ind)])
        adjustment = len(
            anomaly_seg_pred[np.logical_and(~np.isin(anomaly_instances, [0, i]), np.isin(anomaly_seg_pred, seg_ind))])

        adjusted_union = np.sum(np.isin(anomaly_seg_pred, seg_ind)) + np.sum(
            anomaly_instances == i) - intersection - adjustment
        sIoU_gt.append(intersection / adjusted_union)
        size_gt.append(np.sum(anomaly_instances == i))

    """Loop over prediction instances"""
    sIoU_pred = []
    size_pred = []
    prec_pred = []
    for i in np.unique(anomaly_seg_pred[anomaly_seg_pred>0]):
        tp_loc = anomaly_instances[anomaly_seg_pred == i]
        seg_ind = np.unique(tp_loc[tp_loc != 0])
        intersection = len(tp_loc[np.isin(tp_loc, seg_ind)])
        adjustment = len(
            anomaly_instances[np.logical_and(~np.isin(anomaly_seg_pred, [0, i]), np.isin(anomaly_instances, seg_ind))])
        adjusted_union = np.sum(np.isin(anomaly_instances, seg_ind)) + np.sum(
            anomaly_seg_pred == i) - intersection - adjustment
        sIoU_pred.append(intersection / adjusted_union)
        size_pred.append(np.sum(anomaly_seg_pred == i))
        prec_pred.append(intersection / np.sum(anomaly_seg_pred == i))

    sIoU_gt = np.array(sIoU_gt)
    sIoU_pred = np.array(sIoU_pred)
    size_gt = np.array((size_gt))
    size_pred = np.array(size_pred)
    prec_pred = np.array(prec_pred)

    """create results dictionary"""
    results = EasyDict(sIoU_gt=sIoU_gt, sIoU_pred=sIoU_pred, size_gt=size_gt, size_pred=size_pred, prec_pred=prec_pred)
    for t in iou_thresholds:
        results["tp_" + str(int(t*100))] = np.count_nonzero(sIoU_gt >= t)
        results["fn_" + str(int(t*100))] = np.count_nonzero(sIoU_gt < t)
        # results["fp_" + str(int(t*100))] = np.count_nonzero(sIoU_pred < t)
        results["fp_" + str(int(t*100))] = np.count_nonzero(prec_pred < t)

    return results


@dataclasses.dataclass
class ResultsInfo:
    method_name : str
    dataset_name : str

    tp_25 : int
    tp_50 : int
    tp_75 : int
    tp_mean : float

    fp_25 : int
    fp_50 : int
    fp_75 : int
    fp_mean : float

    fn_25 : int
    fn_50 : int
    fn_75 : int
    fn_mean : float

    f1_25 : float
    f1_50 : float
    f1_75 : float
    f1_mean : float

    sIoU_gt : float
    sIoU_pred : float
    prec_pred : float = float(np.nan)

    def __iter__(self):
        return dataclasses.asdict(self).items()

    def save(self, path):
        hdf5_write_hierarchy_to_file(path, dataclasses.asdict(self))

    @classmethod
    def from_file(cls, path):
        return cls(**hdf5_read_hierarchy_from_file(path))


@MetricRegistry.register_class()
class MetricSegment(EvaluationMetric):
    configs = [
        EasyDict(
            name='SegEval',
            thresh_p=None,
            thresh_sIoU=np.linspace(0.25, 0.75, 11, endpoint=True),
            thresh_segsize=500,
            thresh_instsize=100,
        ),
        EasyDict(
            name='SegEval-AnomalyTrack',
            thresh_p=None,
            thresh_sIoU=np.linspace(0.25, 0.75, 11, endpoint=True),
            thresh_segsize=500,
            thresh_instsize=100,
        ),
        EasyDict(
            name='SegEval-ObstacleTrack',
            thresh_p=None,
            thresh_sIoU=np.linspace(0.25, 0.75, 11, endpoint=True),
            thresh_segsize=50,
            thresh_instsize=10,
        )
    ]

    @property
    def name(self):
        return self.cfg.name

    def vis_frame(self, fid, dset_name, method_name, mask_roi, anomaly_p, image=None, **_):
        segmentation = np.copy(anomaly_p)
        segmentation[anomaly_p > self.cfg.thresh_p] = 1
        segmentation[anomaly_p <= self.cfg.thresh_p] = 0
        h, w = mask_roi.shape[:2]
        canvas = image.copy() if image is not None else np.zeros((h, w, 3), dtype=np.uint8)
        heatmap_color = adapt_img_data(segmentation)
        canvas[mask_roi] = canvas[mask_roi] // 2 + heatmap_color[mask_roi] // 2
        imwrite(DIR_OUTPUTS / f'vis_SegPred' / method_name / dset_name / f'{fid}.webp', canvas)

    def process_frame(self, label_pixel_gt: np.ndarray, anomaly_p: np.ndarray, fid : str=None, dset_name : str=None,
                      method_name : str=None, visualize : bool = True, **_):
        """
        @param label_pixel_gt: HxW uint8
            0 = in-distribution / road
            1 = anomaly / obstacle
            255 = void / ignore
        @param anomaly_p: HxW float16
            heatmap of per-pixel anomaly detection, higher values correspond to anomaly / obstacle class
        @param visualize: bool
            saves an image with segment predictions
        """

        if self.cfg.get(default_instancer, True):
            anomaly_gt, anomaly_pred, mask = default_instancer(anomaly_p, label_pixel_gt, self.cfg.thresh_p,
                                                               self.cfg.thresh_segsize, self.cfg.thresh_instsize)
            # imwrite(_["mask_path"], mask)
        else:
            anomaly_mask = imread(_["mask_path"])
            anomaly_gt, anomaly_pred = anomaly_instances_from_mask(anomaly_mask, label_pixel_gt, self.cfg.thresh_instsize)

        results = segment_metrics(anomaly_gt, anomaly_pred, self.cfg.thresh_sIoU)

        mask_roi = label_pixel_gt < 255
        if visualize and fid is not None and dset_name is not None and method_name is not None:
            self.vis_frame(fid=fid, dset_name=dset_name, method_name=method_name, mask_roi=mask_roi,
                           anomaly_p=anomaly_p, **_)

        return results

    def aggregate(self, frame_results: list, method_name: str, dataset_name: str):

        sIoU_gt_mean = sum(np.sum(r.sIoU_gt) for r in frame_results) / sum(len(r.sIoU_gt) for r in frame_results)
        sIoU_pred_mean = sum(np.sum(r.sIoU_pred) for r in frame_results) / sum(len(r.sIoU_pred) for r in frame_results)
        prec_pred_mean = sum(np.sum(r.prec_pred) for r in frame_results) / sum(len(r.prec_pred) for r in frame_results)
        ag_results = {"tp_mean" : 0., "fn_mean" : 0., "fp_mean" : 0., "f1_mean" : 0.,
                      "sIoU_gt" : sIoU_gt_mean, "sIoU_pred" : sIoU_pred_mean, "prec_pred": prec_pred_mean}
        print("Mean sIoU GT   :", sIoU_gt_mean)
        print("Mean sIoU PRED :", sIoU_pred_mean)
        print("Mean Precision PRED :", prec_pred_mean)
        for t in self.cfg.thresh_sIoU:
            tp = sum(r["tp_" + str(int(t*100))] for r in frame_results)
            fn = sum(r["fn_" + str(int(t*100))] for r in frame_results)
            fp = sum(r["fp_" + str(int(t*100))] for r in frame_results)
            f1 = (2 * tp) / (2 * tp + fn + fp)
            if t in [0.25, 0.50, 0.75]:
                ag_results["tp_" + str(int(t * 100))] = tp
                ag_results["fn_" + str(int(t * 100))] = fn
                ag_results["fp_" + str(int(t * 100))] = fp
                ag_results["f1_" + str(int(t * 100))] = f1
            # print("---sIoU thresh =", t)
            # print("Number of TPs  :", tp)
            # print("Number of FNs  :", fn)
            # print("Number of FPs  :", fp)
            # print("F1 score       :", f1)
            ag_results["tp_mean"] += tp
            ag_results["fn_mean"] += fn
            ag_results["fp_mean"] += fp
            ag_results["f1_mean"] += f1

        ag_results["tp_mean"] /= len(self.cfg.thresh_sIoU)
        ag_results["fn_mean"] /= len(self.cfg.thresh_sIoU)
        ag_results["fp_mean"] /= len(self.cfg.thresh_sIoU)
        ag_results["f1_mean"] /= len(self.cfg.thresh_sIoU)
        print("---sIoU thresh averaged")
        print("Number of TPs  :", ag_results["tp_mean"])
        print("Number of FNs  :", ag_results["fn_mean"])
        print("Number of FPs  :", ag_results["fp_mean"])
        print("F1 score       :", ag_results["f1_mean"])

        seg_info = ResultsInfo(
            method_name=method_name,
            dataset_name=dataset_name,
            **ag_results,
        )
        return seg_info

    def persistence_path_data(self, method_name, dataset_name):
        return DIR_OUTPUTS / self.name / 'data' / f'{self.name}Results_{method_name}_{dataset_name}.hdf5'

    def save(self, aggregated_result, method_name: str, dataset_name: str, path_override: Path = None):
        out_path = path_override or self.persistence_path_data(method_name, dataset_name)
        aggregated_result.save(out_path)

    def load(self, method_name: str, dataset_name: str, path_override: Path = None):
        out_path = path_override or self.persistence_path_data(method_name, dataset_name)
        return ResultsInfo.from_file(out_path)

    def fields_for_table(self):
        return ['sIoU_gt', 'prec_pred', 'fn_25', 'fp_25', 'f1_25', 'fn_50', 'fp_50', 'f1_50',
                'fn_75', 'fp_75', 'f1_75', 'f1_mean']

    def init(self, method_name, dataset_name):
        self.get_thresh_p_from_curve(method_name, dataset_name)

    def get_thresh_p_from_curve(self, method_name, dataset_name):
        out_path = DIR_OUTPUTS / "PixBinaryClass" / 'data' / f'PixClassCurve_{method_name}_{dataset_name}.hdf5'
        pixel_results = hdf5_read_hierarchy_from_file(out_path)
        if "best_f1_threshold" in pixel_results.keys():
            self.cfg.thresh_p = pixel_results.best_f1_threshold
        else:
            prc = pixel_results.curve_precision
            rec = pixel_results.curve_recall
            f1_scores = (2 * prc * rec) / (prc + rec)
            ix = np.nanargmax(f1_scores)
            self.cfg.thresh_p  = float(pixel_results.thresholds[ix])

        return self.cfg.thresh_p
