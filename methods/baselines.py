import os
import random
from pathlib import Path
import torch
import h5py
import numpy as np
import torch.nn.functional as F

from PIL import Image

from .image_dissimilarity.data.cityscapes_dataset import one_hot_encoding
from .image_segmentation.datasets.cityscapes_labels import label2trainid
from .options.config_class import Config
from .utils import download_checkpoint, download_zip, init_pytorch_DeepWV3Plus, get_softmax, get_calibrated_softmax, \
    mahalanobis_modification, get_activations, download_tar, get_segmentation, get_synthesis, \
    get_dissimilarity, get_synboost_transformations, get_entropy


DIR_CHECKPOINTS = Path(os.environ.get('DIR_SEGMENTME_BASELINES', "/tmp/checkpoints/"))

class Max_softmax:
    def __init__(self, modelid):
        checkpoint_path = DIR_CHECKPOINTS / "DeepLabV3+_WideResNet38_baseline.pth"
        if not checkpoint_path.is_file():
            checkpoint_download_url = os.path.join("https://uni-wuppertal.sciebo.de/s/", modelid, "download")
            print("Download checkpoint from:", checkpoint_download_url)
            filename = download_checkpoint(checkpoint_download_url, DIR_CHECKPOINTS)
            (DIR_CHECKPOINTS / filename).rename(checkpoint_path)
        self.model = init_pytorch_DeepWV3Plus(checkpoint_path)

    def anomaly_score(self, image):
        probs = get_softmax(self.model, image)
        return 1 - np.max(probs, axis=0)


class ODIN:
    def __init__(self, modelid, magnitude=0.0001, temperature=3.0):
        self.magnitude = magnitude  # default magnitude tuned on LaF
        self.temperature = temperature  # default temperature tuned on LaF
        checkpoint_path = DIR_CHECKPOINTS / "DeepLabV3+_WideResNet38_baseline.pth"
        if not checkpoint_path.is_file():
            checkpoint_download_url = os.path.join("https://uni-wuppertal.sciebo.de/s/", modelid, "download")
            print("Download checkpoint from:", checkpoint_download_url)
            filename = download_checkpoint(checkpoint_download_url, DIR_CHECKPOINTS)
            (DIR_CHECKPOINTS / filename).rename(checkpoint_path)
        self.model = init_pytorch_DeepWV3Plus(checkpoint_path)

    def anomaly_score(self, image):
        calibrated_softmax = get_calibrated_softmax(self.model, image, self.magnitude, self.temperature, as_numpy=True)
        return 1 - np.max(calibrated_softmax, axis=0)


class Mahalanobis:
    def __init__(self, modelid):
        checkpoint_path = DIR_CHECKPOINTS / "DeepLabV3+_WideResNet38_baseline.pth"
        estimates_path = DIR_CHECKPOINTS / "cityscapes_train_estimates_global.h5"
        if not checkpoint_path.is_file() or not estimates_path.is_file():
            zip_download_url = os.path.join("https://uni-wuppertal.sciebo.de/s/", modelid, "download")
            print("Download checkpoint from:", zip_download_url)
            filename = download_zip(zip_download_url, DIR_CHECKPOINTS)
            os.remove(filename)
        with h5py.File(estimates_path, "r") as data:
            self.arithmetic_means = np.array(data['means'])
            self.arithmetic_means = self.arithmetic_means.reshape(self.arithmetic_means.shape[:2] + (-1,))
            self.inverse_covariances = np.array(data['inverse'])
            self.inverse_covariances = torch.from_numpy(self.inverse_covariances.astype("float"))
        self.model = mahalanobis_modification(init_pytorch_DeepWV3Plus(), checkpoint_path)

    def forward_pass(self, image):
        return get_activations(self.model, image)

    def anomaly_score(self, image):
        activations = self.forward_pass(image)
        output_shape = activations.shape
        mahalanobis = np.zeros((len(self.arithmetic_means),) + output_shape[1:])
        for c in range(len(self.arithmetic_means)):
            z = activations.reshape(output_shape[0], -1) - self.arithmetic_means[c]
            z = torch.from_numpy(z.astype("float"))
            left = torch.einsum('ij,ik->kj', z, self.inverse_covariances[c])
            score = torch.einsum('ij,ij->j', left, z).numpy()
            mahalanobis[c] = score.reshape(output_shape[1:])
        return np.min(mahalanobis, axis=0).astype("float32")


class Entropy_max:
    """Code from https://github.com/robin-chan/meta-ood"""
    def __init__(self, modelid):
        checkpoint_path = DIR_CHECKPOINTS / "DeepLabV3+_WideResNet38_epoch_4_alpha_0.9.pth"
        if not checkpoint_path.is_file():
            checkpoint_download_url = os.path.join("https://uni-wuppertal.sciebo.de/s/", modelid, "download")
            print("Download checkpoint from:", checkpoint_download_url)
            filename = download_checkpoint(checkpoint_download_url, DIR_CHECKPOINTS)
            (DIR_CHECKPOINTS / filename).rename(checkpoint_path)
        self.model = init_pytorch_DeepWV3Plus(checkpoint_path)

    def anomaly_score(self, image):
        return get_entropy(self.model, image)


class SynBoost:
    """Code from https://github.com/giandbt/synboost"""
    def __init__(self, seed=0):
        checkpoints_dir = os.path.join(DIR_CHECKPOINTS, "synboost_weights")
        if not os.path.exists(checkpoints_dir):
            pretrained_weights_url = os.path.join("http://robotics.ethz.ch/~asl-datasets/Dissimilarity/models.tar")
            print("Download checkpoint from:", pretrained_weights_url)
            filename = download_tar(pretrained_weights_url, DIR_CHECKPOINTS)
            os.remove(filename)
            os.rename(os.path.join(DIR_CHECKPOINTS, "models"), os.path.join(DIR_CHECKPOINTS, "synboost_weights"))

        self.set_seeds(int(seed))
        # Common options for all models
        torch.cuda.empty_cache()
        self.seg_net = get_segmentation(checkpoints_dir, Config())
        self.syn_net = get_synthesis(checkpoints_dir, Config())
        self.diss_model, self.prior, self.ensemble = get_dissimilarity(checkpoints_dir)
        # self.get_transformations()
        self.img_transform, self.transform_semantic, self.transform_image_syn, self.vgg_diff, \
        self.base_transforms_diss, self.norm_transform_diss = get_synboost_transformations()

    def anomaly_score(self, image):
        image = Image.fromarray(image)
        image_og_h = image.size[1]
        image_og_w = image.size[0]
        img = image.resize((2048, 1024))
        img_tensor = self.img_transform(img)

        # predict segmentation
        with torch.no_grad():
            seg_outs = self.seg_net(img_tensor.unsqueeze(0).cuda())

        seg_softmax_out = F.softmax(seg_outs, dim=1)
        seg_final = np.argmax(seg_outs.cpu().numpy().squeeze(), axis=0)  # segmentation map

        # get entropy
        entropy = torch.sum(-seg_softmax_out * torch.log(seg_softmax_out), dim=1)
        entropy = (entropy - entropy.min()) / entropy.max()
        entropy *= 255  # for later use in the dissimilarity

        # get softmax distance
        distance, _ = torch.topk(seg_softmax_out, 2, dim=1)
        max_logit = distance[:, 0, :, :]
        max2nd_logit = distance[:, 1, :, :]
        result = max_logit - max2nd_logit
        distance = 1 - (result - result.min()) / result.max()
        distance *= 255  # for later use in the dissimilarity

        # get label map for synthesis model
        label_out = np.zeros_like(seg_final)
        for label_id, train_id in label2trainid.items():
            label_out[np.where(seg_final == train_id)] = label_id
        label_img = Image.fromarray((label_out).astype(np.uint8))

        # prepare for synthesis
        label_tensor = self.transform_semantic(label_img) * 255.0
        label_tensor[label_tensor == 255] = 35  # 'unknown' is opt.label_nc
        image_tensor = self.transform_image_syn(img)
        # Get instance map in right format. Since prediction doesn't have instance map, we use semantic instead
        instance_tensor = label_tensor.clone()

        # run synthesis
        syn_input = {'label': label_tensor.unsqueeze(0), 'instance': instance_tensor.unsqueeze(0),
                     'image': image_tensor.unsqueeze(0)}
        generated = self.syn_net(syn_input, mode='inference')
        image_numpy = (np.transpose(generated.squeeze().cpu().numpy(), (1, 2, 0)) + 1) / 2.0
        synthesis_final_img = Image.fromarray((image_numpy * 255).astype(np.uint8))

        # prepare dissimilarity
        entropy = entropy.cpu().numpy()
        distance = distance.cpu().numpy()
        entropy_img = Image.fromarray(entropy.astype(np.uint8).squeeze())
        distance = Image.fromarray(distance.astype(np.uint8).squeeze())
        semantic = Image.fromarray((seg_final).astype(np.uint8))

        # get initial transformation
        semantic_tensor = self.base_transforms_diss(semantic) * 255
        syn_image_tensor = self.base_transforms_diss(synthesis_final_img)
        image_tensor = self.base_transforms_diss(img)
        syn_image_tensor = self.norm_transform_diss(syn_image_tensor).unsqueeze(0).cuda()
        image_tensor = self.norm_transform_diss(image_tensor).unsqueeze(0).cuda()

        # get softmax difference
        perceptual_diff = self.vgg_diff(image_tensor, syn_image_tensor)
        min_v = torch.min(perceptual_diff.squeeze())
        max_v = torch.max(perceptual_diff.squeeze())
        perceptual_diff = (perceptual_diff.squeeze() - min_v) / (max_v - min_v)
        perceptual_diff *= 255
        perceptual_diff = perceptual_diff.cpu().numpy()
        perceptual_diff = Image.fromarray(perceptual_diff.astype(np.uint8))

        # finish transformation
        perceptual_diff_tensor = self.base_transforms_diss(perceptual_diff).unsqueeze(0).cuda()
        entropy_tensor = self.base_transforms_diss(entropy_img).unsqueeze(0).cuda()
        distance_tensor = self.base_transforms_diss(distance).unsqueeze(0).cuda()

        # hot encode semantic map
        semantic_tensor[semantic_tensor == 255] = 20  # 'ignore label is 20'
        semantic_tensor = one_hot_encoding(semantic_tensor, 20).unsqueeze(0).cuda()

        # run dissimilarity
        with torch.no_grad():
            if self.prior:
                diss_pred = F.softmax(
                    self.diss_model(image_tensor, syn_image_tensor, semantic_tensor, entropy_tensor,
                                    perceptual_diff_tensor, distance_tensor), dim=1)
            else:
                diss_pred = F.softmax(self.diss_model(image_tensor, syn_image_tensor, semantic_tensor), dim=1)
        diss_pred = diss_pred.cpu().numpy()

        # do ensemble if necessary
        if self.ensemble:
            diss_pred = diss_pred[:, 1, :, :] * 0.75 + entropy_tensor.cpu().numpy() * 0.25
        else:
            diss_pred = diss_pred[:, 1, :, :]

        out = np.array(Image.fromarray(diss_pred.squeeze()).resize((image_og_w, image_og_h)))
        return out.astype("float32")

    @staticmethod
    def set_seeds(seed=0):
        # set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
