import math
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image

    Parameters
    ----------
    tensors : torch.Tensor
        images after padding reshape
    image_sizes : List[Tuple[int, int]]
        image shapes before padding

    """

    def __init__(
        self,
        tensors: torch.Tensor,
        image_sizes: List[Tuple[int, int]]
    ) -> None:
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


class GeneralizedViBERTgridTransform(nn.Module):
    """
    Performs input / label transformation before feeding the data to a ViBERTgrid
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / label resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Tensor] for the targets

    Adapted from the PyTorch official implementation of Faster R-CNN

    Parameters
    ----------
    image_mean : float
        mean of all images, used in normalization
    image_std : float
        std of all images, used in normalization
    train_min_size : List
        list of min size length of the reshaped image during training,  
        will randomly choose from the given list
    test_min_size : int, optional
        min size length of the reshaped image during inference, by default 512
    max_size : int, optional
        max size length of the reshaped image and labels, by default 800
    num_classed : int, optional
        number of classes, by default 5

    """

    def __init__(
        self,
        image_mean: List[float],
        image_std: List[float],
        train_min_size: List,
        test_min_size: int = 512,
        max_size: int = 800,
        num_classes: int = 5
    ):
        super(GeneralizedViBERTgridTransform, self).__init__()
        if not isinstance(train_min_size, (list, tuple)):
            train_min_size = list(train_min_size)
        self.train_min_size_list = train_min_size       # 指定训练时图像缩放的最小边长范围
        self.test_min_size = test_min_size              # 指定测试过程中的最小边长
        self.max_size = max_size                        # 指定图像的最大边长范围

        assert isinstance(image_mean, (
                          float, List)), f"image_mean must be float or list of float, {type(image_mean)} given"
        assert isinstance(image_std, (
                          float, List)), f"image_std must be float or list of float, {type(image_std)} given"
        if isinstance(image_mean, float):
            image_mean = [image_mean] * 3
        elif len(image_mean) != 3:
            raise ValueError(
                f"image_mean must contain 3 three values, {len(image_mean)} given")
        if isinstance(image_std, float):
            image_std = [image_std] * 3
        elif len(image_std) != 3:
            raise ValueError(
                f"image_std must contain 3 three values, {len(image_std)} given")

        self.image_mean = image_mean                    # 指定图像在标准化处理中的均值
        self.image_std = image_std                      # 指定图像在标准化处理中的方差

        self.num_classes = num_classes

    def normalize(self, image: torch.Tensor):
        """
        Image normalization

        Parameters
        ----------
        image: torch.Tensor
            Original input images

        Returns
        -------
        normalized_image: torch.Tensor
            Normalized image

        """
        dtype, device = image.dtype, image.device
        mean = torch.tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k: List[int]) -> int:
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    @staticmethod
    def _resize_image(
        image: torch.Tensor,
        self_min_size: float,
        self_max_size: float
    ) -> torch.Tensor:
        im_shape = torch.tensor(image.shape[-2:])
        min_size = float(torch.min(im_shape))    # 获取高宽中的最小值
        max_size = float(torch.max(im_shape))    # 获取高宽中的最大值
        scale_factor = self_min_size / min_size  # 根据指定最小边长和图片最小边长计算缩放比例

        # 如果使用该缩放比例计算的图片最大边长大于指定的最大边长
        if max_size * scale_factor > self_max_size:
            scale_factor = self_max_size / max_size  # 将缩放比例设为指定最大边长和图片最大边长之比

        # interpolate利用插值的方法缩放图片
        # image[None]操作是在最前面添加batch维度[C, H, W] -> [1, C, H, W]
        # bilinear只支持4D Tensor
        image = F.interpolate(
            image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
            align_corners=False)[0]

        return image

    @staticmethod
    def _resize_labels(
        class_label: torch.Tensor,
        pos_neg_label: torch.Tensor,
        coor: torch.Tensor,
        resize_shape: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_shape = class_label.shape[-2:]
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=class_label.device) /
            torch.tensor(s_orig, dtype=torch.float32,
                         device=class_label.device)
            for s, s_orig in zip(resize_shape, orig_shape)
        ]
        ratios_height, ratios_width = ratios

        class_label = F.interpolate(
            class_label[None, None].float(), size=resize_shape, mode='nearest'
        )
        pos_neg_label = F.interpolate(
            pos_neg_label[None, None].float(), size=resize_shape, mode='nearest'
        )

        class_label = class_label.squeeze(0).squeeze(0).int()
        pos_neg_label = pos_neg_label.squeeze(0).squeeze(0).int()

        coor = coor.float()
        coor[:, [0, 2]] *= ratios_height
        coor[:, [1, 3]] *= ratios_width
        coor = coor.int()

        return class_label, pos_neg_label, coor

    def resize_and_convert(
        self,
        image: torch.Tensor,
        class_label: torch.Tensor,
        pos_neg_label: torch.Tensor,
        coor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        resize the image and labels to the given shape, 
        convert class_label and pos_neg_label to one-hot encoding format

        Parameters
        ----------
        image : torch.Tensor
            original image from the SROIE dataset
        class_label : torch.Tensor
            class label from the SROIE dataset
        pos_neg_label : torch.Tensor
            pos_neg_label from the SROIE dataset
        coor : torch.Tensor
            ocr_coordinates from the SROIE dataset

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            resized images and labels, labels are converted to one-hot encoding
        """

        if self.training:
            # 指定输入图片的最小边长,注意是self.min_size不是min_size
            size = float(self.torch_choice(self.train_min_size_list))
        else:
            size = float(self.test_min_size)

        image = self._resize_image(image, size, float(self.max_size))

        if class_label is None and pos_neg_label is None:
            return image, class_label, pos_neg_label

        resize_shape = image.shape[-2:]
        class_label, pos_neg_label, coor = self._resize_labels(
            class_label, pos_neg_label, coor, resize_shape)

        class_label = F.one_hot(
            class_label.long(), num_classes=self.num_classes).permute(2, 0, 1)
        pos_neg_label = F.one_hot(
            pos_neg_label.long(), num_classes=3).permute(2, 0, 1)

        return image, class_label, pos_neg_label, coor, resize_shape

    def max_by_axis(
        self,
        the_list: List[List[int]]
    ) -> List[int]:
        """
        get the maximum value of each dimension from the given batch

        Parameters
        ----------
        the_list : List[List[int]]
            list of image shape in a batch

        Returns
        -------
        List[int]
            maximum value of each dimension
        """
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def return_batch(
        self,
        images: List[torch.Tensor],
        class_labels: List[torch.Tensor],
        pos_neg_labels: List[torch.Tensor],
        size_divisible: int = 32
    ) -> torch.Tensor:
        """
        apply zero padding to the given batch of images and labels,
        making sure that the shape of each input can be divided by
        size_divisible for the purpose of backbone propagation

        Parameters
        ----------
        images: List[torch.Tensor]
            batch of input images
        class_labels: List[torch.Tensor]
            batch of class labels
        pos_neg_labels: List[torch.Tensor]
            batch of pos_neg labels
        size_divisible: int

        Returns
        -------
        batched_imgs: 
            reized batch images
        """

        # 分别计算一个batch中所有图片中的最大channel, height, width
        max_size = self.max_by_axis([list(img.shape) for img in images])
        num_classes = class_labels[0].shape[0]
        num_pos_neg_channels = pos_neg_labels[0].shape[0]

        stride = float(size_divisible)
        # max_size = list(max_size)
        # 将height向上调整到stride的整数倍
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        # 将width向上调整到stride的整数倍
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        # [batch, channel, height, width]
        batch_image_shape = [len(images)] + max_size
        batch_class_label_shape = [
            len(class_labels)] + [num_classes] + max_size[1:]
        batch_pos_neg_label_shape = [
            len(pos_neg_labels)] + [num_pos_neg_channels] + max_size[1:]

        # 创建shape为batch_shape且值全部为0的tensor
        batched_imgs = images[0].new_full(batch_image_shape, 0)
        batched_class_labels = images[0].new_full(batch_class_label_shape, 0)
        batched_pos_neg_labels = images[0].new_full(
            batch_pos_neg_label_shape, 0)
        for img, class_label, pos_neg_label, pad_img, pad_class_labels, pad_pos_neg_labels in zip(
            images, class_labels, pos_neg_labels, batched_imgs, batched_class_labels, batched_pos_neg_labels
        ):
            # 将输入images中的每张图片复制到新的batched_imgs的每张图片中，对齐左上角，保证bboxes的坐标不变
            # 这样保证输入到网络中一个batch的每张图片的shape相同
            # copy_: Copies the elements from src into self tensor and returns self
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            pad_class_labels[: class_label.shape[0], : class_label.shape[1],
                             : class_label.shape[2]].copy_(class_label)
            pad_pos_neg_labels[: pos_neg_label.shape[0], : pos_neg_label.shape[1],
                               : pos_neg_label.shape[2]].copy_(pos_neg_label)

        return batched_imgs, batched_class_labels, batched_pos_neg_labels

    def postprocess(self,
                    result,
                    image_shapes: List[Tuple[int, int]],
                    original_image_sizes: List[Tuple[int, int]]
                    ):
        # type: (...) -> List[Dict[str, torch.Tensor]]
        """对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        Args:
            result: list(dict), 网络的预测结果, len(result) == batch_size
            image_shapes: list(torch.Size), 图像预处理缩放后的尺寸, len(image_shapes) == batch_size
            original_image_sizes: list(torch.Size), 图像的原始尺寸, len(original_image_sizes) == batch_size

        Returns:

        """
        if self.training:
            return result

        # TODO implement coordinate reshape in inference mode
        # # 遍历每张图片的预测信息，将boxes信息还原回原尺度
        # for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
        #     boxes = pred["boxes"]
        #     boxes = resize_boxes(boxes, im_s, o_im_s)  # 将bboxes缩放回原图像尺度上
        #     result[i]["boxes"] = boxes
        # return result

    def forward(
        self,
        images: Tuple[torch.Tensor],
        class_labels: Tuple[torch.Tensor],
        pos_neg_labels: Tuple[torch.Tensor],
        ocr_coors: torch.Tensor,
    ):
        images = [img for img in images]
        class_labels = [class_label for class_label in class_labels]
        pos_neg_labels = [pos_neg_label for pos_neg_label in pos_neg_labels]
        image_sizes = []
        for i in range(len(images)):
            image = images[i]
            class_label = class_labels[i]
            pos_neg_label = pos_neg_labels[i]
            coor = ocr_coors[i]

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))

            # image normalization
            image = self.normalize(image)

            # image and label resize
            image, class_label, pos_neg_label, coor, resize_shape = self.resize_and_convert(
                image,
                class_label,
                pos_neg_label,
                coor
            )

            images[i] = image
            if class_label is not None and class_labels is not None:
                class_labels[i] = class_label
            if pos_neg_label is not None and pos_neg_labels is not None:
                pos_neg_labels[i] = pos_neg_label
            if coor is not None and ocr_coors is not None:
                ocr_coors[i] = coor

            image_sizes.append(resize_shape)

        images, class_labels, pos_neg_labels = self.return_batch(
            images, class_labels, pos_neg_labels)
        image_sizes_list = []

        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)

        return image_list, class_labels, pos_neg_labels, ocr_coors
