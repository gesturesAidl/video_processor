import os
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video
from gluoncv.model_zoo import get_model
from gluoncv.data import VideoClsCustom
from gluoncv.utils.filesystem import try_import_decord


class FeatureExtractor:

    def __init__(self):
        self.data_dir = '' # NEED TO COMPLETE
        model = 'i3d_resnet50_v1_kinetics400'
        num_classes = 400
        self.dtype = 'float32'
        self.num_segments = 1
        self.gpu_id = 1 # Number of GPUs.
        
        if self.gpu_id == -1:
            self.context = mx.cpu()
        else:
            gpu_id = self.gpu_id
            self.context = mx.gpu(gpu_id)

        self.net = get_model(name=model, nclass=num_classes, pretrained=True,
                        feat_ext=True, num_segments=self.num_segments, num_crop=self.num_crop)
        self.net.cast(self.dtype)
        self.net.collect_params().reset_ctx(self.context)

        # The default values, as they were not changed during training
        self.input_size = 224 
        self.new_length = 32
        self.new_height = 256
        self.new_width = 340

        self.video_loader = True
        self.use_decord = True
        self.num_crop = 1
        self.data_aug = 'v1'

        image_norm_mean = [0.485, 0.456, 0.406]
        image_norm_std = [0.229, 0.224, 0.225]

        self.transform_test = video.VideoGroupValTransform(size=self.input_size, mean=image_norm_mean, std=image_norm_std)

    def read_data(self, video_name, transform, video_utils):
        decord = try_import_decord()
        decord_vr = decord.VideoReader(video_name, width=self.new_width, height=self.new_height)
        duration = len(decord_vr)

        skip_length = self.new_length
        segment_indices, skip_offsets = video_utils._sample_test_indices(duration)

        clip_input = video_utils._video_TSN_decord_batch_loader(video_name, decord_vr, duration, segment_indices,
                                                                skip_offsets)
        clip_input = transform(clip_input)
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (self.new_length, 3, self.input_size, self.input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

        return nd.array(clip_input)

    def extract_features(self,video_path):
        data_list = self.data_dir+"data_list.txt"
        f= open(data_list,"w+")
        f.write(video_path)
        f.close()
        # build a pseudo dataset instance to use its children class methods
        video_utils = VideoClsCustom(root=self.data_dir,
                                     setting=data_list,
                                     num_segments=self.num_segments,
                                     num_crop=self.num_crop,
                                     new_length=self.new_length,
                                     new_step=self.new_step,
                                     new_width=self.new_width,
                                     new_height=self.new_height,
                                     video_loader=self.video_loader,
                                     use_decord=self.use_decord,
                                     data_aug=self.data_aug,
                                     lazy_init=True)


        video_data = self.read_data(video_path, self.transform_test, video_utils)
        video_input = video_data.as_in_context(self.context)
        video_feat = self.net(video_input.astype(self.dtype, copy=False))
        os.remove(data_list)
        
        return video_feat
