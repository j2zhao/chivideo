# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Perform inference on a single video or all videos with a certain extension
(e.g., .mp4) in a folder.
"""

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from tqdm import tqdm

from chivideo.pipeline import *
from chivideo.cache_streams import *

import subprocess as sp
import numpy as np
import time
import argparse
import sys
import os
import glob

def parse_args():
    # right now we don't save intermediate results
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='./resutls',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: mp4)',
        default='mp4',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )

    parser.add_argument('--dense', action='store_true', help='use dense convolutions instead of dilated convolutions')

    parser.add_argument('-ch', '--channels', default=1024, type=int, metavar='N', help='number of channels in convolution layers')
    parser.add_argument('-arc', '--architecture', default='3,3,3', type=str, metavar='LAYERS', help='filter widths separated by comma')


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

class Skeleton:
    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)
        
        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()
    
    def num_joints(self):
        return len(self._parents)
    
    def parents(self):
        return self._parents
    
    def has_children(self):
        return self._has_children
    
    def children(self):
        return self._children
    
    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove'.
        """
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]
                
        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)
        
        
        if self._joints_left is not None:
            new_joints_left = []
            for joint in self._joints_left:
                if joint in valid_joints:
                    new_joints_left.append(joint - index_offsets[joint])
            self._joints_left = new_joints_left
        if self._joints_right is not None:
            new_joints_right = []
            for joint in self._joints_right:
                if joint in valid_joints:
                    new_joints_right.append(joint - index_offsets[joint])
            self._joints_right = new_joints_right

        self._compute_metadata()
        
        return valid_joints
    
    def joints_left(self):
        return self._joints_left
    
    def joints_right(self):
        return self._joints_right
        
    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

class DetectronPoseOP(Operator):
    # need to initialize appropriate dstreams
    def __init__(self, name, vstream, predictor):
        input_names = [vstream]
        output_names = ['boxes', 'segments', 'keypoints']
        self.predictor = predictor
        self.vstream = vstream
        super().__init__(name, input_names, output_names)

    def next(self):
        im = self.istreams[self.vstream].next(self.name)
        if im == None:
            return False
        outputs = self.predictor(im)['instances'].to('cpu')
        has_bbox = False

        if outputs.has('pred_boxes'):
            bbox_tensor = outputs.pred_boxes.tensor.numpy()
            if len(bbox_tensor) > 0:
                has_bbox = True
                scores = outputs.scores.numpy()[:, None]
                bbox_tensor = np.concatenate((bbox_tensor, scores), axis=1)
        if has_bbox:
            kps = outputs.pred_keypoints.numpy()
            kps_xy = kps[:, :, :2]
            kps_prob = kps[:, :, 2:3]
            kps_logit = np.zeros_like(kps_prob) # Dummy
            kps = np.concatenate((kps_xy, kps_logit, kps_prob), axis=2)
            kps = kps.transpose(0, 2, 1)
        else:
            kps = []
            bbox_tensor = []
            
        # # Mimic Detectron1 format
        # cls_boxes = [[], bbox_tensor]
        # cls_keyps = [[], kps]
        
        self.ostreams['boxes'].append(bbox_tensor)
        self.ostreams['segments'].append(None)
        self.ostreams['keypoints'].append(kps)
        return True

    def init_outputs(self):
        for key in self.output_names:
            self.ostreams[key] = CacheStream(key)
        return super().init_outputs()

class FindBestOP(Operator):
    # need to initialize appropriate dstreams
    def __init__(self, name, boxes = 'boxes', keypoints = 'keypoints'):
        input_names = [boxes, keypoints]
        self.boxes = boxes
        self.keypoints = keypoints
        output_names = ['results_bb', 'results_kp']
        super().__init__(name, input_names, output_names)

    def next(self):
        bb = self.istreams[self.boxes].next()
        kp = self.istreams[self.keypoints].next()
        if bb == None and kp == None:
            return False
        if len(bb) == 0 or len(kp) == 0:
            # No bbox/keypoints detected for this frame -> will be interpolated
            self.ostreams['results_bb'].append(np.full(4, np.nan, dtype=np.float32)) # 4 bounding box coordinates
            self.ostreams['results_kp'].append(np.full((17, 4), np.nan, dtype=np.float32)) # 17 COCO keypoints
        else:
            best_match = np.argmax(bb[:, 4])
            best_bb = bb[best_match, :4]
            best_kp = kp[best_match].T.copy()
            self.ostreams['results_bb'].append(best_bb)
            self.ostreams['results_kp'].append(best_kp)
        return True
    
    def init_outputs(self):
        for key in self.output_names:
            self.ostreams[key] = CacheStream(key)
        return super().init_outputs()

class InterpolateOP(Operator):
# need to initialize appropriate dstreams
    def __init__(self, name, results_bb = 'results_bb', result_kp = 'results_kp'):
        input_names = [results_bb, result_kp]
        self.results_bb = results_bb
        self.result_kp = result_kp
        output_names = ['bbs_np', 'kps_np']
        super().__init__(name, input_names, output_names)

    def next(self):
        bb = []
        kp = []
        while True:
            bb_ = self.istreams[self.results_bb].next()
            kp_ = self.istreams[self.result_kp].next()
            if bb_ == None and kp_ == None:
                break
            bb.append(bb_)
            kp.append(kp_)
            
        bb = np.array(bb, dtype=np.float32)
        kp = np.array(kp, dtype=np.float32)
        kp = kp[:, :, :2] # Extract (x, y)

        # Fix missing bboxes/keypoints by linear interpolation
        mask = ~np.isnan(bb[:, 0])
        indices = np.arange(len(bb))
        for i in range(4):
            bb[:, i] = np.interp(indices, indices[mask], bb[mask, i])
        for i in range(17):
            for j in range(2):
                kp[:, i, j] = np.interp(indices, indices[mask], kp[mask, i, j])

        self.ostreams['bb_np'].insert(bb)
        self.ostreams['kp_np'].insert(kp)
        return True

    def init_outputs(self):
        for key in self.output_names:
            self.ostreams[key] = CacheStream(key)
        return super().init_outputs()

class ReFormat3DOP(Operator):
    def __init__(self, name, pad, causal_shift, meta = 'meta', kp = 'kp_np'):
        self.pad = pad
        self.causal_shift = causal_shift
        input_names = [meta, kp]
        self.meta = meta
        self.kp = kp
        output_names = ['out_kp']
        super().__init__(name, input_names, output_names)


    def next(self):
        X = self.istreams[self.kp].next()
        w, h = self.istreams[self.meta].next()
        X[..., :2] = X[..., :2]/w*2 - [1, h/w]
        X = np.expand_dims(np.pad(X, ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)), 'edge'), axis=0)
        self.ostreams['out_kp'].insert(X)
    
    def init_outputs(self):
        for key in self.output_names:
            self.ostreams[key] = CacheStream(key)
        return super().init_outputs()

class Pose3DOP(Operator):
    def __init__(self, name, model, batch_2d = 'out_kp'):
        input_names = [batch_2d]
        self.batch_2d = batch_2d
        output_names = ['poses']
        self.model = model
        super.__init__(name, input_names, output_names)

    def next(self):
        inputs_2d = self.istreams[self.batch_2d].next()
        inputs_2d = torch.from_numpy(inputs_2d.astype('float32'))
        if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
        
        predicted_3d_pos = self.model(inputs_2d)
        predicted_3d_pos = predicted_3d_pos.squeeze(0).cpu().numpy()
        self.ostreams['poses'].insert(predicted_3d_pos)
        return True

    def init_outputs(self):
        for key in self.output_names:
            self.ostreams[key] = CacheStream(key)
        return super().init_outputs()
    
def initializePipeline(vstream, meta, x, y, predictor, skeleton, filter_widths, channels, dense):
    pipeline = Pipeline({'video': vstream, 'meta': meta})

    model_pos = TemporalModel(x, y, skeleton.num_joints(),
                            filter_widths=filter_widths, causal=False, dropout=0.0, channels=channels,
                            dense=dense)
    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2 
    causal_shift = 0


    pipeline.add_operator(DetectronPoseOP('detectron', 'video', predictor))
    pipeline.add_operator(FindBestOP('single'))
    pipeline.add_operatorInterpolateOP('interpolate'))
    pipeline.add_operator(ReFormat3DOP('reformat', pad, causal_shift))
    pipeline.add_operator(Pose3DOP('pose', model_pos))

    poses = pipeline.get_outputs(['poses'], keep_all = True)

    poses.add_iter('output')

    return pipeline, poses


def main(args):

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.cfg))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.cfg)
    predictor = DefaultPredictor(cfg)

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    filter_widths = [int(x) for x in args.architecture.split(',')]
    
    skeleton = Skeleton(parents=[-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12,
       16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
       joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
       joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
    
    remove_joints = True
    if remove_joints:
        skeleton.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
        skeleton._parents[11] = 8
        skeleton._parents[14] = 8
    

    
    pipeline = None

    for video_name in tqdm(im_list):
        out_name = os.path.join(
                args.output_dir, os.path.basename(video_name)
            )
        
        print('Processing {}'.format(video_name))
        #predictor.filename = video_name
        #predictor.count = 0
        vstream = VideoStream('video', video_name)
        meta = JSONListStream([(vstream.width, vstream.hieght)])
        if pipeline == None:
            pipeline, poses = initializePipeline(vstream, meta, vstream.width, vstream.hieght, predictor, skeleton, filter_widths, args.channels, args.dense)
            
        else:
            pipeline.reset({'video':vstream, 'meta': meta})

        p = poses.next()

        # save poses in whatever format we want -> numpy array
        #snp.savez_compressed(out_name, boxes=boxes.all(), segments=segments.all(), keypoints=keypoints.all(), metadata=metadata)


if __name__ == '__main__':
    setup_logger()
    args = parse_args()
    main(args)
