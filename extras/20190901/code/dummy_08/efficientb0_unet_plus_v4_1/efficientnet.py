#from common import *
import os
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F


from lib.net.sync_bn.nn import BatchNorm2dSync as SynchronizedBatchNorm2d

IS_PYTORCH_PAD = True  # True  # False

BatchNorm2d = SynchronizedBatchNorm2d
#BatchNorm2d = nn.BatchNorm2d

CONVERSION = [
 'stem.0.conv.weight',	(32, 3, 3, 3),	 '_conv_stem.weight',	(32, 3, 3, 3),
 'stem.0.bn.weight',	(32,),	 '_bn0.weight',	(32,),
 'stem.0.bn.bias',	(32,),	 '_bn0.bias',	(32,),
 'stem.0.bn.running_mean',	(32,),	 '_bn0.running_mean',	(32,),
 'stem.0.bn.running_var',	(32,),	 '_bn0.running_var',	(32,),
 'block1.0.bottleneck.0.conv.weight',	(32, 1, 3, 3),	 '_blocks.0._depthwise_conv.weight',	(32, 1, 3, 3),
 'block1.0.bottleneck.0.bn.weight',	(32,),	 '_blocks.0._bn1.weight',	(32,),
 'block1.0.bottleneck.0.bn.bias',	(32,),	 '_blocks.0._bn1.bias',	(32,),
 'block1.0.bottleneck.0.bn.running_mean',	(32,),	 '_blocks.0._bn1.running_mean',	(32,),
 'block1.0.bottleneck.0.bn.running_var',	(32,),	 '_blocks.0._bn1.running_var',	(32,),
 'block1.0.bottleneck.2.squeeze.weight',	(8, 32, 1, 1),	 '_blocks.0._se_reduce.weight',	(8, 32, 1, 1),
 'block1.0.bottleneck.2.squeeze.bias',	(8,),	 '_blocks.0._se_reduce.bias',	(8,),
 'block1.0.bottleneck.2.excite.weight',	(32, 8, 1, 1),	 '_blocks.0._se_expand.weight',	(32, 8, 1, 1),
 'block1.0.bottleneck.2.excite.bias',	(32,),	 '_blocks.0._se_expand.bias',	(32,),
 'block1.0.bottleneck.3.conv.weight',	(16, 32, 1, 1),	 '_blocks.0._project_conv.weight',	(16, 32, 1, 1),
 'block1.0.bottleneck.3.bn.weight',	(16,),	 '_blocks.0._bn2.weight',	(16,),
 'block1.0.bottleneck.3.bn.bias',	(16,),	 '_blocks.0._bn2.bias',	(16,),
 'block1.0.bottleneck.3.bn.running_mean',	(16,),	 '_blocks.0._bn2.running_mean',	(16,),
 'block1.0.bottleneck.3.bn.running_var',	(16,),	 '_blocks.0._bn2.running_var',	(16,),
 'block2.0.bottleneck.0.conv.weight',	(96, 16, 1, 1),	 '_blocks.1._expand_conv.weight',	(96, 16, 1, 1),
 'block2.0.bottleneck.0.bn.weight',	(96,),	 '_blocks.1._bn0.weight',	(96,),
 'block2.0.bottleneck.0.bn.bias',	(96,),	 '_blocks.1._bn0.bias',	(96,),
 'block2.0.bottleneck.0.bn.running_mean',	(96,),	 '_blocks.1._bn0.running_mean',	(96,),
 'block2.0.bottleneck.0.bn.running_var',	(96,),	 '_blocks.1._bn0.running_var',	(96,),
 'block2.0.bottleneck.2.conv.weight',	(96, 1, 3, 3),	 '_blocks.1._depthwise_conv.weight',	(96, 1, 3, 3),
 'block2.0.bottleneck.2.bn.weight',	(96,),	 '_blocks.1._bn1.weight',	(96,),
 'block2.0.bottleneck.2.bn.bias',	(96,),	 '_blocks.1._bn1.bias',	(96,),
 'block2.0.bottleneck.2.bn.running_mean',	(96,),	 '_blocks.1._bn1.running_mean',	(96,),
 'block2.0.bottleneck.2.bn.running_var',	(96,),	 '_blocks.1._bn1.running_var',	(96,),
 'block2.0.bottleneck.4.squeeze.weight',	(4, 96, 1, 1),	 '_blocks.1._se_reduce.weight',	(4, 96, 1, 1),
 'block2.0.bottleneck.4.squeeze.bias',	(4,),	 '_blocks.1._se_reduce.bias',	(4,),
 'block2.0.bottleneck.4.excite.weight',	(96, 4, 1, 1),	 '_blocks.1._se_expand.weight',	(96, 4, 1, 1),
 'block2.0.bottleneck.4.excite.bias',	(96,),	 '_blocks.1._se_expand.bias',	(96,),
 'block2.0.bottleneck.5.conv.weight',	(24, 96, 1, 1),	 '_blocks.1._project_conv.weight',	(24, 96, 1, 1),
 'block2.0.bottleneck.5.bn.weight',	(24,),	 '_blocks.1._bn2.weight',	(24,),
 'block2.0.bottleneck.5.bn.bias',	(24,),	 '_blocks.1._bn2.bias',	(24,),
 'block2.0.bottleneck.5.bn.running_mean',	(24,),	 '_blocks.1._bn2.running_mean',	(24,),
 'block2.0.bottleneck.5.bn.running_var',	(24,),	 '_blocks.1._bn2.running_var',	(24,),
 'block2.1.bottleneck.0.conv.weight',	(144, 24, 1, 1),	 '_blocks.2._expand_conv.weight',	(144, 24, 1, 1),
 'block2.1.bottleneck.0.bn.weight',	(144,),	 '_blocks.2._bn0.weight',	(144,),
 'block2.1.bottleneck.0.bn.bias',	(144,),	 '_blocks.2._bn0.bias',	(144,),
 'block2.1.bottleneck.0.bn.running_mean',	(144,),	 '_blocks.2._bn0.running_mean',	(144,),
 'block2.1.bottleneck.0.bn.running_var',	(144,),	 '_blocks.2._bn0.running_var',	(144,),
 'block2.1.bottleneck.2.conv.weight',	(144, 1, 3, 3),	 '_blocks.2._depthwise_conv.weight',	(144, 1, 3, 3),
 'block2.1.bottleneck.2.bn.weight',	(144,),	 '_blocks.2._bn1.weight',	(144,),
 'block2.1.bottleneck.2.bn.bias',	(144,),	 '_blocks.2._bn1.bias',	(144,),
 'block2.1.bottleneck.2.bn.running_mean',	(144,),	 '_blocks.2._bn1.running_mean',	(144,),
 'block2.1.bottleneck.2.bn.running_var',	(144,),	 '_blocks.2._bn1.running_var',	(144,),
 'block2.1.bottleneck.4.squeeze.weight',	(6, 144, 1, 1),	 '_blocks.2._se_reduce.weight',	(6, 144, 1, 1),
 'block2.1.bottleneck.4.squeeze.bias',	(6,),	 '_blocks.2._se_reduce.bias',	(6,),
 'block2.1.bottleneck.4.excite.weight',	(144, 6, 1, 1),	 '_blocks.2._se_expand.weight',	(144, 6, 1, 1),
 'block2.1.bottleneck.4.excite.bias',	(144,),	 '_blocks.2._se_expand.bias',	(144,),
 'block2.1.bottleneck.5.conv.weight',	(24, 144, 1, 1),	 '_blocks.2._project_conv.weight',	(24, 144, 1, 1),
 'block2.1.bottleneck.5.bn.weight',	(24,),	 '_blocks.2._bn2.weight',	(24,),
 'block2.1.bottleneck.5.bn.bias',	(24,),	 '_blocks.2._bn2.bias',	(24,),
 'block2.1.bottleneck.5.bn.running_mean',	(24,),	 '_blocks.2._bn2.running_mean',	(24,),
 'block2.1.bottleneck.5.bn.running_var',	(24,),	 '_blocks.2._bn2.running_var',	(24,),
 'block3.0.bottleneck.0.conv.weight',	(144, 24, 1, 1),	 '_blocks.3._expand_conv.weight',	(144, 24, 1, 1),
 'block3.0.bottleneck.0.bn.weight',	(144,),	 '_blocks.3._bn0.weight',	(144,),
 'block3.0.bottleneck.0.bn.bias',	(144,),	 '_blocks.3._bn0.bias',	(144,),
 'block3.0.bottleneck.0.bn.running_mean',	(144,),	 '_blocks.3._bn0.running_mean',	(144,),
 'block3.0.bottleneck.0.bn.running_var',	(144,),	 '_blocks.3._bn0.running_var',	(144,),
 'block3.0.bottleneck.2.conv.weight',	(144, 1, 5, 5),	 '_blocks.3._depthwise_conv.weight',	(144, 1, 5, 5),
 'block3.0.bottleneck.2.bn.weight',	(144,),	 '_blocks.3._bn1.weight',	(144,),
 'block3.0.bottleneck.2.bn.bias',	(144,),	 '_blocks.3._bn1.bias',	(144,),
 'block3.0.bottleneck.2.bn.running_mean',	(144,),	 '_blocks.3._bn1.running_mean',	(144,),
 'block3.0.bottleneck.2.bn.running_var',	(144,),	 '_blocks.3._bn1.running_var',	(144,),
 'block3.0.bottleneck.4.squeeze.weight',	(6, 144, 1, 1),	 '_blocks.3._se_reduce.weight',	(6, 144, 1, 1),
 'block3.0.bottleneck.4.squeeze.bias',	(6,),	 '_blocks.3._se_reduce.bias',	(6,),
 'block3.0.bottleneck.4.excite.weight',	(144, 6, 1, 1),	 '_blocks.3._se_expand.weight',	(144, 6, 1, 1),
 'block3.0.bottleneck.4.excite.bias',	(144,),	 '_blocks.3._se_expand.bias',	(144,),
 'block3.0.bottleneck.5.conv.weight',	(40, 144, 1, 1),	 '_blocks.3._project_conv.weight',	(40, 144, 1, 1),
 'block3.0.bottleneck.5.bn.weight',	(40,),	 '_blocks.3._bn2.weight',	(40,),
 'block3.0.bottleneck.5.bn.bias',	(40,),	 '_blocks.3._bn2.bias',	(40,),
 'block3.0.bottleneck.5.bn.running_mean',	(40,),	 '_blocks.3._bn2.running_mean',	(40,),
 'block3.0.bottleneck.5.bn.running_var',	(40,),	 '_blocks.3._bn2.running_var',	(40,),
 'block3.1.bottleneck.0.conv.weight',	(240, 40, 1, 1),	 '_blocks.4._expand_conv.weight',	(240, 40, 1, 1),
 'block3.1.bottleneck.0.bn.weight',	(240,),	 '_blocks.4._bn0.weight',	(240,),
 'block3.1.bottleneck.0.bn.bias',	(240,),	 '_blocks.4._bn0.bias',	(240,),
 'block3.1.bottleneck.0.bn.running_mean',	(240,),	 '_blocks.4._bn0.running_mean',	(240,),
 'block3.1.bottleneck.0.bn.running_var',	(240,),	 '_blocks.4._bn0.running_var',	(240,),
 'block3.1.bottleneck.2.conv.weight',	(240, 1, 5, 5),	 '_blocks.4._depthwise_conv.weight',	(240, 1, 5, 5),
 'block3.1.bottleneck.2.bn.weight',	(240,),	 '_blocks.4._bn1.weight',	(240,),
 'block3.1.bottleneck.2.bn.bias',	(240,),	 '_blocks.4._bn1.bias',	(240,),
 'block3.1.bottleneck.2.bn.running_mean',	(240,),	 '_blocks.4._bn1.running_mean',	(240,),
 'block3.1.bottleneck.2.bn.running_var',	(240,),	 '_blocks.4._bn1.running_var',	(240,),
 'block3.1.bottleneck.4.squeeze.weight',	(10, 240, 1, 1),	 '_blocks.4._se_reduce.weight',	(10, 240, 1, 1),
 'block3.1.bottleneck.4.squeeze.bias',	(10,),	 '_blocks.4._se_reduce.bias',	(10,),
 'block3.1.bottleneck.4.excite.weight',	(240, 10, 1, 1),	 '_blocks.4._se_expand.weight',	(240, 10, 1, 1),
 'block3.1.bottleneck.4.excite.bias',	(240,),	 '_blocks.4._se_expand.bias',	(240,),
 'block3.1.bottleneck.5.conv.weight',	(40, 240, 1, 1),	 '_blocks.4._project_conv.weight',	(40, 240, 1, 1),
 'block3.1.bottleneck.5.bn.weight',	(40,),	 '_blocks.4._bn2.weight',	(40,),
 'block3.1.bottleneck.5.bn.bias',	(40,),	 '_blocks.4._bn2.bias',	(40,),
 'block3.1.bottleneck.5.bn.running_mean',	(40,),	 '_blocks.4._bn2.running_mean',	(40,),
 'block3.1.bottleneck.5.bn.running_var',	(40,),	 '_blocks.4._bn2.running_var',	(40,),
 'block4.0.bottleneck.0.conv.weight',	(240, 40, 1, 1),	 '_blocks.5._expand_conv.weight',	(240, 40, 1, 1),
 'block4.0.bottleneck.0.bn.weight',	(240,),	 '_blocks.5._bn0.weight',	(240,),
 'block4.0.bottleneck.0.bn.bias',	(240,),	 '_blocks.5._bn0.bias',	(240,),
 'block4.0.bottleneck.0.bn.running_mean',	(240,),	 '_blocks.5._bn0.running_mean',	(240,),
 'block4.0.bottleneck.0.bn.running_var',	(240,),	 '_blocks.5._bn0.running_var',	(240,),
 'block4.0.bottleneck.2.conv.weight',	(240, 1, 3, 3),	 '_blocks.5._depthwise_conv.weight',	(240, 1, 3, 3),
 'block4.0.bottleneck.2.bn.weight',	(240,),	 '_blocks.5._bn1.weight',	(240,),
 'block4.0.bottleneck.2.bn.bias',	(240,),	 '_blocks.5._bn1.bias',	(240,),
 'block4.0.bottleneck.2.bn.running_mean',	(240,),	 '_blocks.5._bn1.running_mean',	(240,),
 'block4.0.bottleneck.2.bn.running_var',	(240,),	 '_blocks.5._bn1.running_var',	(240,),
 'block4.0.bottleneck.4.squeeze.weight',	(10, 240, 1, 1),	 '_blocks.5._se_reduce.weight',	(10, 240, 1, 1),
 'block4.0.bottleneck.4.squeeze.bias',	(10,),	 '_blocks.5._se_reduce.bias',	(10,),
 'block4.0.bottleneck.4.excite.weight',	(240, 10, 1, 1),	 '_blocks.5._se_expand.weight',	(240, 10, 1, 1),
 'block4.0.bottleneck.4.excite.bias',	(240,),	 '_blocks.5._se_expand.bias',	(240,),
 'block4.0.bottleneck.5.conv.weight',	(80, 240, 1, 1),	 '_blocks.5._project_conv.weight',	(80, 240, 1, 1),
 'block4.0.bottleneck.5.bn.weight',	(80,),	 '_blocks.5._bn2.weight',	(80,),
 'block4.0.bottleneck.5.bn.bias',	(80,),	 '_blocks.5._bn2.bias',	(80,),
 'block4.0.bottleneck.5.bn.running_mean',	(80,),	 '_blocks.5._bn2.running_mean',	(80,),
 'block4.0.bottleneck.5.bn.running_var',	(80,),	 '_blocks.5._bn2.running_var',	(80,),
 'block4.1.bottleneck.0.conv.weight',	(480, 80, 1, 1),	 '_blocks.6._expand_conv.weight',	(480, 80, 1, 1),
 'block4.1.bottleneck.0.bn.weight',	(480,),	 '_blocks.6._bn0.weight',	(480,),
 'block4.1.bottleneck.0.bn.bias',	(480,),	 '_blocks.6._bn0.bias',	(480,),
 'block4.1.bottleneck.0.bn.running_mean',	(480,),	 '_blocks.6._bn0.running_mean',	(480,),
 'block4.1.bottleneck.0.bn.running_var',	(480,),	 '_blocks.6._bn0.running_var',	(480,),
 'block4.1.bottleneck.2.conv.weight',	(480, 1, 3, 3),	 '_blocks.6._depthwise_conv.weight',	(480, 1, 3, 3),
 'block4.1.bottleneck.2.bn.weight',	(480,),	 '_blocks.6._bn1.weight',	(480,),
 'block4.1.bottleneck.2.bn.bias',	(480,),	 '_blocks.6._bn1.bias',	(480,),
 'block4.1.bottleneck.2.bn.running_mean',	(480,),	 '_blocks.6._bn1.running_mean',	(480,),
 'block4.1.bottleneck.2.bn.running_var',	(480,),	 '_blocks.6._bn1.running_var',	(480,),
 'block4.1.bottleneck.4.squeeze.weight',	(20, 480, 1, 1),	 '_blocks.6._se_reduce.weight',	(20, 480, 1, 1),
 'block4.1.bottleneck.4.squeeze.bias',	(20,),	 '_blocks.6._se_reduce.bias',	(20,),
 'block4.1.bottleneck.4.excite.weight',	(480, 20, 1, 1),	 '_blocks.6._se_expand.weight',	(480, 20, 1, 1),
 'block4.1.bottleneck.4.excite.bias',	(480,),	 '_blocks.6._se_expand.bias',	(480,),
 'block4.1.bottleneck.5.conv.weight',	(80, 480, 1, 1),	 '_blocks.6._project_conv.weight',	(80, 480, 1, 1),
 'block4.1.bottleneck.5.bn.weight',	(80,),	 '_blocks.6._bn2.weight',	(80,),
 'block4.1.bottleneck.5.bn.bias',	(80,),	 '_blocks.6._bn2.bias',	(80,),
 'block4.1.bottleneck.5.bn.running_mean',	(80,),	 '_blocks.6._bn2.running_mean',	(80,),
 'block4.1.bottleneck.5.bn.running_var',	(80,),	 '_blocks.6._bn2.running_var',	(80,),
 'block4.2.bottleneck.0.conv.weight',	(480, 80, 1, 1),	 '_blocks.7._expand_conv.weight',	(480, 80, 1, 1),
 'block4.2.bottleneck.0.bn.weight',	(480,),	 '_blocks.7._bn0.weight',	(480,),
 'block4.2.bottleneck.0.bn.bias',	(480,),	 '_blocks.7._bn0.bias',	(480,),
 'block4.2.bottleneck.0.bn.running_mean',	(480,),	 '_blocks.7._bn0.running_mean',	(480,),
 'block4.2.bottleneck.0.bn.running_var',	(480,),	 '_blocks.7._bn0.running_var',	(480,),
 'block4.2.bottleneck.2.conv.weight',	(480, 1, 3, 3),	 '_blocks.7._depthwise_conv.weight',	(480, 1, 3, 3),
 'block4.2.bottleneck.2.bn.weight',	(480,),	 '_blocks.7._bn1.weight',	(480,),
 'block4.2.bottleneck.2.bn.bias',	(480,),	 '_blocks.7._bn1.bias',	(480,),
 'block4.2.bottleneck.2.bn.running_mean',	(480,),	 '_blocks.7._bn1.running_mean',	(480,),
 'block4.2.bottleneck.2.bn.running_var',	(480,),	 '_blocks.7._bn1.running_var',	(480,),
 'block4.2.bottleneck.4.squeeze.weight',	(20, 480, 1, 1),	 '_blocks.7._se_reduce.weight',	(20, 480, 1, 1),
 'block4.2.bottleneck.4.squeeze.bias',	(20,),	 '_blocks.7._se_reduce.bias',	(20,),
 'block4.2.bottleneck.4.excite.weight',	(480, 20, 1, 1),	 '_blocks.7._se_expand.weight',	(480, 20, 1, 1),
 'block4.2.bottleneck.4.excite.bias',	(480,),	 '_blocks.7._se_expand.bias',	(480,),
 'block4.2.bottleneck.5.conv.weight',	(80, 480, 1, 1),	 '_blocks.7._project_conv.weight',	(80, 480, 1, 1),
 'block4.2.bottleneck.5.bn.weight',	(80,),	 '_blocks.7._bn2.weight',	(80,),
 'block4.2.bottleneck.5.bn.bias',	(80,),	 '_blocks.7._bn2.bias',	(80,),
 'block4.2.bottleneck.5.bn.running_mean',	(80,),	 '_blocks.7._bn2.running_mean',	(80,),
 'block4.2.bottleneck.5.bn.running_var',	(80,),	 '_blocks.7._bn2.running_var',	(80,),
 'block5.0.bottleneck.0.conv.weight',	(480, 80, 1, 1),	 '_blocks.8._expand_conv.weight',	(480, 80, 1, 1),
 'block5.0.bottleneck.0.bn.weight',	(480,),	 '_blocks.8._bn0.weight',	(480,),
 'block5.0.bottleneck.0.bn.bias',	(480,),	 '_blocks.8._bn0.bias',	(480,),
 'block5.0.bottleneck.0.bn.running_mean',	(480,),	 '_blocks.8._bn0.running_mean',	(480,),
 'block5.0.bottleneck.0.bn.running_var',	(480,),	 '_blocks.8._bn0.running_var',	(480,),
 'block5.0.bottleneck.2.conv.weight',	(480, 1, 5, 5),	 '_blocks.8._depthwise_conv.weight',	(480, 1, 5, 5),
 'block5.0.bottleneck.2.bn.weight',	(480,),	 '_blocks.8._bn1.weight',	(480,),
 'block5.0.bottleneck.2.bn.bias',	(480,),	 '_blocks.8._bn1.bias',	(480,),
 'block5.0.bottleneck.2.bn.running_mean',	(480,),	 '_blocks.8._bn1.running_mean',	(480,),
 'block5.0.bottleneck.2.bn.running_var',	(480,),	 '_blocks.8._bn1.running_var',	(480,),
 'block5.0.bottleneck.4.squeeze.weight',	(20, 480, 1, 1),	 '_blocks.8._se_reduce.weight',	(20, 480, 1, 1),
 'block5.0.bottleneck.4.squeeze.bias',	(20,),	 '_blocks.8._se_reduce.bias',	(20,),
 'block5.0.bottleneck.4.excite.weight',	(480, 20, 1, 1),	 '_blocks.8._se_expand.weight',	(480, 20, 1, 1),
 'block5.0.bottleneck.4.excite.bias',	(480,),	 '_blocks.8._se_expand.bias',	(480,),
 'block5.0.bottleneck.5.conv.weight',	(112, 480, 1, 1),	 '_blocks.8._project_conv.weight',	(112, 480, 1, 1),
 'block5.0.bottleneck.5.bn.weight',	(112,),	 '_blocks.8._bn2.weight',	(112,),
 'block5.0.bottleneck.5.bn.bias',	(112,),	 '_blocks.8._bn2.bias',	(112,),
 'block5.0.bottleneck.5.bn.running_mean',	(112,),	 '_blocks.8._bn2.running_mean',	(112,),
 'block5.0.bottleneck.5.bn.running_var',	(112,),	 '_blocks.8._bn2.running_var',	(112,),
 'block5.1.bottleneck.0.conv.weight',	(672, 112, 1, 1),	 '_blocks.9._expand_conv.weight',	(672, 112, 1, 1),
 'block5.1.bottleneck.0.bn.weight',	(672,),	 '_blocks.9._bn0.weight',	(672,),
 'block5.1.bottleneck.0.bn.bias',	(672,),	 '_blocks.9._bn0.bias',	(672,),
 'block5.1.bottleneck.0.bn.running_mean',	(672,),	 '_blocks.9._bn0.running_mean',	(672,),
 'block5.1.bottleneck.0.bn.running_var',	(672,),	 '_blocks.9._bn0.running_var',	(672,),
 'block5.1.bottleneck.2.conv.weight',	(672, 1, 5, 5),	 '_blocks.9._depthwise_conv.weight',	(672, 1, 5, 5),
 'block5.1.bottleneck.2.bn.weight',	(672,),	 '_blocks.9._bn1.weight',	(672,),
 'block5.1.bottleneck.2.bn.bias',	(672,),	 '_blocks.9._bn1.bias',	(672,),
 'block5.1.bottleneck.2.bn.running_mean',	(672,),	 '_blocks.9._bn1.running_mean',	(672,),
 'block5.1.bottleneck.2.bn.running_var',	(672,),	 '_blocks.9._bn1.running_var',	(672,),
 'block5.1.bottleneck.4.squeeze.weight',	(28, 672, 1, 1),	 '_blocks.9._se_reduce.weight',	(28, 672, 1, 1),
 'block5.1.bottleneck.4.squeeze.bias',	(28,),	 '_blocks.9._se_reduce.bias',	(28,),
 'block5.1.bottleneck.4.excite.weight',	(672, 28, 1, 1),	 '_blocks.9._se_expand.weight',	(672, 28, 1, 1),
 'block5.1.bottleneck.4.excite.bias',	(672,),	 '_blocks.9._se_expand.bias',	(672,),
 'block5.1.bottleneck.5.conv.weight',	(112, 672, 1, 1),	 '_blocks.9._project_conv.weight',	(112, 672, 1, 1),
 'block5.1.bottleneck.5.bn.weight',	(112,),	 '_blocks.9._bn2.weight',	(112,),
 'block5.1.bottleneck.5.bn.bias',	(112,),	 '_blocks.9._bn2.bias',	(112,),
 'block5.1.bottleneck.5.bn.running_mean',	(112,),	 '_blocks.9._bn2.running_mean',	(112,),
 'block5.1.bottleneck.5.bn.running_var',	(112,),	 '_blocks.9._bn2.running_var',	(112,),
 'block5.2.bottleneck.0.conv.weight',	(672, 112, 1, 1),	 '_blocks.10._expand_conv.weight',	(672, 112, 1, 1),
 'block5.2.bottleneck.0.bn.weight',	(672,),	 '_blocks.10._bn0.weight',	(672,),
 'block5.2.bottleneck.0.bn.bias',	(672,),	 '_blocks.10._bn0.bias',	(672,),
 'block5.2.bottleneck.0.bn.running_mean',	(672,),	 '_blocks.10._bn0.running_mean',	(672,),
 'block5.2.bottleneck.0.bn.running_var',	(672,),	 '_blocks.10._bn0.running_var',	(672,),
 'block5.2.bottleneck.2.conv.weight',	(672, 1, 5, 5),	 '_blocks.10._depthwise_conv.weight',	(672, 1, 5, 5),
 'block5.2.bottleneck.2.bn.weight',	(672,),	 '_blocks.10._bn1.weight',	(672,),
 'block5.2.bottleneck.2.bn.bias',	(672,),	 '_blocks.10._bn1.bias',	(672,),
 'block5.2.bottleneck.2.bn.running_mean',	(672,),	 '_blocks.10._bn1.running_mean',	(672,),
 'block5.2.bottleneck.2.bn.running_var',	(672,),	 '_blocks.10._bn1.running_var',	(672,),
 'block5.2.bottleneck.4.squeeze.weight',	(28, 672, 1, 1),	 '_blocks.10._se_reduce.weight',	(28, 672, 1, 1),
 'block5.2.bottleneck.4.squeeze.bias',	(28,),	 '_blocks.10._se_reduce.bias',	(28,),
 'block5.2.bottleneck.4.excite.weight',	(672, 28, 1, 1),	 '_blocks.10._se_expand.weight',	(672, 28, 1, 1),
 'block5.2.bottleneck.4.excite.bias',	(672,),	 '_blocks.10._se_expand.bias',	(672,),
 'block5.2.bottleneck.5.conv.weight',	(112, 672, 1, 1),	 '_blocks.10._project_conv.weight',	(112, 672, 1, 1),
 'block5.2.bottleneck.5.bn.weight',	(112,),	 '_blocks.10._bn2.weight',	(112,),
 'block5.2.bottleneck.5.bn.bias',	(112,),	 '_blocks.10._bn2.bias',	(112,),
 'block5.2.bottleneck.5.bn.running_mean',	(112,),	 '_blocks.10._bn2.running_mean',	(112,),
 'block5.2.bottleneck.5.bn.running_var',	(112,),	 '_blocks.10._bn2.running_var',	(112,),
 'block6.0.bottleneck.0.conv.weight',	(672, 112, 1, 1),	 '_blocks.11._expand_conv.weight',	(672, 112, 1, 1),
 'block6.0.bottleneck.0.bn.weight',	(672,),	 '_blocks.11._bn0.weight',	(672,),
 'block6.0.bottleneck.0.bn.bias',	(672,),	 '_blocks.11._bn0.bias',	(672,),
 'block6.0.bottleneck.0.bn.running_mean',	(672,),	 '_blocks.11._bn0.running_mean',	(672,),
 'block6.0.bottleneck.0.bn.running_var',	(672,),	 '_blocks.11._bn0.running_var',	(672,),
 'block6.0.bottleneck.2.conv.weight',	(672, 1, 5, 5),	 '_blocks.11._depthwise_conv.weight',	(672, 1, 5, 5),
 'block6.0.bottleneck.2.bn.weight',	(672,),	 '_blocks.11._bn1.weight',	(672,),
 'block6.0.bottleneck.2.bn.bias',	(672,),	 '_blocks.11._bn1.bias',	(672,),
 'block6.0.bottleneck.2.bn.running_mean',	(672,),	 '_blocks.11._bn1.running_mean',	(672,),
 'block6.0.bottleneck.2.bn.running_var',	(672,),	 '_blocks.11._bn1.running_var',	(672,),
 'block6.0.bottleneck.4.squeeze.weight',	(28, 672, 1, 1),	 '_blocks.11._se_reduce.weight',	(28, 672, 1, 1),
 'block6.0.bottleneck.4.squeeze.bias',	(28,),	 '_blocks.11._se_reduce.bias',	(28,),
 'block6.0.bottleneck.4.excite.weight',	(672, 28, 1, 1),	 '_blocks.11._se_expand.weight',	(672, 28, 1, 1),
 'block6.0.bottleneck.4.excite.bias',	(672,),	 '_blocks.11._se_expand.bias',	(672,),
 'block6.0.bottleneck.5.conv.weight',	(192, 672, 1, 1),	 '_blocks.11._project_conv.weight',	(192, 672, 1, 1),
 'block6.0.bottleneck.5.bn.weight',	(192,),	 '_blocks.11._bn2.weight',	(192,),
 'block6.0.bottleneck.5.bn.bias',	(192,),	 '_blocks.11._bn2.bias',	(192,),
 'block6.0.bottleneck.5.bn.running_mean',	(192,),	 '_blocks.11._bn2.running_mean',	(192,),
 'block6.0.bottleneck.5.bn.running_var',	(192,),	 '_blocks.11._bn2.running_var',	(192,),
 'block6.1.bottleneck.0.conv.weight',	(1152, 192, 1, 1),	 '_blocks.12._expand_conv.weight',	(1152, 192, 1, 1),
 'block6.1.bottleneck.0.bn.weight',	(1152,),	 '_blocks.12._bn0.weight',	(1152,),
 'block6.1.bottleneck.0.bn.bias',	(1152,),	 '_blocks.12._bn0.bias',	(1152,),
 'block6.1.bottleneck.0.bn.running_mean',	(1152,),	 '_blocks.12._bn0.running_mean',	(1152,),
 'block6.1.bottleneck.0.bn.running_var',	(1152,),	 '_blocks.12._bn0.running_var',	(1152,),
 'block6.1.bottleneck.2.conv.weight',	(1152, 1, 5, 5),	 '_blocks.12._depthwise_conv.weight',	(1152, 1, 5, 5),
 'block6.1.bottleneck.2.bn.weight',	(1152,),	 '_blocks.12._bn1.weight',	(1152,),
 'block6.1.bottleneck.2.bn.bias',	(1152,),	 '_blocks.12._bn1.bias',	(1152,),
 'block6.1.bottleneck.2.bn.running_mean',	(1152,),	 '_blocks.12._bn1.running_mean',	(1152,),
 'block6.1.bottleneck.2.bn.running_var',	(1152,),	 '_blocks.12._bn1.running_var',	(1152,),
 'block6.1.bottleneck.4.squeeze.weight',	(48, 1152, 1, 1),	 '_blocks.12._se_reduce.weight',	(48, 1152, 1, 1),
 'block6.1.bottleneck.4.squeeze.bias',	(48,),	 '_blocks.12._se_reduce.bias',	(48,),
 'block6.1.bottleneck.4.excite.weight',	(1152, 48, 1, 1),	 '_blocks.12._se_expand.weight',	(1152, 48, 1, 1),
 'block6.1.bottleneck.4.excite.bias',	(1152,),	 '_blocks.12._se_expand.bias',	(1152,),
 'block6.1.bottleneck.5.conv.weight',	(192, 1152, 1, 1),	 '_blocks.12._project_conv.weight',	(192, 1152, 1, 1),
 'block6.1.bottleneck.5.bn.weight',	(192,),	 '_blocks.12._bn2.weight',	(192,),
 'block6.1.bottleneck.5.bn.bias',	(192,),	 '_blocks.12._bn2.bias',	(192,),
 'block6.1.bottleneck.5.bn.running_mean',	(192,),	 '_blocks.12._bn2.running_mean',	(192,),
 'block6.1.bottleneck.5.bn.running_var',	(192,),	 '_blocks.12._bn2.running_var',	(192,),
 'block6.2.bottleneck.0.conv.weight',	(1152, 192, 1, 1),	 '_blocks.13._expand_conv.weight',	(1152, 192, 1, 1),
 'block6.2.bottleneck.0.bn.weight',	(1152,),	 '_blocks.13._bn0.weight',	(1152,),
 'block6.2.bottleneck.0.bn.bias',	(1152,),	 '_blocks.13._bn0.bias',	(1152,),
 'block6.2.bottleneck.0.bn.running_mean',	(1152,),	 '_blocks.13._bn0.running_mean',	(1152,),
 'block6.2.bottleneck.0.bn.running_var',	(1152,),	 '_blocks.13._bn0.running_var',	(1152,),
 'block6.2.bottleneck.2.conv.weight',	(1152, 1, 5, 5),	 '_blocks.13._depthwise_conv.weight',	(1152, 1, 5, 5),
 'block6.2.bottleneck.2.bn.weight',	(1152,),	 '_blocks.13._bn1.weight',	(1152,),
 'block6.2.bottleneck.2.bn.bias',	(1152,),	 '_blocks.13._bn1.bias',	(1152,),
 'block6.2.bottleneck.2.bn.running_mean',	(1152,),	 '_blocks.13._bn1.running_mean',	(1152,),
 'block6.2.bottleneck.2.bn.running_var',	(1152,),	 '_blocks.13._bn1.running_var',	(1152,),
 'block6.2.bottleneck.4.squeeze.weight',	(48, 1152, 1, 1),	 '_blocks.13._se_reduce.weight',	(48, 1152, 1, 1),
 'block6.2.bottleneck.4.squeeze.bias',	(48,),	 '_blocks.13._se_reduce.bias',	(48,),
 'block6.2.bottleneck.4.excite.weight',	(1152, 48, 1, 1),	 '_blocks.13._se_expand.weight',	(1152, 48, 1, 1),
 'block6.2.bottleneck.4.excite.bias',	(1152,),	 '_blocks.13._se_expand.bias',	(1152,),
 'block6.2.bottleneck.5.conv.weight',	(192, 1152, 1, 1),	 '_blocks.13._project_conv.weight',	(192, 1152, 1, 1),
 'block6.2.bottleneck.5.bn.weight',	(192,),	 '_blocks.13._bn2.weight',	(192,),
 'block6.2.bottleneck.5.bn.bias',	(192,),	 '_blocks.13._bn2.bias',	(192,),
 'block6.2.bottleneck.5.bn.running_mean',	(192,),	 '_blocks.13._bn2.running_mean',	(192,),
 'block6.2.bottleneck.5.bn.running_var',	(192,),	 '_blocks.13._bn2.running_var',	(192,),
 'block6.3.bottleneck.0.conv.weight',	(1152, 192, 1, 1),	 '_blocks.14._expand_conv.weight',	(1152, 192, 1, 1),
 'block6.3.bottleneck.0.bn.weight',	(1152,),	 '_blocks.14._bn0.weight',	(1152,),
 'block6.3.bottleneck.0.bn.bias',	(1152,),	 '_blocks.14._bn0.bias',	(1152,),
 'block6.3.bottleneck.0.bn.running_mean',	(1152,),	 '_blocks.14._bn0.running_mean',	(1152,),
 'block6.3.bottleneck.0.bn.running_var',	(1152,),	 '_blocks.14._bn0.running_var',	(1152,),
 'block6.3.bottleneck.2.conv.weight',	(1152, 1, 5, 5),	 '_blocks.14._depthwise_conv.weight',	(1152, 1, 5, 5),
 'block6.3.bottleneck.2.bn.weight',	(1152,),	 '_blocks.14._bn1.weight',	(1152,),
 'block6.3.bottleneck.2.bn.bias',	(1152,),	 '_blocks.14._bn1.bias',	(1152,),
 'block6.3.bottleneck.2.bn.running_mean',	(1152,),	 '_blocks.14._bn1.running_mean',	(1152,),
 'block6.3.bottleneck.2.bn.running_var',	(1152,),	 '_blocks.14._bn1.running_var',	(1152,),
 'block6.3.bottleneck.4.squeeze.weight',	(48, 1152, 1, 1),	 '_blocks.14._se_reduce.weight',	(48, 1152, 1, 1),
 'block6.3.bottleneck.4.squeeze.bias',	(48,),	 '_blocks.14._se_reduce.bias',	(48,),
 'block6.3.bottleneck.4.excite.weight',	(1152, 48, 1, 1),	 '_blocks.14._se_expand.weight',	(1152, 48, 1, 1),
 'block6.3.bottleneck.4.excite.bias',	(1152,),	 '_blocks.14._se_expand.bias',	(1152,),
 'block6.3.bottleneck.5.conv.weight',	(192, 1152, 1, 1),	 '_blocks.14._project_conv.weight',	(192, 1152, 1, 1),
 'block6.3.bottleneck.5.bn.weight',	(192,),	 '_blocks.14._bn2.weight',	(192,),
 'block6.3.bottleneck.5.bn.bias',	(192,),	 '_blocks.14._bn2.bias',	(192,),
 'block6.3.bottleneck.5.bn.running_mean',	(192,),	 '_blocks.14._bn2.running_mean',	(192,),
 'block6.3.bottleneck.5.bn.running_var',	(192,),	 '_blocks.14._bn2.running_var',	(192,),
 'block7.0.bottleneck.0.conv.weight',	(1152, 192, 1, 1),	 '_blocks.15._expand_conv.weight',	(1152, 192, 1, 1),
 'block7.0.bottleneck.0.bn.weight',	(1152,),	 '_blocks.15._bn0.weight',	(1152,),
 'block7.0.bottleneck.0.bn.bias',	(1152,),	 '_blocks.15._bn0.bias',	(1152,),
 'block7.0.bottleneck.0.bn.running_mean',	(1152,),	 '_blocks.15._bn0.running_mean',	(1152,),
 'block7.0.bottleneck.0.bn.running_var',	(1152,),	 '_blocks.15._bn0.running_var',	(1152,),
 'block7.0.bottleneck.2.conv.weight',	(1152, 1, 3, 3),	 '_blocks.15._depthwise_conv.weight',	(1152, 1, 3, 3),
 'block7.0.bottleneck.2.bn.weight',	(1152,),	 '_blocks.15._bn1.weight',	(1152,),
 'block7.0.bottleneck.2.bn.bias',	(1152,),	 '_blocks.15._bn1.bias',	(1152,),
 'block7.0.bottleneck.2.bn.running_mean',	(1152,),	 '_blocks.15._bn1.running_mean',	(1152,),
 'block7.0.bottleneck.2.bn.running_var',	(1152,),	 '_blocks.15._bn1.running_var',	(1152,),
 'block7.0.bottleneck.4.squeeze.weight',	(48, 1152, 1, 1),	 '_blocks.15._se_reduce.weight',	(48, 1152, 1, 1),
 'block7.0.bottleneck.4.squeeze.bias',	(48,),	 '_blocks.15._se_reduce.bias',	(48,),
 'block7.0.bottleneck.4.excite.weight',	(1152, 48, 1, 1),	 '_blocks.15._se_expand.weight',	(1152, 48, 1, 1),
 'block7.0.bottleneck.4.excite.bias',	(1152,),	 '_blocks.15._se_expand.bias',	(1152,),
 'block7.0.bottleneck.5.conv.weight',	(320, 1152, 1, 1),	 '_blocks.15._project_conv.weight',	(320, 1152, 1, 1),
 'block7.0.bottleneck.5.bn.weight',	(320,),	 '_blocks.15._bn2.weight',	(320,),
 'block7.0.bottleneck.5.bn.bias',	(320,),	 '_blocks.15._bn2.bias',	(320,),
 'block7.0.bottleneck.5.bn.running_mean',	(320,),	 '_blocks.15._bn2.running_mean',	(320,),
 'block7.0.bottleneck.5.bn.running_var',	(320,),	 '_blocks.15._bn2.running_var',	(320,),
 'last.0.conv.weight',	(1280, 320, 1, 1),	 '_conv_head.weight',	(1280, 320, 1, 1),
 'last.0.bn.weight',	(1280,),	 '_bn1.weight',	(1280,),
 'last.0.bn.bias',	(1280,),	 '_bn1.bias',	(1280,),
 'last.0.bn.running_mean',	(1280,),	 '_bn1.running_mean',	(1280,),
 'last.0.bn.running_var',	(1280,),	 '_bn1.running_var',	(1280,),
 'logit.weight',	(1000, 1280),	 '_fc.weight',	(1000, 1280),
 'logit.bias',	(1000,),	 '_fc.bias',	(1000,),

]
PRETRAIN_FILE = '/root/share/project/kaggle/2019/chest/data/__download__/lukemelas/efficientnet-b0-355c32eb.pth'
def load_pretrain(net, skip=[], pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=True):

    #raise NotImplementedError
    print('\tload pretrain_file: %s'%pretrain_file)

    #pretrain_state_dict = torch.load(pretrain_file)
    pretrain_state_dict = torch.load(pretrain_file, map_location=lambda storage, loc: storage)
    state_dict = net.state_dict()

    i = 0
    conversion = np.array(CONVERSION).reshape(-1,4)
    for key,_,pretrain_key,_ in conversion:
        if any(s in key for s in
            ['.num_batches_tracked',]+skip):
            continue

        #print('\t\t',key)
        if is_print:
            print('\t\t','%-48s  %-24s  <---  %-32s  %-24s'%(
                key, str(state_dict[key].shape),
                pretrain_key, str(pretrain_state_dict[pretrain_key].shape),
            ))
        i = i+1

        state_dict[key] = pretrain_state_dict[pretrain_key]

    net.load_state_dict(state_dict)
    print('')
    print('len(pretrain_state_dict.keys()) = %d'%len(pretrain_state_dict.keys()))
    print('len(state_dict.keys())          = %d'%len(state_dict.keys()))
    print('loaded    = %d'%i)
    print('')


### efficientnet #######################################################################

def drop_connect(x, probability, training):
    if not training: return x

    batch_size = len(x)
    keep_probability = 1 - probability
    noise = keep_probability
    noise += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    mask = torch.floor(noise)
    x = x / keep_probability * mask

    return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Identity(nn.Module):
    def forward(self, x):
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(len(x),-1)

class Conv2dBn(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, zero_pad=[0,0,0,0], group=1):
        super(Conv2dBn, self).__init__()
        if IS_PYTORCH_PAD: zero_pad = [kernel_size//2]*4
        self.pad  = nn.ZeroPad2d(zero_pad)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=0, stride=stride, groups=group, bias=False)
        self.bn   = BatchNorm2d(out_channel, eps=1e-03, momentum=0.01)
        #print(zero_pad)

    def forward(self, x):
        x = self.pad (x)
        x = self.conv(x)
        x = self.bn  (x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_channel, reduction_channel, excite_size):
        super(SqueezeExcite, self).__init__()
        self.excite_size=excite_size

        self.squeeze = nn.Conv2d(in_channel, reduction_channel, kernel_size=1, padding=0)
        self.excite  = nn.Conv2d(reduction_channel, in_channel, kernel_size=1, padding=0)
        self.act = Swish()

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x,1)
        s = self.act(self.squeeze(s))
        s = torch.sigmoid(self.excite(s))

        x = s*x
        return x


    # def forward(self, x):
    #
    #     s = F.avg_pool2d(x, kernel_size=self.excite_size)
    #     s = self.act(self.squeeze(s))
    #     s = torch.sigmoid(self.excite(s))
    #     s = F.interpolate(s, size=(x.shape[2],x.shape[3]), mode='nearest')
    #
    #     x = s*x
    #     return x

#---

class EfficientBlock(nn.Module):

    def __init__(self, in_channel, channel, out_channel, kernel_size, stride, zero_pad, excite_size, drop_connect_rate):
        super().__init__()
        self.is_shortcut = stride == 1 and in_channel == out_channel
        self.drop_connect_rate = drop_connect_rate

        if in_channel == channel:
            self.bottleneck = nn.Sequential(
                Conv2dBn(   channel, channel, kernel_size=kernel_size, stride=stride, zero_pad=zero_pad, group=channel),
                Swish(),
                SqueezeExcite(channel, in_channel//4, excite_size) if excite_size>0
                else Identity(),
                Conv2dBn(channel, out_channel, kernel_size=1, stride=1),
            )
        else:
            self.bottleneck = nn.Sequential(
                Conv2dBn(in_channel, channel, kernel_size=1, stride=1),
                Swish(),
                Conv2dBn(   channel, channel, kernel_size=kernel_size, stride=stride, zero_pad=zero_pad, group=channel),
                Swish(),
                SqueezeExcite(channel, in_channel//4, excite_size) if excite_size>0
                else Identity(),
                Conv2dBn(channel, out_channel, kernel_size=1, stride=1)
            )

    def forward(self, x):
        b = self.bottleneck(x)

        if self.is_shortcut:
            if self.training: b = drop_connect(b, self.drop_connect_rate, True)
            x = b + x
        else:
            x = b
        return x



# https://arogozhnikov.github.io/einops/pytorch-examples.html

# actual padding used in tensorflow
class EfficientNet(nn.Module):

    def __init__(self, drop_connect_rate=0.2):
        super(EfficientNet, self).__init__()
        d = drop_connect_rate

        # bottom-top
        self.stem  = nn.Sequential(
            Conv2dBn(3,32, kernel_size=3,stride=2,zero_pad=[0,1,0,1]),
            Swish()
        )

        self.block1 = nn.Sequential(
               EfficientBlock( 32,  32,  16, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size=128, drop_connect_rate=d*1/7),
        )
        self.block2 = nn.Sequential(
               EfficientBlock( 16,  96,  24, kernel_size=3, stride=2, zero_pad=[0,1,0,1], excite_size= 64, drop_connect_rate=d*2/7),
            * [EfficientBlock( 24, 144,  24, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size= 64, drop_connect_rate=d*2/7) for i in range(1,2)],
        )
        self.block3 = nn.Sequential(
               EfficientBlock( 24, 144,  40, kernel_size=5, stride=2, zero_pad=[1,2,1,2], excite_size= 32, drop_connect_rate=d*3/7),
            * [EfficientBlock( 40, 240,  40, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size= 32, drop_connect_rate=d*3/7) for i in range(1,2)],
        )
        self.block4 = nn.Sequential(
               EfficientBlock( 40, 240,  80, kernel_size=3, stride=2, zero_pad=[0,1,0,1], excite_size= 16, drop_connect_rate=d*4/7),
            * [EfficientBlock( 80, 480,  80, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size= 16, drop_connect_rate=d*4/7) for i in range(1,3)],
        )
        self.block5 = nn.Sequential(
               EfficientBlock( 80, 480, 112, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size= 16, drop_connect_rate=d*5/7),
            * [EfficientBlock(112, 672, 112, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size= 16, drop_connect_rate=d*5/7) for i in range(1,3)],
        )
        self.block6 = nn.Sequential(
               EfficientBlock(112, 672, 192, kernel_size=5, stride=2, zero_pad=[1,2,1,2], excite_size=  8, drop_connect_rate=d*6/7),
            * [EfficientBlock(192,1152, 192, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size=  8, drop_connect_rate=d*6/7) for i in range(1,4)],
        )
        self.block7 = nn.Sequential(
               EfficientBlock(192,1152, 320, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size=  8, drop_connect_rate=d*7/7),
        )
        self.last = nn.Sequential(
            Conv2dBn(320,1280, kernel_size=1,stride=1),
            Swish()
        )

        self.logit = nn.Linear(1280,1000)

    def forward(self, x):
        batch_size = len(x)

        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.last(x)

        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = self.logit(x)

        return logit




#########################################################################
def run_check_efficientnet():
    net = EfficientNet()
    #print(net)


    if 0:
        print('*** print key *** ')
        state_dict = net.state_dict()
        keys = list(state_dict.keys())
        #keys = sorted(keys)
        for k in keys:
            if any(s in k for s in [
                'num_batches_tracked'
                # '.kernel',
                # '.gamma',
                # '.beta',
                # '.running_mean',
                # '.running_var',
            ]):
                continue

            p = state_dict[k].data.cpu().numpy()
            print(' \'%s\',\t%s,'%(k,tuple(p.shape)))
        print('')
        exit(0)

    load_pretrain(net, is_print=False)
    if 1:
        IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
        IMAGE_RGB_STD  = [0.229, 0.224, 0.225]

        net = net.cuda().eval()


        image_dir ='/root/share/data/imagenet/dummy/256x256'
        for f in [
            'great_white_shark','screwdriver','ostrich','blad_eagle','english_foxhound','goldfish',
        ]:
            image = cv2.imread(image_dir +'/%s.jpg'%f, cv2.IMREAD_COLOR)
            image = (image.astype(np.float32)/255 -IMAGE_RGB_MEAN)/IMAGE_RGB_STD

            input = image.transpose(2,0,1)
            input = input.reshape(1,3,256,256)
            input = torch.from_numpy(input).float().cuda()

            logit = net(input)
            proability = F.softmax(logit,-1)

            probability = proability.data.cpu().numpy().reshape(-1)
            argsort = np.argsort(-probability)

            print(probability[:5])
            for t in range(5):
                print(t, argsort[t], probability[argsort[t]])
            print('')

            pass

'''

[8.7754931e-03 8.2669204e-01 8.3855812e-05 5.7508185e-04 2.7134272e-04]
0 1 0.82669204
1 392 0.036001623
2 0 0.008775493
3 395 0.0059498292
4 973 0.005688266

'''

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_efficientnet()


    print('\nsucess!')
