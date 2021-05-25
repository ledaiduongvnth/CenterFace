import torch.nn.functional as F
from model.multi_pose import MultiPoseLoss
import pickle

opt = None
with open('/home/thanhnn/PycharmProjects/CenterFace/data_loader/opt.pkl', 'rb') as f:
    opt = pickle.load(f)

multi_pose_loss = MultiPoseLoss(opt)


def center_face_loss(output, target):
    return multi_pose_loss.forward(output, target)


def nll_loss(output, target):
    return F.nll_loss(output, target)
