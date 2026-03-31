import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.gnn_model import GAT
from model.resnet import ResNet, Bottleneck



class MSA(nn.Module):
    def __init__(self,num_heads,hidden_size):
        super().__init__()
        self.num_attention_heads = num_heads # 8
        self.attention_head_size = int(hidden_size/num_heads) # 32
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 8 x 32

        self.query = nn.Linear(hidden_size,self.all_head_size)
        self.key = nn.Linear(hidden_size,self.all_head_size)
        self.value = nn.Linear(hidden_size,self.all_head_size)

        self.dense = nn.Linear(hidden_size,hidden_size)

    def transpose_for_scores(self,x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self,x1,x2):
        mixed_query_layer = self.query(x1)  # bs x hidden_size
        mixed_key_layer = self.key(x2)
        mixed_value_layer = self.value(x2)

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # bs x num_heads x hidden_size
        key_layer = self.transpose_for_scores(mixed_key_layer)  # bs x num_heads x hidden_size
        value_layer = self.transpose_for_scores(
            mixed_value_layer)  # bs x num_heads x hidden_size

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
                                                                         -2))  # bs x num_heads x hidden_size
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)  # bs x num_heads x hidden_size
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # bs x num_heads x hidden_size
        context_layer = torch.matmul(attention_probs,
                                     value_layer)  # bs x num_heads x hidden_size
        context_layer = context_layer.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
        self.all_head_size,)  # bs x num_heads x hidden_size
        context_layer = context_layer.view(*new_context_layer_shape)  # bs x num_heads x hidden_size

        output = self.dense(context_layer)

        return output, mixed_key_layer, mixed_query_layer

class VLMLU_model(nn.Module):
    def __init__(self, poi_in_channels, obj_in_channels, hid_channels, out_channels):
        super().__init__()
        self.POIGCN = GAT(poi_in_channels, hid_channels, hid_channels)
        self.OBJMLP = nn.Sequential(
            nn.Conv1d(obj_in_channels, hid_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hid_channels // 2, hid_channels, kernel_size=1),
        )

        self.IMGEncoder = ResNet(Bottleneck, [3, 4, 6, 3])
        self.IMGEncoder.load_state_dict(torch.load('./ckpt/resnet50-19c8e357.pth'))
        self.IMGEncoder.fc = nn.Linear(2048, hid_channels)

        self.g_poi = nn.Linear(out_channels, 1)
        self.g_hrs = nn.Linear(out_channels, 1)

        self.hrs_head = nn.Linear(hid_channels, out_channels)
        self.poi_head = nn.Linear(hid_channels, out_channels)
        self.fc = nn.Linear(3 * hid_channels, out_channels)
        self.alpha = nn.Parameter(torch.ones((1,hid_channels,21)))

        self.JointProj = nn.Linear(2 * hid_channels, hid_channels)
        self.JointAttn = MSA(num_heads=4, hidden_size=hid_channels)
        self.JointAlpha = nn.Parameter(torch.ones(1,1))
        self.MLPJ = nn.Sequential(
            nn.Linear(hid_channels, hid_channels // 2),
            nn.GELU(),
            nn.Linear(hid_channels // 2, hid_channels)
        )

    def joint_guided(self, obj_features, poi_features, visual_features):
        obj_features = torch.sum(self.alpha * obj_features, dim=2)
        joint_features = torch.concat([obj_features, poi_features], dim=1)
        joint_features = self.JointProj(joint_features)
        f_j2v, _, _ = self.JointAttn(joint_features, visual_features)
        f_j2v = self.JointAlpha * f_j2v + visual_features

        y = self.MLPJ(f_j2v)

        return f_j2v + y

    def forward(self, img, g_poi, g_poi_features, ori_obj_matrix, y=None, stat='train'):
        visual_features = self.IMGEncoder(img)  # N x d
        
        obj_matrix = ori_obj_matrix.transpose(1, 2)
        obj_matrix = self.OBJMLP(obj_matrix)  # N x C x d
        poi_features = self.POIGCN(g_poi, g_poi_features)

        
        obj_features = torch.sum(self.alpha * obj_matrix, dim=2)
        
        hrs_output = self.hrs_head(visual_features)
        poi_output = self.poi_head(poi_features)

        joint_guided_features = self.joint_guided(obj_matrix, poi_features, visual_features)
        visual_features = joint_guided_features

        if stat == 'train':
            one_hot_labels = F.one_hot(y, num_classes=17)
            w_hrs = F.sigmoid(hrs_output) * one_hot_labels
            w_hrs = w_hrs.sum(dim=1)
            w_poi = F.sigmoid(poi_output) * one_hot_labels
            w_poi = w_poi.sum(dim=1)
            w_hrs_pred = self.g_hrs(hrs_output)
            w_hrs_pred = w_hrs_pred.sum(dim=1)
            w_poi_pred = self.g_poi(poi_output)
            w_poi_pred = w_poi_pred.sum(dim=1)

            output = torch.concat(
                [w_poi_pred.unsqueeze(1) * poi_features, w_hrs_pred.unsqueeze(1) * visual_features, obj_features],
                dim=1)

            output = self.fc(output)

            return output, hrs_output, poi_output, obj_features, w_hrs, w_hrs_pred, w_poi, w_poi_pred

        else:
            w_hrs = self.g_hrs(hrs_output)
            w_hrs = w_hrs.sum(dim=1)
            w_poi = self.g_poi(poi_output)
            w_poi = w_poi.sum(dim=1)
            output = torch.concat(
                [w_poi.unsqueeze(1) * poi_features, w_hrs.unsqueeze(1) * visual_features, obj_features], dim=1)

            output = self.fc(output)

            return output

