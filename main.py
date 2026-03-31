import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import random
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from auxiliary.dataset import IMG_OBJ_POI_Dataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model.LUCModel import VLMLU_model
from torchvision import transforms
from auxiliary.evaluation import obj_evaluate,AA_obj_evaluate
import warnings
warnings.filterwarnings('ignore')



def parse_args():
    parser = argparse.ArgumentParser(description='VLM-LU')
    parser.add_argument('--epoch',type=int,default=15)
    parser.add_argument('--bs',type=int,default=4,help='batch size')
    parser.add_argument('--ntrain', type=int, help='number of training samples')
    parser.add_argument('--nval', type=int, help='number of validation samples')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--gam', type=float, default=1.5, help='calibration hyperparameter')
    parser.add_argument('--labelPath', type=str,help='/path/to/data/Multi_CUN_labels.csv')
    parser.add_argument('--graphPath', type=str,help='/path/to/data/Multi_CUN_GraphData.txt')
    parser.add_argument('--graphIndexPath', type=str,help='/path/to/data/Multi_CUN_GraphIndexData.txt')
    parser.add_argument('--objPath', type=str,help='/path/to/data/Multi_CUN_obj_features')
    parser.add_argument('--imgPath', type=str, help='/path/to/data/Multi_CUN_obj_features')

    args = parser.parse_args()
    return args
def dataSplit(aoi_labels,train_ratio,seed=None):
    aoi_labels = np.array(aoi_labels)
    train_idx, val_idx = [], []
    aoi_classes_count = np.bincount(aoi_labels)
    if seed != None:
        np.random.seed(seed)
    for aoi_label, num in enumerate(aoi_classes_count):
        aoi_label_index = np.where(aoi_labels==aoi_label)[0]
        # 打乱顺序
        index = np.random.permutation(aoi_label_index.size)
        aoi_label_index = aoi_label_index[index]
        train_idx = train_idx + aoi_label_index[:int(train_ratio*num)].tolist()

        val_idx = val_idx + aoi_label_index[int(train_ratio*num):].tolist()
    return train_idx, val_idx

    
def train(model,train_loader,val_loader,args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_acc, best_epoch, best_kappa = 0., 0, 0.
    criterion_CE = nn.CrossEntropyLoss()
    criterion_MSE = nn.MSELoss()
    model.train()
    accs, kappas, aas = [], [], []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step_losses = []
    for epoch in range(args.epoch):
        total_loss, count = 0., 0
        obj_features = []
        aoi_labels = []
        with tqdm(total=args.ntrain,desc=f'training {epoch+1}/{args.epoch}') as pbar:
            for img, g_poi, obj_feats, bs_aoi_labels, graphid in train_loader:
                img = img.to(device)
                g_poi = g_poi.to(device)
                obj_feats = obj_feats.to(device)
                bs_aoi_labels = bs_aoi_labels.to(device)

                logits_parcel, hrs_output, poi_output, bs_obj_features, w_hrs, w_hrs_pred, w_poi, w_poi_pred = \
                model(img,g_poi,g_poi.ndata.pop('attr'),obj_feats,bs_aoi_labels)

                obj_features.append(bs_obj_features.data)
                aoi_labels.append(bs_aoi_labels.data)

                L_Conf = criterion_CE(hrs_output,bs_aoi_labels.long()) + criterion_CE(poi_output,bs_aoi_labels.long()) + criterion_MSE(w_hrs,w_hrs_pred) + criterion_MSE(w_poi,w_poi_pred)
                L_CE = criterion_CE(logits_parcel,bs_aoi_labels.long())
                loss = L_CE + args.gam*L_Conf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                step_losses.append(loss.item())
                count += 1
                pbar.update(len(bs_aoi_labels))

        train_acc, train_kappa, _ = obj_evaluate(train_loader,device,model,criterion_CE,args)
        val_acc, val_kappa, val_aa, val_loss = AA_obj_evaluate(val_loader,device,model,criterion_CE,args)
        print(
            "Epoch {:05d} | Loss {:.3f} | ValLoss {:.3f} | Train Acc. {:.4f} | Train Kappa. {:.4f} | Val Acc. {:.4f}| Val Kappa. {:.4f}| Val AA. {:.4f}  ".format(
                epoch, total_loss / count, val_loss, train_acc, train_kappa, val_acc, val_kappa, val_aa
            )
        )
        logging.info(
            "Epoch {:05d} | Loss {:.3f} | ValLoss {:.3f} | Train Acc. {:.4f} | Train Kappa. {:.4f} | Val Acc. {:.4f}| Val Kappa. {:.4f}| Val AA. {:.4f}  ".format(
                epoch, total_loss / count, val_loss, train_acc, train_kappa, val_acc, val_kappa, val_aa
            )
        )
        accs.append(val_acc)
        kappas.append(val_kappa)
        aas.append(val_aa)

        if val_acc > best_acc:
            best_acc = val_acc
            best_kappa = val_kappa
            best_epoch = epoch
            torch.save(model.state_dict(),f'./ckpt/Multi_CUN_ep{epoch}_acc{best_acc}.pth')

    accs, kappas = np.array(accs), np.array(kappas)
    aas = np.array(aas)
    accs.sort()
    kappas.sort()
    aas.sort()
    accs = accs[::-1]
    kappas = kappas[::-1]
    aas = aas[::-1]
    accs, kappas, aas = accs[:3], kappas[:3], aas[:3]


    return accs, kappas, aas

def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True,warn_only=True)



def main(args):
    set_random_seed(3407)
    data = pd.read_csv(args.labelPath,dtype={'id':str})
    data['id'] = data['id'].apply(lambda x: x[2:])
    data = data.drop_duplicates(subset='id')
    data = data.sort_values(by='id')
    valid_aoi_id = np.loadtxt(args.graphIndexPath, dtype=str)
    valid_aoi_id = valid_aoi_id[1:]
    data = data[data['id'].apply(lambda x: valid_aoi_id.__contains__(x))]
    valid_AOI_withPOIs_id, AOI_withPOI_label = data['id'].values, data['class'].values  
    train_idx, val_idx = dataSplit(aoi_labels=AOI_withPOI_label,train_ratio=0.7,seed=3407)

    obj_poi_dataset = IMG_OBJ_POI_Dataset(
        objFeatPath=args.objPath,
        imgPath=args.imgPath,
        transform=transforms.Compose([transforms.ToTensor()]),
        labels=AOI_withPOI_label,
        raw_dir=args.graphPath,
        save_dir=rf'./data/',
        cityName=f'Multi_CUN',
    )

    args.total_num = len(obj_poi_dataset)
    args.ntrain, args.nval = len(train_idx), len(val_idx)

    train_loader = GraphDataLoader(
        obj_poi_dataset,
        sampler=SubsetRandomSampler(np.array(train_idx)),
        batch_size=args.bs,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    val_loader = GraphDataLoader(
        obj_poi_dataset,
        sampler=SubsetRandomSampler(np.array(val_idx)),
        batch_size=args.bs,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VLMLU_model(poi_in_channels=23,obj_in_channels=768,hid_channels=128,
                 out_channels=17).to(device)

    logPath = f'./log/VLM_LU_Multi_CUN_training_log.txt'
    if not os.path.exists(logPath):
        os.makedirs(logPath)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s',
                       filename=logPath)

    best_acc, best_kappa, best_aa = train(model,train_loader,val_loader,args)
    
    print('************************************')
    print(
       'oa={:.4f}, kappa={:.4f}, aa={:.4f}'.format(np.mean(best_acc),np.mean(best_kappa),np.mean(best_aa)))
    print('************************************')

    logging.info('************************************')
    logging.info(
       'oa={:.4f}, kappa={:.4f}, aa={:.4f}'.format(np.mean(best_acc),np.mean(best_kappa),np.mean(best_aa)))
    logging.info('************************************')


if __name__ == '__main__':
    args = parse_args()
    main(args)



