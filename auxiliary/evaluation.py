import torch
from tqdm import tqdm
import numpy as np

def calOA(mtx):
    OA = np.sum(np.diag(mtx))/np.sum(mtx)
    return OA
def calKappa(mtx):
    p0 = calOA(mtx)
    pe = np.sum(np.sum(mtx,axis=1)*np.diag(mtx))/(np.sum(mtx)**2)
    kappa = (p0 - pe)/(1 - pe)
    return kappa
def calAA(mtx):
    class_accuracies = np.diag(mtx) / np.sum(mtx, axis=1)
    class_accuracies = class_accuracies[~np.isnan(class_accuracies)]
    aa = np.mean(class_accuracies)
    return aa


def obj_evaluate(dataloader,device,model,criterion,args,filename=None):
    model.eval()
    total_loss, total = 0., 0
    confusion_mtx = torch.zeros((17,17))

    with tqdm(total=args.nval,desc='inferring') as pbar:
        for batch, (img, g_poi, obj_feats, bs_aoi_labels, graphid) in enumerate(dataloader):
            img = img.to(device)
            g_poi = g_poi.to(device)
            obj_feats = obj_feats.to(device)
            bs_aoi_labels = bs_aoi_labels.to(device)
            logits_parcel = model(img,g_poi,g_poi.ndata.pop('attr'),obj_feats,bs_aoi_labels,stat='val')

            _, predicted = torch.max(logits_parcel,dim=1)
            loss = criterion(logits_parcel, bs_aoi_labels.long())

            pbar.update(len(bs_aoi_labels))
            total_loss += loss.item()
            total += 1
            for i, pred in enumerate(predicted):
                confusion_mtx[bs_aoi_labels[i].cpu().data,pred.cpu().data] += 1

    confusion_mtx = np.array(confusion_mtx)
    total_loss /= total
    kappa = calKappa(confusion_mtx)
    acc = calOA(confusion_mtx)
    if filename is not None:
        np.savetxt(filename,confusion_mtx)

    return acc, kappa, total_loss


def AA_obj_evaluate(dataloader,device,model,criterion,args,filename=None):
    model.eval()
    total_loss, total = 0., 0
    confusion_mtx = torch.zeros((17,17))
    total_samples = 0
    with tqdm(total=args.nval,desc='inferring') as pbar:
        for batch, (img, g_poi, obj_feats, bs_aoi_labels, graphid) in enumerate(dataloader):
            img = img.to(device)
            g_poi = g_poi.to(device)
            obj_feats = obj_feats.to(device)
            bs_aoi_labels = bs_aoi_labels.to(device)
            logits_parcel = model(img,g_poi,g_poi.ndata.pop('attr'),obj_feats,bs_aoi_labels,stat='val')

            _, predicted = torch.max(logits_parcel,dim=1)
            loss = criterion(logits_parcel, bs_aoi_labels.long())

            pbar.update(len(bs_aoi_labels))
            total_loss += loss.item()
            total += 1
            total_samples += len(bs_aoi_labels)
            for i, pred in enumerate(predicted):
                confusion_mtx[bs_aoi_labels[i].cpu().data,pred.cpu().data] += 1

    confusion_mtx = np.array(confusion_mtx)
    total_loss /= total
    kappa = calKappa(confusion_mtx)
    acc = calOA(confusion_mtx)
    aa = calAA(confusion_mtx)
    if filename is not None:
        np.savetxt(filename,confusion_mtx)

    return acc, kappa, aa, total_loss
