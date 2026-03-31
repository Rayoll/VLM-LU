import os
import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info
from PIL import Image
from dgl.convert import graph as dgl_graph
from dgl import backend as F
import json
import warnings
warnings.filterwarnings('ignore')

class IMG_OBJ_POI_Dataset(DGLDataset):
    def __init__(self,
                 objFeatPath,
                 imgPath,
                 labels,
                 transform=None,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False,
                 cityName=None):

        self.objFeatPath = objFeatPath
        self.imgPath = imgPath
        self.labels = labels
        self.transform = transform

        self.cityName = cityName

        self.POI_graphs = [] # graphs
        self.OBJ_feats = []
        self.labels = [] # graphs label
        self.graphID = []  # graphs id

        self.glabel_dict = {}
        self.nlabel_dict = {}

        super().__init__(
            name='dataset_name',
            url=url,
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose
        )

    def __len__(self):
        return len(self.POI_graphs)

    def __getitem__(self,idx):
        img_path = os.path.join(self.imgPath,self.graphID[idx]+'.png')
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        g_poi = self.POI_graphs[idx]
        obj_feats = self.OBJ_feats[idx]
        obj_feats = obj_feats.float()

        return img, g_poi, obj_feats, self.labels[idx], self.graphID[idx]

    def process(self):
        self.file = self.raw_dir
        # POI Graphs
        with open(self.file,mode='r',encoding='utf-8') as f:
            graphNum = int(f.readline().strip())
            for i in range(graphNum):
                n_nodes, glabel, graph_id = f.readline().strip().split() # graph info
                n_nodes, glabel = int(n_nodes), int(glabel)

                self.graphID.append(graph_id)
                self.labels.append(glabel)
                # create dgl graph
                g = dgl_graph(([],[]))
                g.add_nodes(n_nodes)
                # node labels
                nlabels = []
                m_edges = 0
                for j in range(n_nodes):
                    nrow = f.readline().strip().split()
                    nrow = [int(w) for w in nrow]
                    if not nrow[0] in self.nlabel_dict:
                        self.nlabel_dict[nrow[0]] = nrow[0]
                    nlabels.append(nrow[0])
                    m_edges += nrow[1]
                    g.add_edges(j,nrow[2:])
                    # add self loop
                    m_edges += 1
                    g.add_edges(j,j)
                    # info print
                    if (j+1)%10 == 0 and self.verbose is True:
                        print(
                            'processing node {} of graph {}...'.format(
                                j+1,i+1
                            ))
                        print('this node has {} edges'.format(nrow[1]))

                g.ndata['label'] =F.tensor(nlabels)
                self.POI_graphs.append(g)

        for i in range(17):
            self.glabel_dict[i] = i
        self.labels = F.tensor(self.labels)
        nlabel_dict = {}
        '''edit'''
        for i in range(23):
            nlabel_dict[i] = i

        for g in self.POI_graphs:
            attr = np.zeros((
                g.number_of_nodes(),len(nlabel_dict)
            ))
            attr[range(g.number_of_nodes()),[nlabel_dict[nl] for nl in F.asnumpy(g.ndata['label']).tolist()]] = 1
            g.ndata['attr'] = F.tensor(attr, F.float32)

        # OBJ Graphs
        for gid in self.graphID:
            objFeats = torch.load(os.path.join(self.objFeatPath,f'{gid}.tensor'))
            self.OBJ_feats.append(objFeats.cpu().data)


    def save(self):
        POI_graph_path = os.path.join(self.save_dir,f'{self.cityName}_POIGraph_gin.bin')
        info_path = os.path.join(self.save_dir,f'{self.cityName}_gin.pkl')
        label_dict = {'labels':self.labels}
        info_dict = {
            'graph_ids':self.graphID,
            'glabel_dict':self.glabel_dict,
            'nlabel_dict':self.nlabel_dict,
            'obj_feats':self.OBJ_feats,
        }
        save_graphs(str(POI_graph_path),self.POI_graphs,label_dict)
        save_info(str(info_path),info_dict)

    def load(self):
        POI_graph_path = os.path.join(self.save_dir, f'{self.cityName}_POIGraph_gin.bin')
        OBJ_graph_path = os.path.join(self.save_dir, f'{self.cityName}_OBJGraph_gin.bin')
        info_path = os.path.join(self.save_dir,f'{self.cityName}_gin.pkl')
        POI_graphs, label_dict = load_graphs(str(POI_graph_path))
        OBJ_graphs, _ = load_graphs(str(OBJ_graph_path))
        info_dict = load_info(str(info_path))

        self.POI_graphs = POI_graphs
        self.OBJ_graphs = OBJ_graphs

        self.labels = label_dict['labels']
        self.graphID = info_dict['graph_ids']
        self.glabel_dict = info_dict['glabel_dict']
        self.nlabel_dict = info_dict['nlabel_dict']
        self.OBJ_feats = info_dict['obj_feats']

    def has_cache(self):
        POI_graph_path = os.path.join(self.save_dir, f'{self.cityName}_POIGraph_gin.bin')
        info_path = os.path.join(self.save_dir, f'{self.cityName}_gin.pkl')
        if os.path.exists(POI_graph_path) and os.path.exists(info_path):
            return True

        return False

    @property
    def num_classes(self):
        return len(self.glabel_dict)


