import os
import argparse
import re
import json
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='get object embeddings')
    parser.add_argument('--capPath',type=str)
    parser.add_argument('--dstPath',type=str)
    args = parser.parse_args()
    return args

def CapEmbedding(model,tokenizer,captions,device):
    for idx, caption in enumerate(captions):
        if idx == len(captions) - 1:
            caption = caption.strip()
        else:
            caption = caption.split(':')[1].strip()

        inputs = tokenizer(caption,return_tensors='pt')
        inputs = {key:value.to(device) for key,value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            cap_embeddings = last_hidden_states[0][0]
            cap_embeddings = cap_embeddings.unsqueeze(0)

        if idx == len(captions) - 1:
            image_embeddings = cap_embeddings
        elif idx == 0:
            embeddings = cap_embeddings
        else:
            embeddings = torch.cat([embeddings,cap_embeddings],dim=0)

    combined_embeddings = torch.cat([embeddings,image_embeddings],dim=0)
    return combined_embeddings



def getOBJ_Embedding(capPath,dstPath):
    objList = []
    with open('./data/objects.txt',mode='r',encoding='utf-8') as f:
        for line in f.readlines():
            objList.append(line.strip())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BERT_PATH = './ckpt/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    model = BertModel.from_pretrained(BERT_PATH)
    model.to(device)

    with open(capPath,mode='r',encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                data = json.loads(line)
                pid = data['question_id']
                caption = data['text']
                sentences = caption.strip().split('.')
                overall = sentences[-2].strip()
                objects_positions = []
                for obj in objList:
                    match = re.search(r'\b' + re.escape(obj) + r'\b', sentences[0], re.IGNORECASE)
                    if match:
                        objects_positions.append((match.start(), obj))

                objects_positions.sort()
                objects = [obj for _, obj in objects_positions]

                # no objects
                if len(sentences) -2 < len(objects):
                    input_text = [overall]
                    emb = CapEmbedding(model,tokenizer,input_text,device)

                else:
                    input_text = []
                    for obj_id, object in enumerate(objects):
                        input_text.append(f'{object}:{sentences[1+obj_id]}')
                    input_text.append(overall)
                    emb = CapEmbedding(model,tokenizer,input_text,device)

                # save the embeddings
                torch.save(emb,os.path.join(dstPath,f'{pid}.tensor'))

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue




if __name__ == '__main__':
    args = parse_args()
    getOBJ_Embedding(
        capPath=args.capPath,
        dstPath=args.dstPath,
    )





















