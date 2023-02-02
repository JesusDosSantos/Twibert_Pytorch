import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import timeit
from tqdm import tqdm

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.get_device_name())
print(torch.__version__)
print(torch.version.cuda)

print('Loading train.json')
df_train = pd.read_json('./Twibot-20/train.json')
print('Loading test.json')
df_test = pd.read_json('./Twibot-20/test.json')
print('Loading support.json')
df_dev = pd.read_json('./Twibot-20/dev.json')
print('Finished')
df_train = df_train.iloc[:, [0, 1, 2, 3, 5]]
df_test = df_test.iloc[:, [0, 1, 2, 3, 5]]
df_dev = df_dev.iloc[:, [0, 1, 2, 3, 5]]

df_data_labeled = pd.concat([df_train, df_dev, df_test], ignore_index=True)

df_data_labeled = df_data_labeled.dropna()

df_tweets_labeled = df_data_labeled.explode('tweet')

df_tweets_labeled = df_tweets_labeled.sample(n=800000)

df_data_labeled = df_tweets_labeled.dropna()

df_tweets_labeled = df_data_labeled.explode('tweet')
df_tweets_labeled = df_tweets_labeled.reset_index()

df_tweets_labeled = df_tweets_labeled[['tweet','label']]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_ids = []
attention_masks = []
labels = []


df_tweets_labeled = df_tweets_labeled.values.tolist()
for text, label in tqdm(df_tweets_labeled):
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True
    )
    input_ids.append(encoded_text['input_ids'])
    attention_masks.append(encoded_text["attention_mask"])
    labels.append(label)


tensor_input_ids = torch.tensor(input_ids, device=device)
tensor_attention_mask = torch.tensor(attention_masks, device=device)
tensor_labels = torch.tensor(labels, device=device)

torch.save(tensor_input_ids, 'tensor_input_ids_800.pt')
torch.save(tensor_attention_mask, 'tensor_attention_mask_800.pt')
torch.save(tensor_labels, 'tensor_labels_800.pt')