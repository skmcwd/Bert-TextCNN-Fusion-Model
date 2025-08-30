import torch
from transformers import BertModel, BertTokenizer, AdamW
from datasets import load_dataset

# 中文分类的cuda版本

# 快速演示
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device=', device)

# 加载预训练模型
pretrained = BertModel.from_pretrained('model/bert-base-chinese/')
# 需要移动到cuda上
pretrained.to(device)

# 不训练,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)


# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out


model = Model()
# 同样要移动到cuda
model.to(device)


# 后面的计算和中文分类完全一样，只是放在了cuda上计算
# 定义数据集
class Dataset(torch.utils.data.Dataset):

    def __init__(self, split):
        self.dataset = load_dataset('datasets')[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']
        return text, label


dataset = Dataset('train')

# 加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-chinese')


def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    # 编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   return_length=True)
    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    labels = torch.LongTensor(labels).to(device)
    # print(data['length'], data['length'].max())
    return input_ids, attention_mask, token_type_ids, labels


# 数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=16,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)
for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    break

# print(len(loader))
# print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)

# 训练
optimizer = AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()
model.train()
for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    out = model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % 5 == 0:
        out = out.argmax(dim=1)
        accuracy = (out == labels).sum().item() / len(labels)
        print(i, loss.item(), accuracy)
    if i == 300:
        break


# 测试
def test():
    model.eval()
    correct = 0
    total = 0
    loader_test = torch.utils.data.DataLoader(dataset=Dataset('validation'),
                                              batch_size=32,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader_test):
        if i == 5:
            break
        print(i)
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)
    print(correct / total)


test()
