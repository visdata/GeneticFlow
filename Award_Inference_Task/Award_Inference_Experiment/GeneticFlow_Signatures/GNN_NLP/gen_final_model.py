import argparse
import os
import time
from dataset import MultiSessionsGraph
import torch
import torch.nn.functional as F
from models import Model,Net,Net2,Net3
from torch_geometric.loader import DataLoader
import numpy as np
from torch.autograd import Variable


Name=[[] for i in range(1201)] #index by rank, pred_class as value

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=50, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
parser.add_argument('--conv_name', type=str, default='ARMAConv', help='conv')
parser.add_argument('--pool_name', type=str, default='EdgePooling', help='pooling')
parser.add_argument('--PATH', type=str, default='../GNN_NLP_data/data_origin', help='dataset')
parser.add_argument('--ignore_edge', type=bool, default=False, help='whether ignore edge')
parser.add_argument('--train_all', type=bool, default=False, help='train model use all available field')
parser.add_argument('--ignore_node_attr', type=bool, default=False, help='whether ignore node attribute')
parser.add_argument('--datalen', type=int, default=-1, help='limit data length')
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=64868)

def print_to_file(filename, string_info, mode="a"):
	with open(filename, mode) as f:
		f.write(str(string_info) + "\n")

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    args.device='cpu'

dataset = MultiSessionsGraph('./data',PATH=args.PATH,train_all=args.train_all,datalen=args.datalen)
dataset = dataset.shuffle()
dataset = dataset.shuffle()
dataset = dataset.shuffle()

args.num_classes = 2

if args.ignore_node_attr:
    args.num_features=1
else:
    args.num_features = dataset.num_features

print(args)

def train(training_set):
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    model = Model(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            if args.ignore_edge:
                tmp = [[] for i in range(data.edge_index.shape[0])]
                data.edge_index=torch.tensor(tmp,dtype=torch.long)
                data.edge_attr=torch.tensor([[]],dtype=torch.long).t()
            if args.ignore_node_attr:
                data.x=torch.ones(data.x.shape[0],1,dtype=data.x.dtype)
            optimizer.zero_grad()
            y=[]
            temp = torch.reshape(data.y,(-1,1))
            for i in range(temp.shape[0]):
                y.append(int(temp[i][0]))
            y=Variable(torch.LongTensor(y)).to(args.device)

            #global_vector
            paper_count=[0]*(data.batch[data.batch.shape[0]-1]+1) #[0]*3->[0, 0, 0], data.batch give concat of all nodes in 180 selected_subgraph
            for j in range(data.batch.shape[0]):
                paper_count[data.batch[j]]+=1
            paper_count=Variable(torch.FloatTensor(paper_count)).to(args.device)

            data = data.to(args.device)
            out = model(data,paper_count)
            weights = np.array([1,1])
            weights = torch.FloatTensor(weights).to(args.device)
            loss = F.nll_loss(out, y, weight=weights)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(y.to(args.device)).sum().item()
        acc_train = correct / len(train_loader.dataset)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),'acc_train: {:.6f}'.format(acc_train))
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))
    return model.cpu()


if __name__ == '__main__':
    # Model training and save
    dataset = dataset.shuffle()
    for item in dataset:
        print(item.name)
    final_model=train(dataset)
    save_path=os.path.join('./save_model',args.conv_name+'.pt')
    torch.save(final_model, save_path)
