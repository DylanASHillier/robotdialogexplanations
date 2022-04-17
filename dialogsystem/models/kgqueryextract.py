from torch.nn import Module, ModuleList
from torch_geometric.nn import GATv2Conv
from torch.nn import Conv1d, ELU, MSELoss
from torch import sigmoid, topk
from pytorch_lightning import LightningModule

class LightningKGQueryMPNN(LightningModule):
    def __init__(self, embedding_size, k=20, num_layers=3):
        '''
        Operates over a graph to obtain the indices of edges that should be used in the knowledge graph
        Arguments:
            num_layers,
            embedding_size,
            k: value used for top-k layer
        '''
        super(LightningKGQueryMPNN,self).__init__()
        self.final_layer = Conv1d(embedding_size,1,1)
        self.hidden_layers = ModuleList([GATv2Conv(embedding_size,embedding_size,heads=16, concat=False, dropout=0.2,) for i in range(num_layers)])
        self.activations = [ELU() for i in range(num_layers-1)]+[lambda x:x]
        self.k = k
        self.loss = MSELoss()
        self.save_hyperparameters()

    def forward(self, x, edge_index, labels):
        for layer in self.hidden_layers:
            x = layer(x,edge_index)
        x = self.final_layer(x)
        x = sigmoid(x)
        return topk(x,self.k)[1]

    def training_step(self, batch):
        x = batch.x
        y = batch.y
        edge_index = batch.edge_index
        for i,layer in enumerate(self.hidden_layers):
            x = layer(x,edge_index)
            x = self.activations[i](x)
        x = self.final_layer(x)
        y_pred = sigmoid(x)*6
        loss = self.loss(y_pred,y)
        self.log("train_mse_loss",loss.item())
        return loss

        


if __name__ == '__main__':
    model = LightningKGQueryMPNN(1000)
    