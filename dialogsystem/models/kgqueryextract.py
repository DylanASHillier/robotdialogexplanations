from torch.nn import Module, ModuleList
from torch_geometric.nn import GATv2Conv
from torch.nn import Conv1d, ELU, MSELoss, Linear
from torch import sigmoid, topk
from pytorch_lightning import LightningModule
from torch.optim import Adam

class LightningKGQueryMPNN(LightningModule):
    def __init__(self, embedding_size, hidden_dim=50, k=20, num_layers=3, lr=1e-3, heads=4):
        '''
        Operates over a graph to obtain the indices of edges that should be used in the knowledge graph
        Arguments:
            num_layers,
            embedding_size,
            k: value used for top-k layer
        '''
        super(LightningKGQueryMPNN,self).__init__()
        self.final_layer = Linear(hidden_dim,1)
        self.init_layer = Linear(embedding_size,hidden_dim)
        self.hidden_layers = ModuleList([GATv2Conv(hidden_dim,hidden_dim,heads=heads, concat=False, dropout=0.2,) for i in range(num_layers)])
        self.activations = [ELU() for i in range(num_layers-1)]+[lambda x:x]
        self.k = k
        self.lr = lr
        self.loss = MSELoss()
        self.save_hyperparameters()

    def forward(self, x, edge_index):
        x = self.init_layer(x)
        for i,layer in enumerate(self.hidden_layers):
            x = layer(x,edge_index)
            x = self.activations[i](x)
        x = self.final_layer(x)
        x = sigmoid(x).squeeze()
        return topk(x,min(self.k,x.size(0)))[1]

    def training_step(self, batch):
        x = batch.x
        y = batch.y
        edge_index = batch.edge_index
        x = self.init_layer(x)
        for i,layer in enumerate(self.hidden_layers):
            x = layer(x,edge_index)
            x = self.activations[i](x)
        x = self.final_layer(x).squeeze()
        y_pred = sigmoid(x)*6
        loss = self.loss(y_pred,y)
        self.log("train_mse_loss",loss.item())
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)

        


if __name__ == '__main__':
    model = LightningKGQueryMPNN(1000)
    