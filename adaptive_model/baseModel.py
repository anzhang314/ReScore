
from ast import arg
from cmath import inf
from notears.locally_connected import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
from plot_utils import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from notears.loss_func import *
from plot_utils import *
import notears.utils as ut
import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from scipy.linalg import expm
from scipy.special import comb
import math


# def record_weight(reweight_list, cnt, hard_list=[26,558,550,326,915], easy_list=[859,132,82,80,189]):
#     writer = SummaryWriter('logs/weight_record_real')
#     reweight_idx = reweight_list.squeeze()
#     reweight_idx = reweight_idx.tolist()
#     for idx in hard_list:
#         writer.add_scalar(f'hard_real/hard_reweight_list[{idx}]', reweight_idx[idx], cnt)
#     for idx in easy_list:
#         writer.add_scalar(f'easy_real/easy_reweight_list[{idx}]', reweight_idx[idx], cnt) 



class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2    
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)  
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers) 
        

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1] 
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d] 
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i] 
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        # h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        M = torch.eye(d).to(A.device) + A / d  # (Yu et al. 2019)
        E = torch.matrix_power(M, d - 1)
        h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg 

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg
    
    def predict(self,x):
        return self.forward(x)

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W



class GOLEM(nn.Module):
    """Set up the objective function of GOLEM.
    Hyperparameters:
        (1) GOLEM-NV: lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: lambda_1=2e-2, lambda_2=5.0.(not used)
    """

    def __init__(self, args):
        super(GOLEM, self).__init__()
        self.n = args.n
        self.d = args.d
        self.lambda_1 = args.lambda1
        self.lambda_2 = args.lambda2
        self.W=nn.Linear(args.d, args.d, bias=False)
        self.lr=args.golem_lr
        nn.init.zeros_(self.W.weight)

        #nn.init.xavier_normal_(self.W.weight)

        # with torch.no_grad():
        #     #self.W.weight=torch.triu(self.W.weight)
        #     idx=torch.triu_indices(*self.W.weight.shape)
        #     self.W.weight[idx[0],idx[1]]=0
            
    
    def predict(self,X):
        return self.W(X)
    

    def forward(self, X, weight):

        likelihood = self._compute_likelihood(X,weight)
        L1_penalty = self._compute_L1_penalty()
        h = self._compute_h()
        loss= likelihood + self.lambda_1 * L1_penalty + self.lambda_2 * h
        return loss, likelihood, self.lambda_1 * L1_penalty, self.lambda_2 * h

    def _compute_likelihood(self,X,weight):
        """Compute (negative log) likelihood in the linear Gaussian case.
        Returns:
            tf.Tensor: Likelihood term (scalar-valued).
        """
        return 0.5 * self.d * torch.log(
                torch.sum(torch.mul(weight,torch.sum(torch.square(X-self.W(X)),dim=1)))
                # torch.square(
                #     torch.linalg.norm(X - self.W(X))
                # )
            ) - torch.linalg.slogdet(torch.eye(self.d) - self.W.weight.T)[1]
        # return 0.5 * torch.sum(
        #     torch.log(
        #         torch.sum(
        #             torch.square(X - self.W(X)), axis=0
        #         )
        #     )
        # ) - torch.linalg.slogdet(torch.eye(self.d) - self.W.weight.T)[1]

    def _compute_L1_penalty(self):
        """Compute L1 penalty.
        Returns:
            tf.Tensor: L1 penalty term (scalar-valued).
        """
        return torch.norm(self.W.weight, 1)

    def _compute_h(self):
        """Compute DAG penalty.
        Returns:
            tf.Tensor: DAG penalty term (scalar-valued).
        """
        return torch.trace(torch.matrix_exp(self.W.weight.T * self.W.weight.T)) - self.d
    
    @torch.no_grad()
    def W_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        w = self.W.weight.T.cpu().detach().numpy()  # [i, j]
        return w



class DAGGNN_MLPEncoder(nn.Module):
    """MLP encoder module."""
    def __init__(self, n_in, n_xdims, n_hid, n_out, batch_size, do_prob=0., factor=True, tol = 0.1):
        super(DAGGNN_MLPEncoder, self).__init__()

        adj_A = np.zeros((n_in, n_in))
        self.adj_A = nn.Parameter(torch.autograd.Variable(torch.from_numpy(adj_A).float(), requires_grad=True))
        self.factor = factor

        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias = True)
        self.fc2 = nn.Linear(n_hid, n_out, bias = True)
        self.dropout_prob = do_prob
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).float())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs):
        def preprocess_adj_new(adj):
            adj_normalized = (torch.eye(adj.shape[0]) - (adj.transpose(0,1)))
            return adj_normalized

        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)

        # adj_Aforz = I-A^T
        adj_Aforz = preprocess_adj_new(adj_A1) #[d*d]

        H1 = F.relu((self.fc1(inputs)))#[?,d,m(=1)]=>[?,d,hidden]
        x = (self.fc2(H1)) #[?,d,hidden]=>[?,d,n_out]
        logits = torch.matmul(adj_Aforz, x+self.Wa) -self.Wa 


        return logits, adj_A1, self.Wa
    
class DAGGNN_MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_z, n_out, data_variable_size, batch_size, n_hid,
                 do_prob=0.):
        super(DAGGNN_MLPDecoder, self).__init__()

        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias = True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias = True)

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_z, origin_A, Wa):

        def preprocess_adj_new1(adj):
            adj_normalized = torch.inverse(torch.eye(adj.shape[0])-adj.transpose(0,1))
            return adj_normalized
        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(origin_A)
        #print(origin_A.shape)
        #print(input_z.shape)
        #print(Wa.shape)
        mat_z = torch.matmul(adj_A_new1, input_z+Wa) - Wa
        H3 = F.relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H3)

        return out

class DAGGNN(nn.Module):
    """MLP decoder module."""
    def __init__(self, encoder, decoder):
        super(DAGGNN, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.best_ELBO_graph = torch.sinh(3.*self.encoder.adj_A).data.clone().numpy()
        self.best_MSE_graph = torch.sinh(3.*self.encoder.adj_A).data.clone().numpy()
        self.best_NLL_graph = torch.sinh(3.*self.encoder.adj_A).data.clone().numpy()
    def forward(self, X):
        X=torch.unsqueeze(X,2)
        logits, adj_A1, Wa = self.encoder(X)
        out = self.decoder(logits,adj_A1,Wa)
        return torch.squeeze(out)
    
    def predict(self, X):
        return self.forward(X)
    
    def get_adj(self):
        return self.best_NLL_graph



class TrExpScipy(torch.autograd.Function):
    """
    autograd.Function to compute trace of an exponential of a matrix
    """

    @staticmethod
    def forward(ctx, input):
        device=input.device
        with torch.no_grad():
            # send tensor to cpu in numpy format and compute expm using scipy
            expm_input = expm(input.detach().cpu().numpy())
            # transform back into a tensor
            expm_input = torch.as_tensor(expm_input)
            if input.is_cuda:
                expm_input = expm_input.to(device)
                assert expm_input.is_cuda
            # save expm_input to use in backward
            ctx.save_for_backward(expm_input)

            # return the trace
            return torch.trace(expm_input)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            expm_input, = ctx.saved_tensors
            return expm_input.t() * grad_output

def compute_constraint(model, w_adj):
        assert (w_adj >= 0).detach().cpu().numpy().all()
        h = TrExpScipy.apply(w_adj) - model.num_vars
        return h

def compute_A_phi(model, norm="none", square=False):
        weights = model.get_parameters(mode='w')[0]
        prod = torch.eye(model.num_vars).to(model.device)
        if norm != "none":
            prod_norm = torch.eye(model.num_vars).to(model.device)
        for i, w in enumerate(weights):
            if square:
                w = w ** 2
            else:
                w = torch.abs(w)
            if i == 0:
                prod = torch.einsum("tij,ljt,jk->tik", w, model.adjacency.unsqueeze(0), prod)
                if norm != "none":
                    tmp = 1. - torch.eye(model.num_vars).unsqueeze(0).to(model.device)
                    prod_norm = torch.einsum("tij,ljt,jk->tik", torch.ones_like(w).detach(), tmp, prod_norm)
            else:
                prod = torch.einsum("tij,tjk->tik", w, prod)
                if norm != "none":
                    prod_norm = torch.einsum("tij,tjk->tik", torch.ones_like(w).detach(), prod_norm)

        # sum over density parameter axis
        prod = torch.sum(prod, 1)
        if norm == "paths":
            prod_norm = torch.sum(prod_norm, 1).to(model.device)
            denominator = prod_norm + torch.eye(model.num_vars).to(model.device)  # avoid / 0 on diagonal
            return (prod / denominator).t()
        elif norm == "none":
            return prod.t()
        else:
            raise NotImplementedError

class BaseModel(nn.Module):
    def __init__(self, num_vars, num_layers, hid_dim, num_params, nonlin="leaky-relu", norm_prod='path',
                 square_prod=False,device='cpu'):
        """

        :param num_vars: number of variables in the system
        :param num_layers: number of hidden layers
        :param hid_dim: number of hidden units per layer
        :param num_params: number of parameters per conditional *outputted by MLP*
        :param nonlin: which nonlinearity
        """
        super(BaseModel, self).__init__()
        self.num_vars = num_vars
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.num_params = num_params
        self.nonlin = nonlin
        self.norm_prod = norm_prod
        self.square_prod = square_prod
        self.device = device

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.extra_params = []  # Those parameter might be learnable, but they do not depend on parents.

        # initialize current adjacency matrix
        self.adjacency = nn.Parameter(torch.ones((self.num_vars, self.num_vars)) - torch.eye(self.num_vars), requires_grad=False)

        #self.adjacency=self.adjacency.to(self.device)

        self.zero_weights_ratio = 0.
        self.numel_weights = 0

        # Instantiate the parameters of each layer in the model of each variable
        for i in range(self.num_layers + 1):
            in_dim = self.hid_dim
            out_dim = self.hid_dim
            if i == 0:
                in_dim = self.num_vars
            if i == self.num_layers:
                out_dim = self.num_params
            self.weights.append(nn.Parameter(torch.zeros(self.num_vars, out_dim, in_dim)))
            self.biases.append(nn.Parameter(torch.zeros(self.num_vars, out_dim)))
            self.numel_weights += self.num_vars * out_dim * in_dim

    def forward_given_params(self, x, weights, biases):
        """

        :param x: batch_size x num_vars
        :param weights: list of lists. ith list contains weights for ith MLP
        :param biases: list of lists. ith list contains biases for ith MLP
        :return: batch_size x num_vars * num_params, the parameters of each variable conditional
        """
        bs = x.size(0)
        num_zero_weights = 0
        for k in range(self.num_layers + 1):
            # apply affine operator
            if k == 0:
                adj = self.adjacency.unsqueeze(0).to(self.device)
                x = torch.einsum("tij,ljt,bj->bti", weights[k], adj, x) + biases[k]
            else:
                x = torch.einsum("tij,btj->bti", weights[k], x) + biases[k]

            # count num of zeros
            num_zero_weights += weights[k].numel() - weights[k].nonzero().size(0)

            # apply non-linearity
            if k != self.num_layers:
                x = F.leaky_relu(x) if self.nonlin == "leaky-relu" else torch.sigmoid(x)

        self.zero_weights_ratio = num_zero_weights / float(self.numel_weights)

        return torch.unbind(x, 1)

    
    
    
    def get_w_adj(self):
        """Get weighted adjacency matrix"""
        return compute_A_phi(self, norm=self.norm_prod, square=self.square_prod)

    def reset_params(self):
        with torch.no_grad():
            for node in range(self.num_vars):
                for i, w in enumerate(self.weights):
                    w = w[node]
                    nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('leaky_relu'))
                for i, b in enumerate(self.biases):
                    b = b[node]
                    b.zero_()

    def get_parameters(self, mode="wbx"):
        """
        Will get only parameters with requires_grad == True
        :param mode: w=weights, b=biases, x=extra_params (order is irrelevant)
        :return: corresponding dicts of parameters
        """
        params = []

        if 'w' in mode:
            weights = []
            for w in self.weights:
                weights.append(w)
            params.append(weights)
        if 'b'in mode:
            biases = []
            for j, b in enumerate(self.biases):
                biases.append(b)
            params.append(biases)

        if 'x' in mode:
            extra_params = []
            for ep in self.extra_params:
                if ep.requires_grad:
                    extra_params.append(ep)
            params.append(extra_params)

        return tuple(params)

    def set_parameters(self, params, mode="wbx"):
        """
        Will set only parameters with requires_grad == True
        :param params: tuple of parameter lists to set, the order should be coherent with `get_parameters`
        :param mode: w=weights, b=biases, x=extra_params (order is irrelevant)
        :return: None
        """
        with torch.no_grad():
            k = 0
            if 'w' in mode:
                for i, w in enumerate(self.weights):
                    w.copy_(params[k][i])
                k += 1

            if 'b' in mode:
                for i, b in enumerate(self.biases):
                    b.copy_(params[k][i])
                k += 1

            if 'x' in mode and len(self.extra_params) > 0:
                for i, ep in enumerate(self.extra_params):
                    if ep.requires_grad:
                        ep.copy_(params[k][i])
                k += 1

    def get_grad_norm(self, mode="wbx"):
        """
        Will get only parameters with requires_grad == True, simply get the .grad
        :param mode: w=weights, b=biases, x=extra_params (order is irrelevant)
        :return: corresponding dicts of parameters
        """
        grad_norm = 0

        if 'w' in mode:
            for w in self.weights:
                grad_norm += torch.sum(w.grad ** 2)

        if 'b'in mode:
            for j, b in enumerate(self.biases):
                grad_norm += torch.sum(b.grad ** 2)

        if 'x' in mode:
            for ep in self.extra_params:
                if ep.requires_grad:
                    grad_norm += torch.sum(ep.grad ** 2)

        return torch.sqrt(grad_norm)

    def save_parameters(self, exp_path, mode="wbx"):
        params = self.get_parameters(mode=mode)
        # save
        with open(os.path.join(exp_path, "params_"+mode), 'wb') as f:
            pickle.dump(params, f)

    def load_parameters(self, exp_path, mode="wbx"):
        with open(os.path.join(exp_path, "params_"+mode), 'rb') as f:
            params = pickle.load(f)
        self.set_parameters(params, mode=mode)

    def get_distribution(self, density_params):
        raise NotImplementedError

class LearnableModel(BaseModel):
    def __init__(self, num_vars, num_layers, hid_dim, num_params, nonlin="leaky-relu", norm_prod='path',
                 square_prod=False,device='cpu'):
        super(LearnableModel, self).__init__(num_vars, num_layers, hid_dim, num_params, nonlin=nonlin,
                                             norm_prod=norm_prod, square_prod=square_prod,device=device)
        self.reset_params()

    def compute_log_likelihood(self, x, weights, biases, extra_params, detach=False):
        """
        Return log-likelihood of the model for each example.
        WARNING: This is really a joint distribution only if the DAGness constraint on the mask is satisfied.
                 Otherwise the joint does not integrate to one.
        :param x: (batch_size, num_vars)
        :param weights: list of tensor that are coherent with self.weights
        :param biases: list of tensor that are coherent with self.biases
        :return: (batch_size, num_vars) log-likelihoods
        """
        density_params = self.forward_given_params(x, weights, biases)

        if len(extra_params) != 0:
            extra_params = self.transform_extra_params(self.extra_params)
        log_probs = []
        for i in range(self.num_vars):
            density_param = list(torch.unbind(density_params[i], 1))
            if len(extra_params) != 0:
                density_param.extend(list(torch.unbind(extra_params[i], 0)))
            conditional = self.get_distribution(density_param)
            x_d = x[:, i].detach() if detach else x[:, i]
            log_probs.append(conditional.log_prob(x_d).unsqueeze(1))

        return torch.cat(log_probs, 1)
    
    def compute_weighted_log_likelihood(self, x, weights, biases, extra_params, sample_weight, detach=False):
        """
        Return log-likelihood of the model for each example.
        WARNING: This is really a joint distribution only if the DAGness constraint on the mask is satisfied.
                 Otherwise the joint does not integrate to one.
        :param x: (batch_size, num_vars)
        :param weights: list of tensor that are coherent with self.weights
        :param biases: list of tensor that are coherent with self.biases
        :return: (batch_size, num_vars) log-likelihoods
        """
        log_probs=self.compute_log_likelihood(x, weights, biases, extra_params, detach)

        return 

    def get_distribution(self, dp):
        raise NotImplementedError

    def transform_extra_params(self, extra_params):
        raise NotImplementedError

class LearnableModel_NonLinGauss(LearnableModel):
    def __init__(self, num_vars, num_layers, hid_dim, nonlin="leaky-relu", norm_prod='path',
                 square_prod=False,device='cpu'):
        super(LearnableModel_NonLinGauss, self).__init__(num_vars, num_layers, hid_dim, 2, nonlin=nonlin,
                                                         norm_prod=norm_prod, square_prod=square_prod,device=device)

    def get_distribution(self, dp):
        return torch.distributions.normal.Normal(dp[0], torch.exp(dp[1]))

class LearnableModel_NonLinGaussANM(LearnableModel):
    def __init__(self, num_vars, num_layers, hid_dim, nonlin="leaky-relu", norm_prod='path',
                 square_prod=False,device='cpu'):
        super(LearnableModel_NonLinGaussANM, self).__init__(num_vars, num_layers, hid_dim, 1, nonlin=nonlin,
                                                            norm_prod=norm_prod, square_prod=square_prod,device=device)
        # extra parameters are log_std
        extra_params = np.ones((self.num_vars,))
        np.random.shuffle(extra_params)  # TODO: make sure this init does not bias toward gt model
        # each element in the list represents a variable, the size of the element is the number of extra_params per var
        self.extra_params = nn.ParameterList()
        for extra_param in extra_params:
            self.extra_params.append(nn.Parameter(torch.tensor(np.log(extra_param).reshape(1)).type(torch.Tensor)))

    def get_distribution(self, dp):
        return torch.distributions.normal.Normal(dp[0], dp[1])

    def transform_extra_params(self, extra_params):
        transformed_extra_params = []
        for extra_param in extra_params:
            transformed_extra_params.append(torch.exp(extra_param))
        return transformed_extra_params  # returns std_dev


def dual_ascent_step_golem(args, model, X, train_loader, adp_flag, adaptive_model):
    X = X - X.mean(axis=0, keepdims=True)
    X = X.to(args.device)
    #print(X)
    patience=args.golem_patience
    cur_patience=0
    last_loss=inf
    epoch=0
    while cur_patience<patience:
        optimizer = torch.optim.Adam([ param for param in model.parameters() if param.requires_grad == True], lr=model.lr)

        primal_obj = torch.tensor(0.).to(args.device)
        tot_loss = torch.tensor(0.).to(args.device)
        tot_likelihood = torch.tensor(0.).to(args.device)
        tot_L1 = torch.tensor(0.).to(args.device)
        tot_h = torch.tensor(0.).to(args.device)

        for _ , tmp_x in enumerate(train_loader):
            batch_x = tmp_x[0].to(args.device)
            batch_x = batch_x - torch.mean(batch_x)
            
            X_hat = model.predict(batch_x)
                
                # TODO: the adaptive loss should add here
            if adp_flag == False or args.run_mode == False:
                reweight_list = torch.ones(batch_x.shape[0],1)/batch_x.shape[0]
                reweight_list = reweight_list.to(args.device)
            else:
                with torch.no_grad():
                    model.eval()
                    reweight_list = adaptive_model((batch_x-X_hat)**2)
            
                model.train()
                # print(reweight_list.squeeze(1))
                # print(reweight_list)
                # print(model.W.weight)
                # input()
            loss, likelihood, L1_penalty, h = model(batch_x,reweight_list)#adaptive_loss(X_hat, batch_x, reweight_list)
            #print(loss)
           
                

            tot_loss+=loss
            tot_likelihood+=likelihood
            tot_L1+=L1_penalty
            tot_h+=h
        
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()

        if tot_loss.detach().item() < last_loss:
            last_loss= tot_loss.detach().item()
            cur_patience=0
        else:
            cur_patience+=1
        #print(model.W.weight)
        h_cur = model._compute_h().detach().item()
        perf_str='Epoch %d : training loss ==[%.5f = %.5f + %.5f +  %.5f], curr H: %.5f, curr patience: %d' % (
                epoch, tot_loss.detach().item(),tot_likelihood.detach().item(), 
                tot_L1.detach().item(), tot_h.detach().item(), h_cur,cur_patience)
        epoch+=1
        #print(perf_str)
       
    
    return h

def dual_ascent_step(args, model, X, train_loader, lambda1, lambda2, rho, alpha, h, rho_max, adp_flag, adaptive_model):
    """Perform one step of dual ascent in augmented Lagrangian."""
    def adaptive_loss(output, target, reweight_list):
        R = output-target
        # reweight_matrix = torch.diag(reweight_idx).to(args.device)
        # loss = 0.5 * torch.sum(torch.matmul(reweight_matrix, R))
        loss = 0.5 * torch.sum(torch.mul(reweight_list, R**2))
        return loss
    
    def closure():
        X.to(args.device)
        model.to(args.device)
        optimizer.zero_grad()
        #print([param.device for param in model.parameters()])
        X_hat = model(X)
        loss = squared_loss(X_hat, X)
        h_val = model.h_func()
        penalty = 0.5 * rho * h_val * h_val + alpha * h_val
        l2_reg = 0.5 * lambda2 * model.l2_reg()
        l1_reg = lambda1 * model.fc1_l1_reg()
        primal_obj = loss + penalty + l2_reg + l1_reg
        primal_obj.backward()
        # if COUNT % 100 == 0:
        #     print(f"{primal_obj}: {primal_obj.item():.4f}; count: {COUNT}")
        return primal_obj

    def r_closure():
        optimizer.zero_grad()

        primal_obj = torch.tensor(0.).to(args.device)
        loss = torch.tensor(0.).to(args.device)

        for _ , tmp_x in enumerate(train_loader):
            batch_x = tmp_x[0].to(args.device)
            X_hat = model(batch_x)
            
            # TODO: the adaptive loss should add here
            if adp_flag == False:
                reweight_list = torch.ones(batch_x.shape[0],1)/batch_x.shape[0]
                reweight_list = reweight_list.to(args.device)
            else:
                with torch.no_grad():
                    model.eval()
                    reweight_list = adaptive_model((batch_x-X_hat)**2)
                
                model.train()
            # print(reweight_list.squeeze(1))
            primal_obj += adaptive_loss(X_hat, batch_x, reweight_list)
        
        h_val = model.h_func()
        penalty = 0.5 * rho * h_val * h_val + alpha * h_val
        l2_reg = 0.5 * lambda2 * model.l2_reg()
        l1_reg = lambda1 * model.fc1_l1_reg()
        primal_obj += penalty + l2_reg + l1_reg
        primal_obj.backward()
        # if COUNT % 100 == 0:
        #     print(f"{primal_obj}: {primal_obj.item():.4f}; count: {COUNT}")
        return primal_obj


    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    # X_torch = torch.from_numpy(X)
    while rho < rho_max:
        #for i in range(5):
        if args.run_mode:
            optimizer.step(closure)  # NOTE: updates model in-place
        else:                        # NOTE: the adaptive reweight operation
            optimizer.step(r_closure)
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new

def dual_ascent_step_daggnn(args, model, X, train_loader, rho, alpha, h, rho_max, adp_flag, adaptive_model,true_graph):
    def _h_A(A, m):
        def matrix_poly(matrix, d):
            x = torch.eye(d).double()+ torch.div(matrix, d)
            return torch.matrix_power(x, d)
        expm_A = matrix_poly(A*A, m)
        h_A = torch.trace(expm_A) - m
        return h_A

    def update_optimizer(optimizer, original_lr, c_A):
        '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
        MAX_LR = 1e-2
        MIN_LR = 1e-4

        estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
        if estimated_lr > MAX_LR:
            lr = MAX_LR
        elif estimated_lr < MIN_LR:
            lr = MIN_LR
        else:
            lr = estimated_lr

        # set LR
        for parame_group in optimizer.param_groups:
            parame_group['lr'] = lr

        return optimizer, lr
    
    def adaptive_nll_gaussian(preds, target, variance, reweight_list):
        neg_log_p = variance + torch.div(torch.pow(preds - target, 2), 2.*np.exp(2. * variance))
        return torch.sum(torch.mul(reweight_list,neg_log_p)) / (target.size(0))
    
    def nll_gaussian(preds, target, variance, add_const=False):
        mean1 = preds
        mean2 = target
        neg_log_p = variance + torch.div(torch.pow(mean1 - mean2, 2), 2.*np.exp(2. * variance))
        # if add_const:
        #     const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        #     neg_log_p += const
        return neg_log_p.sum() / (target.size(0))
    
    def kl_gaussian_sem(preds):
        mu = preds
        kl_div = mu * mu
        kl_sum = kl_div.sum()
        return (kl_sum / (preds.size(0)))*0.5

        
    def train(epoch, lambda_A, c_A, optimizer):
        # update optimizer
        optimizer, lr = update_optimizer(optimizer, args.daggnn_lr, c_A)

        nll_train = []
        kl_train = []
        mse_train = []
        shd_trian = []

        model.train()
        #scheduler.step()
            

        for _ , tmp_x in enumerate(train_loader):
            batch_x = tmp_x[0].to(args.device)

            optimizer.zero_grad()

            batch_x=torch.unsqueeze(batch_x,dim=2)

            logits, origin_A, Wa = model.encoder(batch_x)  # logits is of size: [num_sims, z_dims]
            edges = logits

            output = model.decoder(edges, origin_A, Wa)

            if torch.sum(output != output):
                print('nan error\n')

            target = batch_x.squeeze()
            preds = output.squeeze()
            variance = 0.

            if adp_flag == False or args.run_mode == 1:
                reweight_list = torch.ones(batch_x.shape[0],1)/batch_x.shape[0]
                reweight_list = reweight_list.to(args.device)
            else:
                with torch.no_grad():
                    model.eval()
                    reweight_list = adaptive_model((target-preds)**2)

            # reconstruction accuracy loss
            #loss_nll = adaptive_nll_gaussian(preds, target, variance, reweight_list)
            loss_nll = nll_gaussian(output, batch_x, variance)

            # KL loss
            loss_kl = kl_gaussian_sem(logits)

            # ELBO loss:
            loss = loss_kl + loss_nll

            # add A loss
            one_adj_A = origin_A # torch.mean(adj_A_tilt_decoder, dim =0)
            sparse_loss = args.lambda1 * torch.sum(torch.abs(one_adj_A))

            # compute h(A)
            h_A = _h_A(origin_A, args.d)
            loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A*origin_A) + sparse_loss #+  0.01 * torch.sum(variance * variance)


            loss.backward()
            loss = optimizer.step()

            #myA.data = stau(myA.data, args.tau_A*lr)

            if torch.sum(origin_A != origin_A):
                print('nan error\n')

            # compute metrics
            graph = origin_A.data.clone().numpy()

            mse_train.append(F.mse_loss(preds, target).item())
            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())

        # my_graph=graph
        # my_graph[np.abs(my_graph) < 0.3]=0
        #print(graph)
        print(h_A.item())
        print('Epoch: {:04d}'.format(epoch),
            'nll_train: {:.10f}'.format(np.mean(nll_train)),
            'kl_train: {:.10f}'.format(np.mean(kl_train)),
            'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
            'mse_train: {:.10f}'.format(np.mean(mse_train)))
        
        return np.mean(np.mean(kl_train)  + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A
    
    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    best_epoch = 0
    best_ELBO_graph = []
    best_NLL_graph = []
    best_MSE_graph = []
    # optimizer step on hyparameters
    c_A = rho
    lambda_A = alpha

    optimizer = torch.optim.Adam(model.parameters(),lr=args.daggnn_lr)
    #epoch=0
    while c_A < rho_max:
        for epoch in range(args.daggnn_epochs):
            ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = train(epoch, lambda_A, c_A, optimizer)
            if ELBO_loss < best_ELBO_loss:
                best_ELBO_loss = ELBO_loss
                best_epoch = epoch
                model.best_ELBO_graph=graph

            if NLL_loss < best_NLL_loss:
                best_NLL_loss = NLL_loss
                best_epoch = epoch
                model.best_NLL_graph = graph

            if MSE_loss < best_MSE_loss:
                best_MSE_loss = MSE_loss
                best_epoch = epoch
                model.best_MSE_graph = graph
            
            #print(graph)
            #graph[np.abs(graph) < 0.3] = 0
            
            #print(ut.count_accuracy(true_graph, graph != 0))

        print("Optimization Finished!")
        print("Best Epoch: {:04d}".format(best_epoch))

        A_new = origin_A.data.clone()
        h_A_new = _h_A(A_new, args.d)
        #print("Epoch: {:04d}, ELBO: {:.10f}, NLL:{:.10f}, MSE:{:.10f}".format(best_epoch,ELBO_loss,NLL_loss,MSE_loss))
        if ELBO_loss > 2 * best_ELBO_loss:
            break
        #epoch+=1
        # update parameters
        
        if h_A_new.item() > 0.25 * h:
            c_A*=10
        else:
            break
    lambda_A += c_A * h_A_new.item()
    # print(graph)          break
    lambda_A += c_A * h_A_new.item()
    # print(graph)
    # graph[np.abs(graph) < 0.3] = 0
    # print(ut.count_accuracy(true_graph, graph != 0))

    return c_A, lambda_A, h_A_new
    
def dual_ascent_step_grandag(args, model, X, train_loader, rho, alpha, h, rho_max, adp_flag, adaptive_model,true_graph, _mus,_lambdas,_w_adjs,_iter_cnt):
    if args.gran_optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.gran_lr)
    elif args.gran_optim == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.gran_lr)
    else:
        raise NotImplementedError("optimizer {} is not implemented".format(args.gran_optim))

    
    #print([param.device for param in model.parameters()])
    aug_lagrangians = []
    aug_lagrangian_ma = [] 
    aug_lagrangians_val = []
    grad_norms = []
    grad_norm_ma = []

    not_nlls = []  # Augmented Lagrangrian minus (pseudo) NLL
    nlls = []  # NLL on train
    mus = _mus
    lambdas = _lambdas
    w_adjs = _w_adjs
    mu=rho
    lamb=alpha

    cur_h=h

    iter_cnt=_iter_cnt
    cur_min=inf
    cur_patience=0
    while mu < rho_max:
        for _ , tmp_x in enumerate(train_loader):
            batch_x = tmp_x[0].to(args.device)
            model.train()
            weights, biases, extra_params = model.get_parameters(mode="wbx")
            log_likelihood=model.compute_log_likelihood(batch_x, weights, biases, extra_params)
            if adp_flag == False or args.run_mode == 1:
                reweight_list = torch.ones(batch_x.shape[0],1)#/batch_x.shape[0]
                reweight_list = reweight_list.to(args.device)
            else:
                with torch.no_grad():
                    model.eval()
                    reweight_list = adaptive_model(-log_likelihood)
            loss = - torch.mean(torch.mul(reweight_list,log_likelihood))
            nlls.append(loss.item())
            w_adj = model.get_w_adj()
            cur_h = compute_constraint(model, w_adj)
            aug_lagrangian = loss + 0.5 * mu * cur_h ** 2 + lamb * cur_h
            optimizer.zero_grad()
            aug_lagrangian.backward()
            optimizer.step()

            if args.edge_clamp_range != 0:
                with torch.no_grad():
                    to_keep = (w_adj > args.edge_clamp_range).type(torch.Tensor).to(model.device)
                    model.adjacency *= to_keep
            
            if not args.no_w_adjs_log:
                w_adjs.append(w_adj.detach().cpu().numpy().astype(np.float32))
            mus.append(mu)
            lambdas.append(lamb)
            not_nlls.append(0.5 * mu * cur_h.item() ** 2 + lamb * cur_h.item())
            
            
            if iter_cnt % args.plot_freq == 0:
                if not args.no_w_adjs_log:
                    plot_weighted_adjacency(w_adjs, true_graph, args.graph_path,
                                            name="w_adj", mus=mus, lambdas=lambdas)
            
            if iter_cnt==_iter_cnt:
                aug_lagrangians.append(aug_lagrangian.item())
                aug_lagrangian_ma.append(aug_lagrangian.item())
                grad_norms.append(model.get_grad_norm("wbx").item())
                grad_norm_ma.append(model.get_grad_norm("wbx").item())
            else:
                aug_lagrangians.append(aug_lagrangian.item())
                aug_lagrangian_ma.append(aug_lagrangian_ma[-1]+ 0.01 * (aug_lagrangian.item() - aug_lagrangian_ma[-1]))
                grad_norms.append(model.get_grad_norm("wbx").item())
                grad_norm_ma.append(grad_norm_ma[-1] + 0.01 * (grad_norms[-1] - grad_norm_ma[-1]))

            if aug_lagrangian.item() < cur_min:
                cur_min=aug_lagrangian.item()
                cur_patience=0
            else:
                cur_patience+=1
            perf_str='Iter %d : training loss ==[%.5f = %.5f + %.5f], curr H: %.5f, curr patience: %d' % (
                iter_cnt, aug_lagrangians[-1], nlls[-1], not_nlls[-1], cur_h, cur_patience)
            #print(perf_str)
            iter_cnt+=1
            
            if cur_patience>args.gran_patience:
                with torch.no_grad():
                    h_new = compute_constraint(model, w_adj).item()
                if h_new > 0.9 * h:
                    mu *= 10
                    cur_patience=0
                    cur_min=inf
                else:
                    lamb += mu * h_new
                    return mu, lamb, h_new, mus, lambdas, w_adjs, iter_cnt
                    
    lamb += mu * h_new
    return mu, lamb, h_new, mus, lambdas, w_adjs, iter_cnt







            