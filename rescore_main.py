from cProfile import run
from cmath import exp, inf
from curses import curs_set
from os import rename
from turtle import goto
from sympy import re
import os

from sklearn.ensemble import ExtraTreesRegressor
from cdt.utils.R import RPackages, launch_R_script
from sklearn.feature_selection import SelectFromModel
from notears.trace_expm import trace_expm
import torch
import torch.nn as nn
from notears.lbfgsb_scipy import LBFGSBScipy
import numpy as np
import tqdm as tqdm
from notears.loss_func import *
import random
import time
from utils import *
import igraph as ig
import notears.utils as ut
import torch.utils.data as data
from adaptive_model.adapModel import *
from adaptive_model.baseModel import *
from runhelps.runhelper import config_parser
from torch.utils.tensorboard import SummaryWriter
from sachs_data.load_sachs import *
import json

COUNT = 0



parser = config_parser()
args = parser.parse_args()
print("args.n=======", args.n)
IF_baseline = args.run_mode
IF_figure = args.figure
print(args)


class TensorDatasetIndexed(data.Dataset):
    def __init__(self,tensor):
        self.tensor=tensor
    
    def __getitem__(self,index):
        return (self.tensor[index],index)
    
    def __len__(self):
        return self.tensor.size(0)



def record_weight(reweight_list, cnt, hard_list=[26,558,550,326,915], easy_list=[859,132,82,80,189]):
    writer = SummaryWriter('logs/weight_record_real')
    reweight_idx = reweight_list.squeeze()
    reweight_idx = reweight_idx.tolist()
    for idx in hard_list:
        writer.add_scalar(f'hard_real/hard_reweight_list[{idx}]', reweight_idx[idx], cnt)
    for idx in easy_list:
        writer.add_scalar(f'easy_real/easy_reweight_list[{idx}]', reweight_idx[idx], cnt) 

def record_distribution(reweight_list,R,j,idx):
    writer = SummaryWriter('logs/distribution_record')
    reweight_idx = reweight_list.squeeze()
    reweight_idx = reweight_idx.tolist()
    R=torch.sum(R,dim=1).squeeze()
    R=R.tolist()
    for i in range(len(reweight_idx)):
        writer.add_scalar(f'weight_distribution_step{j}', reweight_list[i], i)
    
    import matplotlib.pyplot as plt
   
    if args.scaled_noise:
        color=['b' if idx[i].item() < int(args.n*(1-args.p_n)) else 'r' for i in range(len(idx)) ]
    else:
        color='b'

    plt.cla()
    plt.scatter(R,reweight_idx,c=color)

    plt.savefig(f'logs/R_vs_weight_{j}_seed_{args.p_n}_{args.p_d}_{COUNT}.png')

def is_acyclic(adjacency):
    prod = np.eye(adjacency.shape[0])
    for _ in range(1, adjacency.shape[0] + 1):
        prod = np.matmul(adjacency, prod)
        if np.trace(prod) != 0: return False
    return True


def golem_linear(model: nn.Module,
                      adaptive_model: nn.Module,
                      X: np.ndarray,
                      train_loader: data.DataLoader,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 5,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3,
                      ):
    adp_flag = False
    for j in tqdm.tqdm(range(max_iter)):
        if j > args.reweight_epoch:
            adp_flag = True
            if not IF_baseline:
                print("Re-weighting")
                reweight_idx_tmp,R_tmp,idx= adap_reweight_step(args,adaptive_model, train_loader, args.adaptive_lambda , model, args.adaptive_epoch, args.adaptive_lr)
              
            h=dual_ascent_step_golem(args, model, X, train_loader, adp_flag, adaptive_model)
        else:
            h=dual_ascent_step_golem(args, model, X, train_loader, adp_flag, adaptive_model)
        
        if h <= h_tol:
            break
    

    W_est = model.W_to_adj()
 
    W_est[np.abs(W_est) < w_threshold] = 0

    while not ut.is_dag(W_est):
        w_threshold+=0.01
        W_est[np.abs(W_est) < w_threshold] = 0

    hard_index, easy_index = hard_mining(args, X, model, single_loss, ratio=0.01)
   
    return W_est, hard_index, easy_index

def notears_nonlinear(model: nn.Module,
                      adaptive_model: nn.Module,
                      X: np.ndarray,
                      train_loader: data.DataLoader,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 50,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3
                      ):
    
    rho, alpha, h = 1.0, 0.0, np.inf
    adp_flag = False
    for j in tqdm.tqdm(range(max_iter)):
        if j > args.reweight_epoch:
            adp_flag = True
            if not IF_baseline:
                print("Re-weighting")
                reweight_idx_tmp = adap_reweight_step(args,adaptive_model, train_loader, args.adaptive_lambda , model, args.adaptive_epoch, args.adaptive_lr)
             
                if IF_figure:
                    record_distribution(reweight_idx_tmp,j)
                
            rho, alpha, h = dual_ascent_step(args, model, X, train_loader, lambda1, lambda2,
                                         rho, alpha, h, rho_max, adp_flag, adaptive_model)

        else:
            rho, alpha, h = dual_ascent_step(args, model, X, train_loader, lambda1, lambda2,
                                         rho, alpha, h, rho_max, adp_flag, adaptive_model)
        
       
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0


    # if not is_acyclic(W_est):
    #     W_est_pre = W_est
    #     thresholds = np.unique(W_est_pre)
    #     for step, t in enumerate(thresholds):
    #         #print("Edges/thresh", model.adjacency.sum(), t)
    #         to_keep = np.array(W_est_pre > t + 1e-8)
    #         new_adj =  W_est_pre * to_keep

    #         if is_acyclic(new_adj):
    #             W_est=new_adj
    #             #model.adjacency.copy_(new_adj)
    #             break

  
    hard_index, easy_index = hard_mining(args, X, model, single_loss, ratio=0.01)
   
    return W_est, hard_index, easy_index

def daggnn_nonlinear(model: nn.Module,
                      adaptive_model: nn.Module,
                      X: np.ndarray,
                      train_loader: data.DataLoader,
                      max_iter: int = 20,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3,
                      true_graph=None
                      ):
    rho, alpha, h = 1.0, 0.0, np.inf
    adp_flag = False
    for j in tqdm.tqdm(range(max_iter)):
        if j > args.reweight_epoch:
            
            adp_flag = True
            if not IF_baseline:
                print("Re-weighting")
                reweight_idx_tmp = adap_reweight_step(args,adaptive_model, train_loader, args.adaptive_lambda , model, args.adaptive_epoch, args.adaptive_lr)
                
                if IF_figure:
                    record_distribution(reweight_idx_tmp,j)
                
            rho, alpha, h = dual_ascent_step_daggnn(args, model, X, train_loader,
                                         rho, alpha, h, rho_max, adp_flag, adaptive_model,true_graph)

        else:
            rho, alpha, h = dual_ascent_step_daggnn(args, model, X, train_loader,
                                         rho, alpha, h, rho_max, adp_flag, adaptive_model,true_graph)
        
        print(rho," ",alpha," ",h)

        W_est = model.get_adj()
        W_est[np.abs(W_est) < w_threshold] = 0

        acc = ut.count_accuracy(true_graph, W_est != 0)
        print(acc)
        
        if h <= h_tol or rho >= rho_max:
            break
  
    hard_index, easy_index = hard_mining(args, X, model, single_loss, ratio=0.01)
    return W_est, hard_index, easy_index

def pns_(model_adj, x, num_neighbors, thresh):
    """Preliminary neighborhood selection"""

    num_samples = x.shape[0]
    num_nodes = x.shape[1]
    print("PNS: num samples = {}, num nodes = {}".format(num_samples, num_nodes))
    for node in range(num_nodes):
        print("PNS: node " + str(node))
        x_other = np.copy(x.cpu())
        x_other[:, node] = 0
        reg = ExtraTreesRegressor(n_estimators=500)
        reg = reg.fit(x_other, x[:, node].cpu())
        selected_reg = SelectFromModel(reg, threshold="{}*mean".format(thresh), prefit=True,
                                       max_features=num_neighbors)
        mask_selected = selected_reg.get_support(indices=False).astype(np.float)

        model_adj[:, node] *= mask_selected

    return model_adj

def pns(model, x,  num_neighbors, thresh, exp_path):
    # Prepare path for saving results
    save_path = os.path.join(exp_path, f"pns_{thresh}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Check if already computed
    if os.path.exists(os.path.join(save_path, "DAG.npy")):
        print("pns already computed. Loading result from disk.")
        return load(save_path, "model.pkl")

    model_adj = model.adjacency.detach().cpu().numpy()
    time0 = time.time()
    model_adj = pns_(model_adj, x, num_neighbors, thresh)

    with torch.no_grad():
        model.adjacency.copy_(torch.Tensor(model_adj))

    timing = time.time() - time0
    print("PNS took {}s".format(timing))

    # save
    dump(model, save_path, 'model')
    dump(timing, save_path, 'timing')
    np.save(os.path.join(save_path, "DAG"), model.adjacency.detach().cpu().numpy())

    return model

def cam_pruning_(model_adj, x, cutoff, save_path, verbose=False):
    # convert numpy data to csv, so R can access them
    data_np = x.detach().cpu().numpy()
    data_csv_path = np_to_csv(data_np, save_path)
    dag_csv_path = np_to_csv(model_adj, save_path)

    #dag_pruned = cam_pruning(path_data, path_dag, cutoff, verbose)
    if not RPackages.CAM:
        raise ImportError("R Package CAM is not available.")

    arguments = dict()
    arguments['{PATH_DATA}'] = data_csv_path
    arguments['{PATH_DAG}'] = dag_csv_path
    arguments['{PATH_RESULTS}'] = os.path.join(save_path, "results.csv")
    arguments['{CUTOFF}'] = str(cutoff)

    if verbose:
        arguments['{VERBOSE}'] = "TRUE"
    else:
        arguments['{VERBOSE}'] = "FALSE"

    def retrieve_result():
        return pd.read_csv(arguments['{PATH_RESULTS}']).values

    dag_pruned = launch_R_script("{}/cam_pruning.R".format(os.path.dirname(os.path.realpath(__file__))),
                                     arguments, output_function=retrieve_result)

    # remove the temporary csv files
    os.remove(data_csv_path)
    os.remove(dag_csv_path)

    return dag_pruned

def cam_pruning(model, x, exp_path, cutoff=0.001, verbose=False):
    """Execute CAM pruning on a given model and datasets"""
    time0 = time.time()
    model.eval()

    # Prepare path for saving results
    stage_name = "cam-pruning/cutoff_%.6f" % cutoff
    save_path = os.path.join(exp_path, stage_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Check if already computed
    if os.path.exists(os.path.join(save_path, "DAG.npy")):
        print(stage_name, "already computed. Loading result from disk.")
        return np.load(os.path.join(save_path, "DAG.npy"))

    model_adj = model.adjacency.detach().cpu().numpy()

    dag_pruned = cam_pruning_(model_adj, x, cutoff, save_path, verbose)

    # set new adjacency matrix to model
    # model.adjacency.copy_(torch.as_tensor(dag_pruned).type(torch.Tensor))
    # Compute SHD and SID metrics
    np.save(os.path.join(save_path, "DAG"), dag_pruned)

    return dag_pruned

def file_exists(prefix, suffix):
    return os.path.exists(os.path.join(prefix, suffix))

def grandag_nonlinear(model: nn.Module,
                      adaptive_model: nn.Module,
                      X: np.ndarray,
                      train_loader: data.DataLoader,
                      max_iter: int = 1000,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3,
                      true_graph=None,
                      exp_path=None
                      ):
    
    if args.pns:
        num_neighbors = args.d
        print("Making pns folder")
        model=pns(model, X, num_neighbors, args.pns_thresh, exp_path)
        W_est = model.adjacency.cpu().detach().numpy().astype(np.float32)
        #acc = ut.count_accuracy(true_graph, W_est != 0)
        print("STAGE:pns")
        #print(acc)


    if file_exists(exp_path, f"pns_{args.pns_thresh}"):
        print("Training with pns folder")
        model = load(os.path.join(exp_path, f"pns_{args.pns_thresh}"), "model.pkl")
    else:
        print("Training from scratch")

    

    if os.path.exists(os.path.join(exp_path, "model.pkl")):
        print("Train already computed. Loading result from disk.")
        model=load(exp_path, "model.pkl")
    else:
        rho, alpha, h = 1e-3, 0.0, np.inf
        mus, lambdas, w_adjs= [],[],[]
        iter_cnt=0
        adp_flag = False
        for j in tqdm.tqdm(range(max_iter)):
            if j > args.reweight_epoch:
             
                adp_flag = True
                if not IF_baseline:
                    print("Re-weighting")
                    reweight_idx_tmp = adap_reweight_step(args,adaptive_model, train_loader, args.adaptive_lambda , model, args.adaptive_epoch, args.adaptive_lr)
                    if IF_figure:
                        record_distribution(reweight_idx_tmp,j)
                    
                rho, alpha, h, mus, lambdas,w_adjs,iter_cnt= dual_ascent_step_grandag(args, model, X, train_loader,
                                            rho, alpha, h, rho_max, adp_flag, adaptive_model,true_graph,mus,lambdas,w_adjs,iter_cnt)

            else:
                rho, alpha, h, mus, lambdas,w_adjs,iter_cnt = dual_ascent_step_grandag(args, model, X, train_loader,
                                            rho, alpha, h, rho_max, adp_flag, adaptive_model,true_graph,mus,lambdas,w_adjs,iter_cnt)
            
            if h <= h_tol or rho >= rho_max:
                break

        W_est_pre = model.get_w_adj().detach().cpu().numpy().astype(np.float32)


        # Find the smallest threshold that removes all cycle-inducing edges
        thresholds = np.unique(W_est_pre)
        for step, t in enumerate(thresholds):
            #print("Edges/thresh", model.adjacency.sum(), t)
            to_keep = torch.Tensor(W_est_pre > t + 1e-8).to(model.device)
            new_adj = model.adjacency * to_keep
            if is_acyclic(new_adj.cpu().detach()):
                model.adjacency.copy_(new_adj)
                break

    W_est = model.adjacency.cpu().detach().numpy().astype(np.float32)
    print(W_est)
    acc = ut.count_accuracy(true_graph, W_est != 0)
    print("STAGE:train")
    dump(model, exp_path, 'model')
    print(acc)

    if args.cam_pruning:
        best_shd=acc['shd']
        best_W=W_est
        prune_stats={-1:acc}
        # if opt.cam_pruning is iterable, extract the different cutoffs, otherwise use only the cutoff specified
        for cutoff in [1e-3,5e-3,1e-2,5e-2,1e-1,2e-1,3e-1]:
            # run
            dag_pruned = cam_pruning(model, X, exp_path, cutoff=cutoff)
            acc=ut.count_accuracy(true_graph, dag_pruned!=0)
            prune_stats[cutoff]=acc
            if acc['shd']<best_shd:
                best_shd=acc['shd']
                best_W=dag_pruned
            print(f"STAGE: cam_pruning_{cutoff}")
            print(acc)
        with open(os.path.join(exp_path, "pruning_stat.json"),'w') as f:
            json.dump(prune_stats,f)
        model.adjacency.copy_(torch.as_tensor(best_W).type(torch.Tensor))
        W_est=best_W
        
    

    hard_index, easy_index = hard_mining(args, X, model, single_loss, ratio=0.01)
    return W_est, hard_index, easy_index


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def hard_mining(args, data, model, loss_func, ratio = 0.01):
    """
    data: (N_observations, nodes)
    """
    data.to(args.device)
    model.to(args.device)
    N_sample = data.shape[0]
    model.eval()
    if args.modeltype!="grandag":
        data_hat = model.predict(data)
        loss_col = loss_func(data_hat, data)
        loss_col = torch.sum(loss_col, dim=1)
    else:
        weights, biases, extra_params = model.get_parameters(mode="wbx")
        log_likelihood = model.compute_log_likelihood(data, weights, biases, extra_params)
        loss_col=-torch.mean(log_likelihood)
    loss_col = loss_col.cpu().detach().numpy()

    hard_index_list = np.argsort(loss_col)[::-1][:int(N_sample * ratio)]
    easy_index_list = np.argsort(loss_col)[:int(N_sample * ratio)]
    return hard_index_list, easy_index_list

def main(trials,seed):


    tot_perf={}
    reweight_str=f"_reweight_{args.temperature}" if not IF_baseline else ""
    device=torch.device(args.device)
    args.device=device
    for trial in range(trials):
        print('==' * 20)

        import notears.utils as ut
        cur_seed=trial+seed
        global COUNT
        COUNT=cur_seed
        set_random_seed(cur_seed)

        if args.modeltype=="notears":
            if args.data_type == 'real' or args.data_type == 'sachs_full':
                model = NotearsMLP(dims=[11, 1], bias=True) # for the real data (sachs)   the nodes of sachs are 11
            else:
                if args.linear:
                    model = NotearsMLP(dims=[args.d, 1], bias=True)
                else:
                    model = NotearsMLP(dims=[args.d, 10, 1], bias=True)
        elif args.modeltype=="golem":
            model = GOLEM(args)
        elif args.modeltype=="daggnn":
            encoder=DAGGNN_MLPEncoder(args.d,1,10,1,args.batch_size)
            decoder=DAGGNN_MLPDecoder(1,1,10,args.batch_size,10)
            model = DAGGNN(encoder,decoder)
        elif args.modeltype=="grandag":
            if args.gran_model == "NonLinGauss":
                model = LearnableModel_NonLinGauss(args.d, args.gran_layers, args.gran_hid_dim, nonlin=args.nonlin,
                                                norm_prod=args.norm_prod, square_prod=args.square_prod,device=args.device)
            elif args.gran_model == "NonLinGaussANM":
                model = LearnableModel_NonLinGaussANM(args.d, args.gran_layers, args.granhid_dim, nonlin=args.nonlin,
                                                    norm_prod=args.norm_prod,
                                                    square_prod=args.square_prod,device=args.device)        

        noise_scale=None
        if args.scaled_noise:
          
            p_n=int((args.p_n)*args.n)
            p_d=int((args.p_d)*args.d)
            info_scale=np.concatenate([args.noise_1*np.ones(args.d-p_d),args.noise_0*np.ones(p_d)])
            non_info_scale=np.concatenate([args.noise_0*np.ones(p_d),args.noise_1*np.ones(args.d-p_d)])
            a1=np.tile(info_scale, (args.n-p_n,1))
            a2=np.tile(non_info_scale,(p_n,1))
            print(np.shape(a1))
            print(np.shape(a2))
            noise_scale=np.concatenate([a1,a2],axis=0)
            noise_scale=noise_scale.T

            print(noise_scale)
        

        
        
        
        if args.data_type=="synthetic" and args.linear:
            linearity = "linear"
        else:
            linearity = "non-linear"
        datatype = args.data_type
        sem_type = args.sem_type if linearity=="non-linear" else args.linear_sem_type
        if args.scaled_noise:
            sem_type+="_scaled"


        
        
        data_dir=f'data/{linearity}/{args.graph_type}_{sem_type}/'
        ensureDir(data_dir)

        
        
        if not args.scaled_noise:
            data_name=f'{args.d}_{args.s0}_{args.n}_{cur_seed}'
        else:
            data_name=f'{args.d}_{args.s0}_{args.n}_{args.p_d}_{args.p_n}_{args.noise_1}_{args.noise_0}_{cur_seed}'

        
        try:
            X=np.load(data_dir+data_name+"_X.npy")
            B_true=np.load(data_dir+data_name+"_B.npy")
            print("data loaded...")
        except:
            print("generating data from scratch...")
            if args.data_type == 'real':
            
                X = np.loadtxt('/opt/data2/git_fangfu/notears/sachs_data/sachs.csv', delimiter=',')
                B_true = np.loadtxt('/opt/data2/git_fangfu/notears/sachs_data/sachs_B_true.csv', delimiter=',')
              
            elif args.data_type == 'synthetic':
                B_true = ut.simulate_dag(args.d, args.s0, args.graph_type)
                if args.linear:
                    W_true=ut.simulate_parameter(B_true)
                    X = ut.simulate_linear_sem(W_true, args.n, args.linear_sem_type)
                    np.save(data_dir+data_name+"_W",W_true)
                else:
                    X = ut.simulate_nonlinear_sem(B_true, args.n, args.sem_type)
               
            elif args.data_type == 'testing':
                B_true = np.loadtxt('testing_B_true.csv', delimiter=',')
                X = np.loadtxt('testing_X.csv', delimiter=',')
    

            elif args.data_type == 'sachs_full':
                X = np.loadtxt('sachs_data/sachs7466.csv', delimiter=',')
                B_true = np.loadtxt('sachs_data/sachs_B_true.csv', delimiter=',')
             
            np.save(data_dir+data_name+"_X",X)
            np.save(data_dir+data_name+"_B",B_true)

        
        print("X[-1]====",X.shape[-1])
        print("args.n====",args.n)
        adaptive_model = adaptiveMLP(args.batch_size, input_size=X.shape[-1], hidden_size=args.hidden_size , output_size=1, temperature=args.temperature,device=args.device,linear=(linearity=="linear")).to(args.device)

        prefix = f'complexity_ob_{args.n}'
        #print(B_true)
        if args.data_type == 'real' or args.data_type == 'sachs_full':
            f_dir=f'{prefix}/{linearity}/{args.data_type}/{args.d}_{args.s0}_{args.n}/{args.data_type}/'
        else:
            f_dir=f'{prefix}/{linearity}/{args.data_type}/{args.d}_{args.s0}_{args.n}/{args.graph_type}_{sem_type}/'
        import os
        if not os.path.exists(f_dir):
            os.makedirs(f_dir)
        
        X = torch.from_numpy(X).float().to(args.device)
        model.to(args.device)

        X_data = TensorDatasetIndexed(X)
        train_loader = data.DataLoader(X_data, batch_size=args.batch_size, shuffle=True)
        if args.modeltype=="golem":
            W_est , _, _= golem_linear(model, adaptive_model, X, train_loader, args.lambda1, args.lambda2)
        elif args.modeltype=='notears':
            W_est , _, _= notears_nonlinear(model, adaptive_model, X, train_loader, args.lambda1, args.lambda2)
        elif args.modeltype=='daggnn':
            W_est , _, _= daggnn_nonlinear(model, adaptive_model, X, train_loader,true_graph=B_true)
        elif args.modeltype=='grandag':
            gran_dir=f_dir+"grandag"+reweight_str+f"/seed_{cur_seed}"
            ensureDir(gran_dir+'/')
            W_est , _, _= grandag_nonlinear(model, adaptive_model, X, train_loader,true_graph=B_true,exp_path=gran_dir)

        #assert ut.is_dag(W_est)
        # np.savetxt('W_est.csv', W_est, delimiter=',')
        #print(B_true)
        #print(W_est)
        acc = ut.count_accuracy(B_true, W_est != 0)
        print(acc)
       
        
        f_path=f'{prefix}/{linearity}/{args.data_type}/{args.d}_{args.s0}_{args.n}/{args.graph_type}_{sem_type}/seed_{cur_seed}.txt'
        

        if args.data_type == 'synthetic':
            with open(f_path, 'a') as f:
                f.write(f'args:{args}\n')
                f.write(f'run_mode: {IF_baseline}\n')
                f.write(f'observation_num: {args.n}\n')
                if not IF_baseline:
                    f.write(f'temperature: {args.temperature}\n')
                    f.write(f'batch_size:{args.batch_size}\n')
                    f.write(f'hidden_size:{args.hidden_size}\n')
                f.write(f'dataset_type:{args.data_type}\n')
                f.write(f'modeltype:{args.modeltype}\n')
                f.write(f'acc:{acc}\n')
                f.write('-----------------------------------------------------\n')
        

        for key,value in acc.items():
            if key not in tot_perf:
                tot_perf[key]={"value":[],"mean":[],"std":[]}
            tot_perf[key]["value"].append(value)
    for key,value in tot_perf.items():
        perf=np.array(value["value"])
        tot_perf[key]['mean']=float(np.mean(perf))
        tot_perf[key]['std']=float(np.std(perf))
    
    
    with open(f_dir+"stats_"+str(args.lambda1)+"_"+str(args.lambda2)+"_"+args.modeltype+reweight_str+str(args.hidden_size)+".json",'w') as f:
        json.dump(tot_perf,f)

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=10)
    main(args.trial,args.seed)
