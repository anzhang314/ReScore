import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True,help='config file path')
    parser.add_argument('--device', default='cpu', help='cuda or cpu')

    parser.add_argument('--modeltype',default="notears",choices=['notears','golem','daggnn','grandag'])

    parser.add_argument("--s0", default=60, type=int)
    parser.add_argument("--d", default=20, type=int) 
    parser.add_argument("--n", default=1000, type=int)
    parser.add_argument("--sem_type", default="gp", choices=["gp-add","mlp", "gp", "mim"])
    parser.add_argument("--linear_sem_type", default="gauss", choices=["gauss","exp", "gumbel", "uniform","logistic","poisson"])
    parser.add_argument("--graph_type", default='ER', choices=['SF', 'ER', 'BA'])


    parser.add_argument('--data_dir', type=str, default='data', help='dataset_path')
    parser.add_argument('--seed', type=int, default=0, help='starting seed')
    parser.add_argument('--trial', type=int, default=10, help='number of trials')
    #parser.add_argument("--patience", default=20, type=int, help="learning patience for notears")
    parser.add_argument('--lambda1', type=float, default=0.01, help='lambda1')
    parser.add_argument('--lambda2', type=float, default=0.01, help='lambda2')
    parser.add_argument('--reweight', action='store_true', help='if reweight')
    parser.add_argument('--beta', type=float, default=0.9, help='beta')
    
    parser.add_argument("--w_threshold", default=0.3, type=float)
    parser.add_argument("--data_type", default='synthetic', type=str, help = 'real or synthetic', choices=['real', 'synthetic','testing', 'sachs_full'])#'synthetic_gran'])
    parser.add_argument("--linear", action='store_true', help ="whether to use linear synthetic data")

    # add the batch_size
    parser.add_argument("--run_mode", type = int, default=1, help ="run baseline or reweight operation")
    parser.add_argument("--figure", action='store_true', help ="record weight distribution")
    parser.add_argument('--batch_size', type=int, default=200, help='batch_size')
    parser.add_argument('--reweight_epoch', type=int, default=0, help='the epoch begin to reweight')
    
    parser.add_argument('--temperature', type=int, default=20, help='softmax_tmperature')
    parser.add_argument("--adaptive_epoch", default=20, type=int, help="number of iterations for adaptive reweight")
    parser.add_argument("--adaptive_lr", default=0.001, type=float, help="learning rate for adaptive reweight")
    parser.add_argument("--adaptive_lambda", default=0.001, type=float, help="adaptive lambda for l1 regularization")   

    # GOLEM 
    parser.add_argument("--duel_epoch", default=10, type=int, help = "duel ascent epoch for golem")
    parser.add_argument("--golem_lr", default=0.001, type=float, help="learning rate for golem")
    parser.add_argument("--golem_patience", default=20, type=int, help="learning patience for golem")


    # DAG-GNN

    parser.add_argument("--daggnn_lr", default=0.003, type=float, help="learning rate for daggnn")
    parser.add_argument("--daggnn_epochs", default=300, type=float, help="train epoch in each iteration for daggnn")


    # GraN-DAG
    parser.add_argument('--gran_optim', type=str, default="rmsprop",
                        help='sgd|rmsprop')
    parser.add_argument('--gran_lr', type=int, default=1e-3,
                        help="number of hidden layers")
    parser.add_argument('--gran_model', type=str, default="NonLinGaussANM",
                        help='model class')
    parser.add_argument('--gran_layers', type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument('--granhid_dim', type=int, default=10,
                        help="number of hidden units per layer")
    parser.add_argument('--norm-prod', type=str, default="paths",
                        help='how to normalize prod: paths|none')
    parser.add_argument('--square-prod', action="store_true",
                        help="square weights instead of absolute value in prod")
    parser.add_argument('--nonlin', type=str, default='leaky-relu',
                        help="leaky-relu | sigmoid")
    parser.add_argument('--no-w-adjs-log', action="store_true",
                        help='do not log weighted adjacency (to save RAM). One plot will be missing (A_\phi plot)')
    parser.add_argument('--edge-clamp-range', type=float, default=1e-4,
                        help='as we train, clamping the edges (i,j) to zero when prod_ij is that close to zero. '
                             '0 means no clamping. Uses masks on inputs. Once an edge is clamped, no way back.')
    parser.add_argument('--plot-freq', type=int, default=500,
                        help='plotting frequency')
    parser.add_argument('--stop-crit-win', type=int, default=100,
                        help='window size to compute stopping criterion')
    parser.add_argument('--graph_path', type=str, default="./logs/",
                        help='path to save plot')
    parser.add_argument("--gran_patience", default=100, type=int, help="learning patience for grandag")

    parser.add_argument('--pns', action="store_true",
                        help='Run `pns` function, get /pns folder')
    parser.add_argument('--cam-pruning', action="store_true",
                        help='Run `cam_pruning` function, get /cam-pruning folder')
    parser.add_argument('--pns-thresh', type=float, default=0.75,
                        help='threshold in PNS')



    #Group Plot Exp
    parser.add_argument("--scaled_noise", action='store_true', help ="noise scaling varying samples")
    parser.add_argument("--p_d",default=0.5, type=float, help="percent of non-informative nodes")
    parser.add_argument("--p_n",default=0.1, type=float, help="percent of informative samples")
    parser.add_argument("--noise_0",default=0, type=float, help="scale of informative noise" )
    parser.add_argument("--noise_1",default=1, type=float, help="scale of non-informative noise")


    return parser