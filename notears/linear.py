import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt
import os
import tqdm as tqdm
beta = 0.1
reweight_list = []
epoch = 6
def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3, B_true=None):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps #
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W  
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R 

        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _reweighloss(W):
        M = X @ W
        R = X - M
        assert loss_type in ['l2', 'logistic', 'poisson']
        rewei_num = len(reweight_list)
        re_matrix = np.eye(100,100)
        for idx in reweight_list:
            re_matrix[idx][idx] = beta
        loss = 0.5 / X.shape[0] * ((re_matrix @ R) ** 2).sum()
        G_loss = - 1.0 / X.shape[0] * X.T @ ((re_matrix**2) @ R)
        # return loss, G_loss
        return loss,G_loss

    
    def _single_loss(W):
        
        M = X @ W
        assert loss_type == 'l2'
        R = X - M
        
        loss = 0.5 / X.shape[0] * (R ** 2)
        return loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        
        if np.any(np.isnan(E)) or np.any(np.isinf(E)):
            print('nan in expm')
            return 1
        h = np.trace(E) - d
        if np.any( np.isnan(h)) or np.any(np.isinf(h)):
            print('nan in h')
            return 1
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate(
            (G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    def _refunc(w):
        W = _adj(w)
        loss, G_loss = _reweighloss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate(
            (G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape  
    
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf

    bnds = [(0, 0) if i == j else (0, None) for _ in range(2)
            for i in range(d) for j in range(d)]  
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)

    ob_loss = []
    total_loss = []
    xepoch = 0
    for iter in tqdm.tqdm(range(max_iter)):
        w_new, h_new = None, None
        while rho < rho_max:
            xepoch+=1
            if xepoch<epoch:
                sol = sopt.minimize(
                    _func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            elif xepoch==epoch:
                w_new = sol.x
                loss_record = _single_loss(_adj(w_new))
                each_loss = loss_record.sum(axis=1)
                each_loss_idx = each_loss.argsort()
                reweight_list = each_loss_idx[:int(len(each_loss_idx)*0.1)]
                sol = sopt.minimize(
                    _refunc, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
                # sol = sopt.minimize(
                #     _func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            elif xepoch>epoch:
                sol = sopt.minimize(
                    _refunc, w_est, method='L-BFGS-B', jac=True, bounds=bnds)

            w_new = sol.x  
            loss_record = _single_loss(_adj(w_new))
            
            each_loss = loss_record.sum(axis=1)
            ob_loss.append(each_loss)
            total_loss.append(_loss(_adj(w_new))[0])
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:       
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new     
        alpha += rho * h            
        if h <= h_tol or rho >= rho_max:  
            break
    observed_loss = ob_loss[0].reshape(100, 1)
    for i in range(1, len(ob_loss)):
        observed_loss = np.concatenate(
            (observed_loss, ob_loss[i].reshape(100, 1)), axis=1)
    plt.figure(figsize=(50, 50))
    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.plot(observed_loss[i])
        plt.title(i, fontsize=10)
        plt.title('node {}'.format(i))
        plt.axis('on')
        plt.box(True)

    if not os.path.exists('linear_notears'):
        os.makedirs('linear_notears')
    plt.savefig('linear_notears/observation.png')    

    plt.figure(figsize=(20, 10))
    plt.plot(total_loss)
    plt.title('total loss(30nodes+noise*10)')
    plt.savefig('linear_notears/total_loss.png')


    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est, iter


if __name__ == '__main__':
    # from notears import utils
    import utils
    utils.set_random_seed(1)

    # n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
    n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    # np.savetxt('W_true.csv', W_true, delimiter=',')

    X = utils.simulate_linear_sem(W_true, n, sem_type)
    # np.savetxt('X.csv', X, delimiter=',')

    W_est, iter_num = notears_linear(
        X, max_iter=100, lambda1=0.1, loss_type='l2', B_true=B_true)
    print(f'iteration: {iter_num}')
    assert utils.is_dag(W_est)
    # np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)
