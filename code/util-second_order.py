import numpy as np
from time import time
import theano
import theano.tensor as T
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import fmin_cg
from scipy.linalg import eigvalsh
import pandas as pd

X = T.fmatrix("X")
Y = T.lvector("Y")
V = T.fvector("V")
e = T.fvector("e")

tolerance = 1e-6

def compile(d):
    start = time()

    global loss
    global params
    global iterate
    global iterate_neg
    global iterate_loss
    global iterate_hessian
    global numParams
    global error

    global iterate_Hv
    
    nHidden = int(d['n_hidden'])
    nLayers = int(d['n_layers'])
    n_input = int(d['input_size'])

    layer_dims = [n_input, nHidden]
    numParams = (n_input + 1)*nHidden
    
    for i in range(1, nLayers):
        layer_dims.append(nHidden)
        numParams += (nHidden + 1)*nHidden
    layer_dims.append(10)
    numParams += (nHidden + 1)*10
    parameters = np.random.normal(0, 1, numParams).reshape(numParams)
    params = theano.shared(parameters, "P")
    index = 0
    layers = []
    for i in range(1, len(layer_dims)):
        W_size = layer_dims[i - 1] * layer_dims[i]
        W = params[index:(index + W_size)].reshape((layer_dims[i], layer_dims[i - 1])).T
        b = params[(index + W_size):(index + W_size + layer_dims[i])]
        index += W_size + layer_dims[i]
        layers.append((W, b))
    if nLayers == 1:
        hiddens = T.nnet.relu(T.dot(X, layers[0][0]) + layers[0][1])
    if nLayers == 2:
        hiddens_ = T.nnet.relu(T.dot(X, layers[0][0]) + layers[0][1])
        hiddens = T.nnet.relu(T.dot(hiddens_, layers[1][0]) + layers[1][1]) 
    outputs = T.nnet.softmax(T.dot(hiddens, layers[nLayers][0]) + layers[nLayers][1])
    loss = -T.mean(T.log(outputs)[T.arange(Y.shape[0]), Y])
    
    Hv = T.Lop(T.grad(loss, params), params, V) # hessian times a vector mu bu
    # Hv = T.Rop(T.grad(loss, params), params, V) # hessian times a vector mu bu
    iterate_Hv = theano.function([X, Y, V], Hv)

    raleigh = T.Lop(T.grad(loss, params), params, V / V.dot(V)).dot(V) #put a minus in front of V for max eig
    raleigh_grad = T.grad(raleigh, V)
    iterate = theano.function([X, Y, V], (raleigh, raleigh_grad))
    iterate_neg = theano.function([X, Y, V], (-raleigh, -raleigh_grad))
    error = theano.function([X, Y], T.neq(T.argmax(outputs, 1), Y).mean())
    iterate_loss = theano.function([X, Y], loss)
    hessian = T.Lop(T.grad(loss, params), params, e)
    iterate_hessian = theano.function([X, Y, e], hessian)
    elapsed = (time() - start)
    print('time passed for compiling: %.3f ' % elapsed) 


# def rayleigh_minimization(d, o)
#     tolerance = o['rayleight_tolerane']
#     train_x = d['tr_x'].reshape(d['nSamples'], d['input_size'])
#     train_y = d['tr_y_python'].astype('int')
#     params.set_value((d['p'].astype("float32")).reshape(numParams)) 
#     n_params = len(d['p'])

#     def helper(vec):
#         eig_, V_ = iterate_neg(train_x.astype("float32"), train_y.astype("int64"), vec.astype("float32"))
#         return eig_, V_.astype("float64")



def rightEig(d): # on its negative so negate the result
    train_x = d['tr_x'].reshape(d['nSamples'], d['input_size'])
    train_y = d['tr_y_python']
    params.set_value((d['p'].astype("float32")).reshape(numParams)) 
    n_params = len(d['p'])
    def helper(vec):
        eig_, V_ = iterate_neg(train_x.astype("float32"), train_y.astype("int64"), vec.astype("float32"))
        return eig_, V_.astype("float64")
    V0 = np.random.normal(0, 1, n_params).astype("float32")
    start = time()
    V, eig, _ = fmin_l_bfgs_b(helper, V0, pgtol=tolerance) 
    pd.DataFrame([-eig]).to_csv(d['path'], mode='a', header=None, index=False)
    elapsed = (time() - start)
    print('at right most ev: %.5f ' % -eig, 'time passed: %.3f ' % elapsed) 
    dictRightEig = dict(vector = V, eigenvalue = eig.item(), 
                       loss=iterate_loss(train_x.astype("float32"), train_y.astype("int64")).item(),
                       elapsed=elapsed)
    return dictRightEig

def lanczos_(d): # on its negative so negate the result
    train_x = d['tr_x'].reshape(d['nSamples'], d['input_size'])
    train_y = d['tr_y_python']
    params.set_value((d['p'].astype("float32")).reshape(numParams)) 
    n_params = len(d['p'])
    def helper(vec):
        return iterate_Hv(train_x.astype("float32"), train_y.astype("int64"), vec.astype("float32"))

    V_ = np.random.normal(0, 1, n_params).astype("float32")
    V_ = V_/np.linalg.norm(V_)
    V = 0
    b = 0
    m = 100
    for i in range(m):
        W_ = iterate_Hv(train_x.astype("float32"), train_y.astype("int64"), V_.astype("float32"))
        alpha = W_.dot(V_)
        W = (W_ - alpha*V_ - b*V).copy()
        b = np.linalg.norm(W)
        V = V_.copy()
        V_ = (W/b).copy()
        print alpha





def secondRightEig(d): # on its negative so negate the result
    train_x = d['tr_x']
    train_y = d['tr_y_python']
    largestEV = d['right_vector']
    params.set_value((d['p'].astype("float32")).reshape(numParams)) 
    n_params = len(d['p'])
    list_EV = [largestEV]
    eigs = []
    n = int(d['from_right'])
    for i in range(1, n):
        def helper(vec):
            matrix = np.concatenate((list_EV, [vec]), axis=0)
            q, r = np.linalg.qr(matrix.transpose())
            vec = q.transpose()[i]*r[i][i]
            eig_, V_ = iterate_neg(train_x.astype("float32"), train_y.astype("int64"), vec.astype("float32"))
            return eig_, V_.astype("float64")
        V0 = np.random.normal(0, 1, n_params).astype("float32")
        start = time()
        V, eig, _ = fmin_l_bfgs_b(helper, V0, pgtol=tolerance) # 1e-9 is better
        pd.DataFrame([-eig]).to_csv(d['path'], mode='a', header=None, index=False)
        elapsed = (time() - start)
        print('at step ' + str(i) + ' ev: %.5f ' % -eig, 'time passed: %.3f ' % elapsed)
        list_EV.append(V)
        eigs.append(eig.item())
        dictSecondRightEig = dict(eigenvalues = eigs[::-1])
    return dictSecondRightEig


def leftEig(d): 
    train_x = d['tr_x']
    train_y = d['tr_y_python']
    params.set_value((d['p'].astype("float32")).reshape(numParams)) 
    n_params = len(d['p'])
    def helper(vec):
        eig_, V_ = iterate(train_x.astype("float32"), train_y.astype("int64"), vec.astype("float32"))
        return eig_, V_.astype("float64")
    V0 = np.random.normal(0, 1, n_params).astype("float32")
    start = time()
    V, eig, _ = fmin_l_bfgs_b(helper, V0, pgtol=tolerance)
    pd.DataFrame([eig]).to_csv(d['path'], mode='a', header=None, index=False)
    elapsed = (time() - start)
    print('at left most ev: %.5f ' % eig, 'time passed: %.3f ' % elapsed)
    dictLeftEig = dict(vector = V, eigenvalue = eig.item(), 
                       err=error(train_x.astype("float32"), train_y.astype("int64")).item(),
                       elapsed=elapsed)
    return dictLeftEig


def secondLeftEig(d): # on its negative so negate the result
    train_x = d['tr_x']
    train_y = d['tr_y_python'] 
    smallestEV = d['left_vector']
    params.set_value((d['p'].astype("float32")).reshape(numParams)) 
    n_params = len(d['p'])
    list_EV = [smallestEV]
    eigs = []
    n = int(d['from_left'])
    for i in range(1, n):
        def helper(vec):
            matrix = np.concatenate((list_EV, [vec]), axis=0)
            q, r = np.linalg.qr(matrix.transpose())
            vec = q.transpose()[i]*r[i][i]
            eig_, V_ = iterate(train_x.astype("float32"), train_y.astype("int64"), vec.astype("float32"))
            return eig_, V_.astype("float64")
        V0 = np.random.normal(0, 1, n_params).astype("float32")

        start = time()
        V, eig, _ = fmin_l_bfgs_b(helper, V0, pgtol=tolerance)
        pd.DataFrame([eig]).to_csv(d['path'], mode='a', header=None, index=False)
        elapsed = (time() - start)
        print('at step ' + str(i) + ' ev: %.5f ' % -eig, 'time passed: %.3f ' % elapsed)
        list_EV.append(V)
        eigs.append(eig.item())
        dictSecondLeftEig = dict(eigenvalues = eigs)
    return dictSecondLeftEig


def fullH(d):
    # if d['i']:
    #     compile(d)
    train_x = d['tr_x']
    train_y = d['tr_y_python']
    params.set_value((d['p'].astype("float32")).reshape(numParams)) 
    n_params = len(d['p'])
    fullHessian = []
    start = time()

    for i in range(n_params):
        e_ = np.zeros(n_params).astype("float32")
        e_[i] = 1
        hes_ = iterate_hessian(train_x.astype("float32"), train_y.astype("int64"), e_.astype("float32"))
        fullHessian.append(hes_)  # check if symmetric...
    elapsed = (time() - start)
    H = np.asarray(fullHessian).reshape(n_params, n_params)
    return dict(H=H, elapsed=elapsed)


def helper_for_torch(d): # on its negative so negate the result
    train_x = d['tr_x']
    train_y = d['tr_y_python']
    params.set_value((d['p'].astype("float32")).reshape(numParams)) 
    vec = d['p']
    f, df = iterate(train_x.astype("float32"), train_y.astype("int64"), vec.astype("float32"))
    dictHelper = dict(f = f.item(), df = df)
    return dictHelper


