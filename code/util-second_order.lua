local py = require('fb.python')


function util.second_order_prep() py.exec('execfile("util-second_order.py")') end
function util.compiled(d) return py.eval('compile(d)', {d = d}) end
function util.helper_torch(d) return py.eval('helper_for_torch(d)', {d = d}) end
function util.leftEigGet(d) return py.eval('leftEig(d)', {d = d}) end
function util.rightEigGet(d) return py.eval('rightEig(d)', {d = d}) end
function util.secondRightEigGet(d) return py.eval('secondRightEig(d)', {d = d}) end
function util.secondLeftEigGet(d) return py.eval('secondLeftEig(d)', {d = d}) end
function util.fullHGet(d) return py.eval('fullH(d)', {d = d}) end
function util.Lanczos(d) return py.eval('lanczos_(d)', {d = d}) end

-- function util.data_to_python()
    -- d['X'] = data['tr_x']:clone():float():reshape(opt.n_samples_train, data['input_size'])
    -- subtract 1: line 121 at https://github.com/nicholas-leonard/dp/blob/master/data/mnist.lua
    data['tr_y_python'] = data['tr_y']:clone():add(-1)--:float()
    data['n_hidden'] = opt.n_hidden
    data['n_layers'] = opt.n_layers
    data['nSamples'] = opt.n_samples_train
-- end

-- changed d to data
-- could pass a subset...
-- bs = 0.10*opt.n_samples_train -- use ten percent of the data for ev calculations.
-- local indices = util.permuteNfold(data['tr_y'], 1, bs)
-- d['X'] = data['tr_x']:index(1, indices[1][i]):clone():float():reshape(bs, data['input_size'])
-- d['y'] = data['tr_y']:index(1, indices[1][i]):clone():add(-1):float()

if opt.hessian then
    util.second_order_prep()
    util.compiled(data)
end


function util.log_top(worker_id)
    if not opt.silent then print('for worker ' .. worker_id) end        
    data['p'] = worker[worker_id].W:float()
    data['path'] = opt.save .. '-eigenvalues_' .. worker_id .. '_right.csv'
    local rightEig = util.rightEigGet(data)
    local lanc = util.Lanczos(data)
    data['right_vector'] = rightEig['vector']
    return rightEig['eigenvalue']
end


function util.log_bottom(worker_id)
    if not opt.silent then print('for worker ' .. worker_id) end        
    data['p'] = worker[worker_id].W:float()
    data['path'] = opt.save .. '-eigenvalues_' .. worker_id .. '_left.csv'
    local leftEig = util.leftEigGet(data)
    data['left_vector'] = leftEig['vector'] 
end


function util.log_top_K(worker_id, K)
    data['from_right'] = K
    data['p'] = worker[worker_id].W:float()
    data['path'] = opt.save .. '-eigenvalues_' .. worker_id .. '_right.csv'
    local secondRightEig = util.secondRightEigGet(data)
end


function util.log_bottom_K(worker_id, K)
    data['from_left'] = K
    data['p'] = worker[worker_id].W:float()
    data['path'] = opt.save .. '-eigenvalues_' .. worker_id .. '_left.csv'
    local secondLeftEig = util.secondLeftEigGet(data)
end


function util.log_FULL_hessian(i) -- FIX THIS
    local dict_results = fullHGet(d)
    local H = dict_results['H']:float()
    print('hessian calc: ', util.round(dict_results['elapsed'], 2))
    local t = torch.tic()
    local e = torch.symeig(H)
    if not opt.silent then 
        print('theano allEigs - from top: ', e[{{-d['from_right'], -1}}],
              'theano allEigs - from bottom: ', e[{{1, d['from_left']}}],
              'time: ', util.round(torch.toc(t), 2)) 
    end 
    -- data = torch.totable(e)
    -- add save functions for eigenvalues and the full Hessian...
end

