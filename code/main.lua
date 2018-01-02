require 'optim'

-- guzellikler
local cmd = torch.CmdLine()
-- data parameters
cmd:option('--model', 'nn', 'model type: nn | hinton | lenet | nin ')
cmd:option('--data', 'mnist', 'data: mnist | cifar10 ')
cmd:option('--mnist_type', 'regular', 'regular | double | scrambled | all_noise | some_noise ')
cmd:option('--n_samples_train', 1000, 'number of training samples')
cmd:option('--n_samples_test', 10000, 'number of test samples') -- make this obsolete...
-- model parameters
cmd:option('--n_hidden', 100, 'number of hidden units in nn')
cmd:option('--n_layers', 2, 'number of hidden layers in nn')
cmd:option('--weight_init_std', 0, 'std of weight initialization')
cmd:option('--p', 0.91, 'correct label hintonized')
cmd:option('--q', 0.01, 'incorrect label hintonized')
cmd:option('--optimization', 'SGD', 'optimization method: SGD | GD | LBFGS')
-- training parameters
cmd:option('--batch_size', 100, 'size of the mini batch')
cmd:option('--lr', 0.1, 'learning rate')
cmd:option('--mom', 0.5, 'momentum')
cmd:option('--stopwhen', 0, 'stopping condition on the training cost')
cmd:option('--n_steps', 100, 'total number of epochs')
cmd:option('--n_workers', 1, 'number of workers (for the EASGD)')
cmd:option('--alpha', 0, 'interection strength of workers')
-- system level parameters
cmd:option('--type', 'float', 'tensor type: float | cuda')
cmd:option('--device_num', 1, 'GPU device number')
cmd:option('--dummy', true, 'dummy for compiling theano functions')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:option('--user_input', false, 'asks user input')
cmd:option('--save', '', 'keep it empty string for experiments, give a name for random trials and erase them')
-- data collection ways and frequency
cmd:option('--stop', 'grad', 'stopping criterion: norm_grad | cost_value')
cmd:option('--freq', 10, 'how often keep a record and/or print')
cmd:option('--freq_i', 50, 'how often keep a record of interpolations and/or print')
cmd:option('--resolution', 10, 'number of points to interpolate between')
cmd:option('--c1', -1, 'starting point is the initial point if 0') -- for the resolution of interpolation
cmd:option('--c2', 2, 'ending point is the final point if 1')
cmd:option('--hessian', false, 'second order methods')
cmd:option('--rayleight_tolerance', 1e-6, 'tolerance for the lbfgs step, try 1e-6 and 9')
cmd:option('--sharpness', false, 'sharpness calculation')
cmd:option('--sharpness_epsilon', 0.01, 'sharpness stopping distance')
-- random walk after training
cmd:option('--explore', false, 'do random walk for the central GD at the end when alpha is 0')
cmd:option('--threshold', 0.05, 'threshold for the excursion set exploration')
cmd:option('--rw_step_size', 0.005, 'step size of the random walk')
opt = cmd:parse(arg or {})
opt.seed = os.time()
opt.save = '../results/' .. opt.save .. opt.seed
if not opt.silent then print(opt) end
--initialize the system
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(24)
torch.manualSeed(opt.seed)
-- load other files
paths.dofile('util-main.lua') 
paths.dofile('util-stats.lua')
paths.dofile('util-train.lua')
paths.dofile('util-interpolation.lua')
paths.dofile('load-data.lua')
paths.dofile('load-model.lua')
paths.dofile('util-second_order.lua')
paths.dofile('util-sharpness.lua')
paths.dofile('util-random_walk.lua')

--------------------------------------------------------------------
-- experiment modules ----------------------------------------------
--------------------------------------------------------------------

util.logAll() -- initial state of the system
-- opt.lr = -2/util.log_top(0)
-- print('adapted lr: ', opt.lr)

util.train(data['tr_x'], data['tr_y'])
util.log_top(0)

-- if opt.user_input then 
--     util.train_user_input(data['tr_x'], data['tr_y'])
-- else    
--     util.train(data['tr_x'], data['tr_y'])
-- end

-- if opt.user_input then
-- 	for i = 0, opt.n_workers do
-- 		util.local_max(i, opt.sharpness_epsilon)
-- 	end
-- end


-- if opt.explore then 
--     local id = 0 -- worker id for the random walk, usually zero for GD
--     print('random walking for, ' .. id)
--     util.random_walker(id)
-- end


-- if opt.hessian then 
--     util.log_top(1)
--     util.log_top_K(1, 5)
--     util.log_bottom(1)
--     util.log_top_K(1, 3)
-- end

-- --------------------------------------------------------------------
-- -- compose experiment ----------------------------------------------
-- --------------------------------------------------------------------







