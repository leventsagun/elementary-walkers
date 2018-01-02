import cPickle, gzip, numpy

def load_mnist():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    data = {}
    data['tr_x'] = ((train_set[0] - numpy.mean(train_set[0]))/numpy.std(train_set[0])).reshape((50000, 28, 28))
    data['te_x'] = ((test_set[0] - numpy.mean(test_set[0]))/numpy.std(test_set[0])).reshape((10000, 28, 28))
    data['val_x'] = ((valid_set[0] - numpy.mean(valid_set[0]))/numpy.std(valid_set[0])).reshape((10000, 28, 28))
    data['tr_y'] = train_set[1]
    data['te_y'] = test_set[1]
    data['val_y'] = valid_set[1]
    return data


# Compare these two ways of normalization...
   #  function dataset:normalize(mean_, std_)
   #    local mean = mean_ or data:view(data:size(1), -1):mean(1)
   #    local std = std_ or data:view(data:size(1), -1):std(1, true)
   #    for i=1,data:size(1) do
   #       data[i]:add(-mean[1][i])
   #       if std[1][i] > 0 then
   #          tensor:select(2, i):mul(1/std[1][i])
   #       end
   #    end
   #    return mean, std
   # end

   # function dataset:normalizeGlobal(mean_, std_)
   #    local std = std_ or data:std()
   #    local mean = mean_ or data:mean()
   #    data:add(-mean)
   #    data:mul(1/std)
   #    return mean, std
   # end