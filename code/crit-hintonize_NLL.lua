local Hinton_1, parent = torch.class('nn.Hinton_1', 'nn.Criterion')

function Hinton_1:__init()
    parent.__init(self)
end


function Hinton_1:updateOutput(input, target)
    self.out = torch.Tensor(input:size()[1], 10):copy(input)
    self.output = -self.out:cmul(hintonize(target, opt.q, opt.p)):sum(2):mean()
    return self.output
end


function Hinton_1:updateGradInput(input, target)
    self.gradInput:resizeAs(input):zero():add(-hintonize(target, opt.q, opt.p)):div(input:size()[1])
    return self.gradInput
end
---------------------------------------------------------------------------------
local Hinton_2, parent = torch.class('nn.Hinton_2', 'nn.Criterion')

function Hinton_2:__init()
    parent.__init(self)
end


function Hinton_2:updateOutput(input, target)
    self.out = util.cast(torch.Tensor(input:size()[1], 10):copy(input))
    self.output = -self.out:cmul(target):sum(2):mean()
    return self.output
end


function Hinton_2:updateGradInput(input, target)
    self.gradInput:resizeAs(input):zero():add(-target):div(input:size()[1])
    return self.gradInput
end