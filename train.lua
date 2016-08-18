require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'LSTM'
require 'LRCN'
require 'util.DataLoader'

local utils = require 'util.utils'

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-trainList', '')
cmd:option('-valList', '')
cmd:option('-testList', '')
cmd:option('-imageType', 'jpg')
cmd:option('-scaledHeight', '')
cmd:option('-scaledWidth', '')
cmd:option('-videoHeight', '')
cmd:option('-videoWidth', '')
cmd:option('-batchSize', 2)

-- Model options

-- Optimization options
cmd:option('numEpochs', 30)
cmd:option('-learningRate', 1e-6)
cmd:option('-lrDecayFactor', 0.5)
cmd:option('lrDecayEvery', 5)

-- Output options
cmd:option('-printEvery', 1)
cmd:option('-checkpointName', 'checkpoints/checkpoint')

cmd:option('-cuda', 1)

local opt = cmd:parse(arg)
for k, v in pairs(opt) do
  if tonumber(v) then
    opt[k] = tonumber(v)
  end
end

-- set GPU
opt.dtype = 'torch.FloatTensor'
if opt.cuda == 1 then
  require 'cunn'
  opt.dtype = 'torch.CudaTensor'
end

-- load dataset
utils.printTime("Initializing DataLoader")
local loader = DataLoader(opt)

-- initialize model
utils.printTime("Initializing LRCN")
local model = LRCN(opt):type(opt.dtype)

function compress(phi)

end

function train(model)
  utils.printTime(string.format("Starting training for %d epochs"), opt.numEpochs)

  local trainLossHistory = {}
  local valLossHistory = {}
  local testLossHistory = {}

  local params, gradParams = model.getParameters()

  for i = 1, opt.numEpochs do
    collectgarbage()

    local opochLoss = {}

    if i % opt.lrDecayEvery == 0 then
      local oldLearningRate = config.learningRate
      config = {learningRate = oldLearningRate * opt.lrDecayFactor}
    end


    local batch = loader:nextBatch('train')

    while batch ~= nil do
      if opt.cuda == 1 then
        batch.data = batch.data:cuda()
        batch.label = batch.label:cuda()
      end
    end

    local function feval(x)
      collectgarbage()

      if x ~= params then
        params:copy(x)
      end

      gradParams:zero()

      local criterion = nn.MSECriterion()
      local modelOut = model:forward(batch.data)
      local frameLoss = criterion:forward(modelOut, batch.labels)
      local gradOutput = criterion:backward(modelOut, batch.labels)
      local gradModel = model:backward(batch.data, gradOutput)

      return frameLoss, gradParams
    end


    local _, loss = optim.adam(feval, params, config)
    table.insert(epochLoss, loss[1])

    batch = loader:nextBatch('train')
  end

  local epochLoss = torch.mean(torch.Tensor(epochLoss))
  table.insert(trainLossHistory, opochLoss)

  if(opt.printEvery > 0) then
    utils. printTime(string.format("Epoch %d training loss: %f", i), epochLoss)
end

  if (opt.checkpointEvery > 0) or i == opt.numEpochs then
    local valLoss = test(model, 'val', 'loss')
    utils.printTime(string.format("Epoch %d validation loss: %f", i, valLoss))
    table.insert(valLossHistory, valLoss)
    table.insert(valLossHistoryEpochs, i)
  end
end

function test(model, split, task)
  collectgarbage()
  utils.printTime(string.format("Starting %s testing on the %s split", task, split))

  local evalData = {}
  evalData.predictedLabels = {}
  evalData.trueLabels = {}

  local batch = loader:nextBatch(split)

  while batch ~= nil do
    if opt.cuda == 1 then
      batch.data = batch.data:cuda()
      batch.labels = batch.labels:cuda()
    end

    local numData = batch:size() / checkpoint.opt.seqLength
    local scores = mode:forward(batch.data)

    for i = 1, numData do
      local startIndex = (i - 1) * checkpoint.opt.seqLength + 1
      local endIndex = i * checkpoint.opt.seqLength
    end
  end
end

train(model)

