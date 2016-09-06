local cutorch = require "cutorch"
local optim = require "optim"
local nn = require "nn"
local tnt = require "torchnet"
local mp = require "MessagePack"
local cmd = torch.CmdLine()

-- Set up command-line arguments
cmd:text("Multimodal selectional preference prediction.")
cmd:option("-trainFile", "/local/filespace/am2156/mm_verbs/data/example.mpk", "Path to the training file")
cmd:option("-testFile", "/local/filespace/am2156/mm_verbs/data/example.mpk", "Path to the test file")
cmd:option("-modelDir", "/local/filespace/am2156/mm_verbs/models", "Path to the directory where to save models.")
cmd:option("-dimHid", 100, "Size of hidden vector.")
cmd:option("-nEpochs", 1, "Number of training epochs.")
cmd:option("-margin", 0.1, "Margin for ranking loss.")
cmd:option("-cuda", false, "Train on GPU.")

opt = cmd:parse(arg)
print("opt:", opt)

-- GPU data set up
print("Specifying GPU settings...")
local tensor
if cuda then
   tensor = cutorch.CudaTensor
else
   tensor = torch.Tensor
end

-- Load data
print("Loading data...")
function loadDataset(filepath)
   rawData = mp.unpack(io.open(filepath)
			       :read("*all"))
   Data = tnt.ListDataset{
      list = torch.range(1, #rawData):long(),
      load = function(idx)
	 return {
	    input = {
	       torch.cat(tensor(rawData[idx][1][2]),
			 tensor(rawData[idx][2][2])),
	       torch.cat(tensor(rawData[idx][3][2]),
			 tensor(rawData[idx][4][2]))
	       
	    },
	    target = tensor({1})
	 }
      end
   }

   return tnt.DatasetIterator{dataset = Data}
end

trainIterator = loadDataset(opt.trainFile)
testIterator = loadDataset(opt.testFile)

dimIn = 2*4096

-- Define model
print("Defining model...")
model = nn.MapTable()
   :add(
      nn.Sequential()
	 :add(nn.Linear(dimIn, opt.dimHid))
	 :add(nn.Linear(opt.dimHid, 1)))

criterion = nn.MarginRankingCriterion(opt.margin)

if cuda then
   model:cuda()
   criterion:cuda()
end

-- Define engine
print("Setting up engine...")
engine = tnt.OptimEngine{}
local loss = tnt.AverageVaylueMeter()
engine.hooks.onStartEpoch = function(state)
   print("Starting epoch ", state.epoch)
   loss:reset()
end

engine.hooks.onForwardCriterion = function(state)
   print("Example: ", state.t)
   print("Loss: ", state.criterion.output)
   loss:add(state.criterion.output)
end

engine.hooks.onEndEpoch = function(state)
   print("Average loss:", loss:value())
end

engine.hooks.onEnd = function(state)
   torch.save(opt.modelDir .. "/model" .. state.epoch .. ".t7",
	      state.network, "binary")
end

-- Train model
print("Training model...")
engine:train{
   network = model,
   iterator = trainIterator,
   criterion = criterion,
   optimMethod = optim.adam,
   maxepoch = opt.nEpochs
}

-- Evaluate model
print("Evaluating model...")
engine:test{
   network = model,
   criterion = criterion,
   iterator = testIterator
}
