#2weeks assgiment -> change tensor device cpu to gpu
from sre_constants import error
try:
  import torch
  print(torch.__version__)
  print("torch import success")
  torch.device = "cuda"
  test_tensor = torch.tensor([1,2,3])
  print(test_tensor)
  print(test_tensor.shape)
  print(test_tensor.device)

  #change tensor resource cpu to gpu
  test_tensor = test_tensor.to(torch.device)
  print(torch.device)
  print(test_tensor.device)
except:
  print(error)

