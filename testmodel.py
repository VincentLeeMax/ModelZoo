#coding=utf-8
import tools.visualize as vis
import torchvision.models as models
import numpy as np
import torch
from torch.autograd import Variable

# x = np.arange(2*224*224*3)
# x = x.reshape(2,3,224,224)
# x = x/float(x.max())
# x = torch.from_numpy(x)
# x = x.float()
# x = Variable(x)
#
# densenet = models.densenet161()
# y = densenet(x)
# g = vis.make_dot(y)
# g.render("image", view=False)

# a = torch.IntTensor(3)
# print torch.__all__

from visdom import Visdom
import numpy as np
viz = Visdom(env='my_wind')#设置环境窗口的名称是'my_wind',如果不设置名称就在main中
tr_loss=list(range(100))
viz.line(Y=np.array(tr_loss), opts=dict(showlegend=True))