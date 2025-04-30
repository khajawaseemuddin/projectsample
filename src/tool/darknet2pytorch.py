import torch
import torch.nn as nn
import numpy as np
import math

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = self.parse_cfg(cfgfile)
        self.net_info, self.module_list = self.create_modules(self.blocks)
        
    def parse_cfg(self, cfgfile):
        """Parse the configuration file"""
        with open(cfgfile, 'r') as file:
            lines = file.read().split('\n')
            lines = [x.strip() for x in lines if x.strip() and not x.startswith('#')]
            
        blocks = []
        block = {}
        
        for line in lines:
            if line.startswith('['):
                if block:
                    blocks.append(block)
                block = {'type': line[1:-1].strip()}
            else:
                key, value = line.split('=')
                block[key.strip()] = value.strip()
                
        blocks.append(block)
        return blocks
    
    def create_modules(self, blocks):
        """Create PyTorch modules from parsed blocks"""
        net_info = blocks[0]
        module_list = nn.ModuleList()
        prev_filters = 3
        output_filters = []
        
        for index, block in enumerate(blocks[1:]):
            module = nn.Sequential()
            
            if block["type"] == "convolutional":
                filters = int(block["filters"])
                kernel_size = int(block["size"])
                stride = int(block["stride"])
                pad = int(block["pad"])
                activation = block["activation"]
                
                padding = (kernel_size - 1) // 2 if pad else 0
                
                conv = nn.Conv2d(prev_filters, filters, kernel_size, 
                                stride, padding, bias=not ("batch_normalize" in block))
                module.add_module(f"conv_{index}", conv)
                
                if "batch_normalize" in block:
                    bn = nn.BatchNorm2d(filters)
                    module.add_module(f"batch_norm_{index}", bn)
                
                if activation == "leaky":
                    activn = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module(f"leaky_{index}", activn)
                
            elif block["type"] == "upsample":
                stride = int(block["stride"])
                upsample = nn.Upsample(scale_factor=stride, mode="nearest")
                module.add_module(f"upsample_{index}", upsample)
            
            elif block["type"] == "route":
                layers = block["layers"].split(',')
                layers = [int(i.strip()) for i in layers]
                
                if len(layers) == 1:
                    filters = output_filters[layers[0]]
                else:
                    filters = sum([output_filters[l] for l in layers])
                
                route = EmptyLayer()
                module.add_module(f"route_{index}", route)
            
            elif block["type"] == "shortcut":
                shortcut = EmptyLayer()
                module.add_module(f"shortcut_{index}", shortcut)
            
            elif block["type"] == "yolo":
                mask = block["mask"].split(",")
                mask = [int(x) for x in mask]
                
                anchors = block["anchors"].split(",")
                anchors = [float(a.strip()) for a in anchors]
                anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]
                
                num_classes = int(block["classes"])
                img_height = int(net_info["height"])
                
                yolo = YOLOLayer(anchors, num_classes, img_height)
                module.add_module(f"yolo_{index}", yolo)
            
            module_list.append(module)
            prev_filters = filters
            output_filters.append(filters)
        
        return net_info, module_list
    
    def forward(self, x):
        outputs = {}
        detections = None
        
        for i, module in enumerate(self.module_list):
            block = self.blocks[i+1]
            
            if block["type"] in ["convolutional", "upsample"]:
                x = module(x)
            
            elif block["type"] == "route":
                layers = block["layers"].split(',')
                layers = [int(i.strip()) for i in layers]
                
                if len(layers) == 1:
                    x = outputs[layers[0]]
                else:
                    maps = [outputs[i] for i in layers]
                    x = torch.cat(maps, 1)
            
            elif block["type"] == "shortcut":
                from_layer = int(block["from"])
                x = outputs[i-1] + outputs[i+from_layer]
            
            elif block["type"] == "yolo":
                x = module(x)
                if detections is None:
                    detections = x
                else:
                    detections = torch.cat((detections, x), 1)
            
            outputs[i] = x
        
        return detections

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.grid_size = 0
    
    def forward(self, x):
        batch_size = x.size(0)
        grid_size = x.size(2)
        
        prediction = x.view(batch_size, self.num_anchors,
                          self.num_classes + 5, grid_size, grid_size)
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
        
        # Get outputs
        tx = torch.sigmoid(prediction[..., 0])  # Center x
        ty = torch.sigmoid(prediction[..., 1])  # Center y
        tw = prediction[..., 2]  # Width
        th = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred
        
        # Calculate offsets for each grid
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        self.grid_size = grid_size
        g_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).type(FloatTensor)
        g_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w / self.img_dim, a_h / self.img_dim) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
        
        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = tx + g_x
        pred_boxes[..., 1] = ty + g_y
        pred_boxes[..., 2] = torch.exp(tw) * anchor_w
        pred_boxes[..., 3] = torch.exp(th) * anchor_h
        
        output = torch.cat(
            (
                pred_boxes.view(batch_size, -1, 4) * self.img_dim,
                conf.view(batch_size, -1, 1),
                pred_cls.view(batch_size, -1, self.num_classes),
            ),
            -1,
        )
        
        return output 