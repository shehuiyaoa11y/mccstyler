#!/usr/bin/env python3



import argparse
from functools import partial
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models
from torchvision.transforms import functional as TF
from tqdm import trange
import torch.optim as optim

from resnet_unetplusplus import NestedUResnet
from CLIP import clip
from diffusion import get_model, get_models, sampling, utils


import net
from packaging import version
#import StyleNet

import globalloss

import lpips

MODULE_DIR = Path(__file__).resolve().parent



import torch
import torch.nn.functional as F
import clip
from template import imagenet_templates
#import StyleNet
from torchvision import transforms, models


def clip_normalize(image,device):
    image = F.interpolate(image,size=224,mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

source = "a Photo"
device='cuda'
clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]

def clip_loss(prompt,content_image):
    template_text = compose_text_with_templates(prompt, imagenet_templates)
    tokens = clip.tokenize(template_text).to(device)
    text_features = clip_model.encode_text(tokens).detach()
    text_features = text_features.mean(axis=0, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    template_source = compose_text_with_templates(source, imagenet_templates)
    tokens_source = clip.tokenize(template_source).to(device)
    text_source = clip_model.encode_text(tokens_source).detach()
    text_source = text_source.mean(axis=0, keepdim=True)
    text_source /= text_source.norm(dim=-1, keepdim=True)
    source_features = clip_model.encode_image(clip_normalize(content_image, device))
    source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

    cropper = transforms.Compose([
        transforms.RandomCrop(128)
    ])
    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
        transforms.Resize(224)
    ])

    style_net = StyleNet.UNet()
    target = style_net(content_image).to(device)
    img_proc = []
    for n in range(64):
        target_crop = cropper(target)
        target_crop = augment(target_crop)
        img_proc.append(target_crop)

    img_proc = torch.cat(img_proc, dim=0)
    img_aug = img_proc

    image_features = clip_model.encode_image(clip_normalize(img_aug, device))

    image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
    text_direction = (text_features - text_source).repeat(image_features.size(0), 1)
    text_direction /= text_direction.norm(dim=-1, keepdim=True)

    glob_features = clip_model.encode_image(clip_normalize(target, device))
    glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))

    glob_direction = (glob_features - source_features)
    glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)

    lossglobal = (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()
    return lossglobal




class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size  # cut_size=224
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]  # sideY=256,sideX=256 input:256*256


        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutout = F.adaptive_avg_pool2d(cutout, self.cut_size)
            cutouts.append(cutout)
        return torch.cat(cutouts)




def parse_prompt(prompt, default_weight=3.):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', default_weight][len(vals):]
    return vals[0], float(vals[1])

def resize_and_center_crop(image, size):
    fac = max(size[0] / image.size[0], size[1] / image.size[1])
    image = image.resize((int(fac * image.size[0]), int(fac * image.size[1])), Image.LANCZOS)
    return TF.center_crop(image, size[::-1])

def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

    return loss_var_l2


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out





device = 'cuda'
clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)
preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] +

                                             preprocess.transforms[:2] +

                                             preprocess.transforms[4:])

def encode_text( tokens: list) -> torch.Tensor:
    return clip_model.encode_text(tokens)

def encode_images(images: torch.Tensor) -> torch.Tensor:
    images = preprocess(images).to(device)
    return clip_model.encode_image(images)

def get_text_features(text: str, norm: bool = True) -> torch.Tensor:
    tokens = clip.tokenize(text).to(device)

    text_features = encode_text(tokens).detach()

    if norm:
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features

def get_image_features(img: torch.Tensor, norm: bool = True) -> torch.Tensor:
    image_features = encode_images(img)

    if norm:
        image_features /= image_features.clone().norm(dim=-1, keepdim=True)

    return image_features

def txt_to_img_similarity(text, generated_images):
    text_features = get_text_features(text)
    gen_img_features = get_image_features(generated_images)

    return (text_features @ gen_img_features.T).mean()

class PatchNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.similarity_function = self._get_similarity_function()
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def _get_similarity_function(self):

        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        return self._cosine_simililarity

    def _cosine_simililarity(self, x, y):

        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        feat_k = feat_k.detach()
        l_pos = self.cos(feat_q,feat_k)
        l_pos = l_pos.view(batchSize, 1)
        l_neg_curbatch = self.similarity_function(feat_q.view(batchSize,1,-1),feat_k.view(1,batchSize,-1))
        l_neg_curbatch = l_neg_curbatch.view(1,batchSize,-1)

        diagonal = torch.eye(batchSize, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, batchSize)
        out = torch.cat((l_pos, l_neg), dim=1) / 0.07
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('init', type=str,
                   help='the init image')
    p.add_argument('prompts', type=str, default=[], nargs='*',
                   help='the text prompts to use')
    p.add_argument('--images', type=str, default=[], nargs='*', metavar='IMAGE',
                   help='the image prompts')
    p.add_argument('--checkpoint', type=str,
                   help='the checkpoint to use')
    p.add_argument('--device', type=str, 
                   help='the device to use')
    p.add_argument('--max-timestep', '-mt', type=float, default=10.,
                   help='the maximum timestep')
    p.add_argument('--method', type=str, default='iplms',
                   choices=['ddim', 'prk', 'plms', 'pie', 'plms2', 'iplms'],
                   help='the sampling method to use')
    p.add_argument('--model', type=str, default='cc12m_1_cfg', choices=['cc12m_1_cfg'],
                   help='the model to use')
    p.add_argument('--output', '-o', type=str, default='out.png',
                   help='the output filename')
    p.add_argument('--size', type=int, nargs=2,
                   help='the output image size')
    p.add_argument('--steps', type=int, default=50,
                   help='the number of timesteps')

    
    p.add_argument('--cutn', type=int, default=1,
                   help='the number of random crops to use')
    p.add_argument('--cut-pow', type=float, default=1.,
                   help='the random crop size power')
    p.add_argument('--clip-guidance-scale', '-cs', type=float, default=500.,
                   help='the CLIP guidance scale')
    p.add_argument('-n', type=int, default=1,
                   help='the number of images to sample')
    p.add_argument('--checkpoint1', type=str,
                   help='the checkpoint to use')
    p.add_argument('--model1', type=str, default='wikiart_256', choices=get_models(),
                   help='the model to use')
    p.add_argument('--wikiart_scale', '-ws', type=float, default=0.2,  # 0.5
                   help='wikiart_scale')
    p.add_argument('--free_scale', '-fs', type=float, default=0.8,  # 0.5
                   help='cfg_scale')

    p.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')

    p.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
    p.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for G')
    p.add_argument('--no_dropout', type=str2bool, nargs='?', const=True, default=True,
                            help='no dropout for the generator')
    p.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
    p.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    p.add_argument('--no_antialias', action='store_true', help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
    p.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    p.add_argument('--num_patches', type=int, default=512, help='number of patches per layer')
    p.add_argument('--content_nce_layers', type=str, default='1,2,3,4', help='compute NCE loss on which layers')
    # p.add_argument('--content_nce_layers', type=str, default='1,2,3,4', help='compute NCE loss on which layers')
    p.add_argument('-ns', '--nce_scale', type=float, default=1., help='the nce loss scale')

    p.add_argument("-lc", '--lambda_c', type=float, default=3.,
                    help='content loss parameter')

    p.add_argument("-tvs",  "--tv_scale", type=float, help="Smoothness scale", default=80, dest='tv_scale')
    p.add_argument("-is",   "--init_scale", type=int, help="Initial image scale (e.g. 1000)", default=10, dest='init_scale')
    p.add_argument("-as",   "--aes_scale", type=float, default=10., help='aesthetic_loss_scale')

    args = p.parse_args()


    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    vgg = net.vgg

    netAE = net.Mcc_Encoder(vgg, args.gpu_ids).to(device)


    VGG = models.vgg19(pretrained=True).features
    VGG.to(device)

    criterionNCE = []
    for nce_layer in args.content_nce_layers:
        criterionNCE.append(PatchNCELoss().to(device))

    model = get_model(args.model)()
    _, side_y, side_x = model.shape
    if args.size:
        side_x, side_y = args.size
        print(f"side_x={side_x}, side_y={side_y}")

    checkpoint = args.checkpoint  #
    if not checkpoint:
        checkpoint = MODULE_DIR / f'checkpoints/{args.model}.pth'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    if device.type == 'cuda':
        model = model.half()


    model = model.to(device).eval().requires_grad_(False)

    clip_model_name = model.clip_model if hasattr(model, 'clip_model') else 'ViT-B/16'
    clip_model = clip.load(clip_model_name, jit=False, device=device)[0]  # clip

    clip_model.eval().requires_grad_(False)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])



    init = Image.open(utils.fetch(args.init)).convert('RGB')

    init = resize_and_center_crop(init, (side_x, side_y))  # side_x=256,side_y=256

    init = utils.from_pil_image(init).to(device)[None]


    make_cutouts = MakeCutouts(clip_model.visual.input_resolution, args.cutn, args.cut_pow)
    # clip_model.visual.input_resolution=224


    zero_embed = torch.zeros([1, clip_model.visual.output_dim], device=device)
    target_embeds, weights = [zero_embed], []

    target_embeds1, weights1 = [], []
    for prompt in args.prompts:

        txt, weight = parse_prompt(prompt)



        target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights.append(weight)
        target_embeds1.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
        weights1.append(weight) # 3.0

    for prompt in args.images:
        path, weight = parse_prompt(prompt)
        img = Image.open(utils.fetch(path)).convert('RGB')
        img1 = TF.resize(img, min(side_x, side_y, *img.size),
                        transforms.InterpolationMode.LANCZOS)
        clip_size = clip_model.visual.input_resolution
        img = resize_and_center_crop(img, (clip_size, clip_size))
        batch = TF.to_tensor(img)[None].to(device)
        embed = F.normalize(clip_model.encode_image(normalize(batch)).float(), dim=-1)
        target_embeds.append(embed)
        weights.append(weight)

        batch1= make_cutouts(TF.to_tensor(img1)[None].to(device))
        embeds1 = F.normalize(clip_model.encode_image(normalize(batch1)).float(), dim=-1)
        target_embeds1.append(embeds1)
        weights1.extend([weight / args.cutn] * args.cutn)

    weights = torch.tensor([1 - sum(weights), *weights], device=device)


    target_embeds1 = torch.cat(target_embeds1)
    # type(target_embeds1) tensor
    # target_embeds1.shape torch.Size([1, 512])
    weights1 = torch.tensor(weights1, device=device)  # tensor([3.], device='cuda:0')
    if weights1.sum().abs() < 1e-3:
        raise RuntimeError('The weights must not sum to 0.')
    weights1 /= weights1.sum().abs()  # weights1是tensor([1.], device='cuda:0')


    model1 = get_model(args.model1)()

    _, side_y, side_x = model1.shape

    checkpoint1 = args.checkpoint1
    if not checkpoint1:
        checkpoint1 = MODULE_DIR / f'checkpoints/{args.model1}.pth'
    model1.load_state_dict(torch.load(checkpoint1, map_location='cpu'))

    if device.type == 'cuda':
        model1 = model1.half()
    model1 = model1.to(device).eval().requires_grad_(False)

    make_cutouts = MakeCutouts(clip_model.visual.input_resolution, args.cutn, args.cut_pow)

    aesthetic_model_16 = torch.nn.Linear(512,1).cuda()

    aesthetic_model_16.load_state_dict(torch.load("./checkpoints/ava_vit_b_16_linear.pth"))

    def calculate_loss(src, tgt):
        content_nce_layers = [int(i) for i in args.content_nce_layers.split(',')] # args.content_nce_layers='1,2,3,4'
        # content_nce_layers = [0, 3, 7, 11, 15, 19, 23, 27, 31]

        n_layers = len(content_nce_layers)
        feat_q, feat_k,loss_cc,loss_ccp,loss_cc2 = netAE(src, tgt, encoded_only=True)

        return loss_cc,loss_ccp,loss_cc2

    def load_image2(img_path, img_height=None, img_width=None):

        image = Image.open(img_path)
        if img_width is not None:
            image = image.resize((img_width, img_height))

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        image = transform(image)[:3, :, :].unsqueeze(0)

        return image

    def img_normalize(image):
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)

        image = (image - mean) / std
        return image

    def get_features(image, model, layers=None):

        if layers is None:
            layers = {'0': 'conv1_1',
                      '5': 'conv2_1',
                      '10': 'conv3_1',
                      '19': 'conv4_1',
                      '21': 'conv4_2',
                      '28': 'conv5_1',
                      '31': 'conv5_2'
                      }
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features

    style_net = NestedUResnet()
    style_net = style_net.half()
    style_net.to(device)



    def cond_fn(x, t, pred,txt):
        clip_embed = F.normalize(target_embeds1.mul(weights1[:, None]).sum(0, keepdim=True), dim=-1)


        clip_embed = clip_embed.repeat([args.n, 1])


        if min(pred.shape[2:4]) < 256:
            pred = F.interpolate(pred, scale_factor=2, mode='bilinear', align_corners=False)
            # pred.shape: torch.Size([1, 3, 256, 256])
        clip_in = normalize(make_cutouts((pred + 1) / 2))
        # clip_in.shape torch.Size([1, 3, 224, 224])

        image_embeds = clip_model.encode_image(clip_in).view([args.cutn, x.shape[0], -1])
        # image_embeds.shape torch.Size([1, 1, 512])
        new_image_embeds = image_embeds.squeeze(dim=1)
        content_image = load_image2(args.init, 256, 256)


        target = style_net(content_image).to(device)
        target.requires_grad_(True)

        content_image = content_image.to(device)
        content_features = get_features(img_normalize(content_image), VGG)
        # target =
        target_features = get_features(img_normalize(target), VGG)
        content_loss = 0
        content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

        tv_losses = tv_loss(pred)


        tokens = clip.tokenize(txt).to(device)  # #
        text_features = clip_model.encode_text(tokens).detach()  # #
        text_features = text_features.mean(axis=0, keepdim=True)  # #
        text_features /= text_features.norm(dim=-1, keepdim=True)  # #

        loss_patch1 = 0
        loss_patch2 = 0

        image_features = clip_model.encode_image(clip_normalize(content_image, device))  # #
        image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
        pred_features = clip_model.encode_image(clip_normalize(pred, device))  # #
        pred_features /= (image_features.clone().norm(dim=-1, keepdim=True))
        loss_temp = (1 - torch.cosine_similarity(text_features, pred_features, dim=1))  # #
        loss_temp[loss_temp < 0.7] = 0
        loss_patch2 += loss_temp.mean()

        loss_temp1 = (1 - torch.cosine_similarity(image_features, pred_features, dim=1))  # #
        loss_temp[loss_temp < 0.7] = 0
        loss_patch1 += loss_temp1.mean()
        loss_patch = 20 * loss_patch1 + loss_patch2

        lossglobal = globalloss.clip_loss(txt, x, target)



        # image_embeds.shape torch.Size([1, 1, 512]) clip_embed.shape   torch.Size([1, 512]
        loss_cc, _,loss_cc2 = calculate_loss(init.detach().to(device), pred.detach().to(device))
        # init.detach().to(device).shape: torch.Size([1, 3, 256, 256])
        # pred.detach().to(device).shape: torch.Size([1, 3, 256, 256])

        l1_loss = torch.nn.SmoothL1Loss()
        loss_c = loss_cc + loss_cc2



        total_loss = tv_losses.sum() * 80 + 2.0*loss_c  + loss_patch  + 400*lossglobal


        grad = -torch.autograd.grad(total_loss, x)[0]

        return grad


    def cfg_model_fn(x, t):
        for prompt in args.prompts:

            txt, _ = parse_prompt(prompt)  # txt是文本

        n = x.shape[0]
        n_conds = len(target_embeds)
        x_in = x.repeat([n_conds, 1, 1, 1])
        t_in = t.repeat([n_conds])
        clip_embed_in = torch.cat([*target_embeds]).repeat_interleave(n, 0)
        vs = model(x_in, t_in, clip_embed_in).view([n_conds, n, *x.shape[1:]])

        v = vs.mul(weights[:, None, None, None, None]).sum(0)

        extra_args = {}

        with torch.enable_grad():
            x = x.detach().requires_grad_()

            alphas, sigmas = utils.t_to_alpha_sigma(t)
            pred = x * alphas[:, None, None, None] - v * sigmas[:, None, None, None]
            cond_grad = cond_fn(x, t, pred,txt, **extra_args).detach()
            v = v.detach() - cond_grad * (sigmas[:, None, None, None] / alphas[:, None, None, None])

        return v

    def run():
        t = torch.linspace(0, 1, args.steps + 1, device=device)

        steps = utils.get_spliced_ddpm_cosine_schedule(t)
        steps = steps[steps <= args.max_timestep]
        if args.method == 'ddim':
            x = sampling.reverse_sample(model, init, steps, {'clip_embed': zero_embed})
            out = sampling.sample(cfg_model_fn, x, steps.flip(0)[:-1], 0, {})
        if args.method == 'prk':
            x = sampling.prk_sample(model, init, steps, {'clip_embed': zero_embed}, is_reverse=True)
            out = sampling.prk_sample(cfg_model_fn, x, steps.flip(0)[:-1], {})
        if args.method == 'plms':
            x = sampling.plms_sample(model, init, steps, {'clip_embed': zero_embed}, is_reverse=True)
            out = sampling.plms_sample(cfg_model_fn, x, steps.flip(0)[:-1], {})
            out1 = sampling.plms_sample(cfg_model_fn, x, steps.flip(0)[:-3], {})
            out2 = sampling.plms_sample(cfg_model_fn, x, steps.flip(20)[:-1], {})
            out3 = sampling.plms_sample(cfg_model_fn, x, steps.flip(30)[:-1], {})
            out4 = sampling.plms_sample(cfg_model_fn, x, steps.flip(40)[:-1], {})
            out5 = sampling.plms_sample(cfg_model_fn, x, steps.flip(50)[:-1], {})
        if args.method == 'pie':
            x = sampling.pie_sample(model, init, steps, {'clip_embed': zero_embed}, is_reverse=True)
            out = sampling.pie_sample(cfg_model_fn, x, steps.flip(0)[:-1], {})
        if args.method == 'plms2':
            x = sampling.plms2_sample(model, init, steps, {'clip_embed': zero_embed}, is_reverse=True)
            out = sampling.plms2_sample(cfg_model_fn, x, steps.flip(0)[:-1], {})
        if args.method == 'iplms':
            x = sampling.iplms_sample(model, init, steps, {'clip_embed': zero_embed}, is_reverse=True)

            out = sampling.iplms_sample(cfg_model_fn, x, steps.flip(0)[:-1], {})
        utils.to_pil_image(out[0]).save(args.output)

    try:
        run()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
