import torch
from torchvision.transforms.functional import pil_to_tensor
from dataclasses import dataclass
from typing import Optional, Union, Dict
from abc import ABC

from transformers import (LlavaForConditionalGeneration, AutoProcessor, )

from .base_model import Model, BaseConfig
from .processor import ProcessorFunction
from .hf_access_token import ACCESS_TOKEN
from .vcd_utils.vcd_sample import evolve_vcd_sampling
from .vcd_utils.vcd_add_noise import add_diffusion_noise


@dataclass
class LLaVAConfig(BaseConfig):
    name: str = ''
    model_id: str =  ''

@dataclass
class LLaVA15Config(LLaVAConfig):
    name: str = 'vcd_LLaVA-v1.5-7B'
    model_id: str = 'llava-hf/llava-1.5-7b-hf'

def get_llava_vcd_config_from_name(vlm: str) -> LLaVAConfig:
    if '1.5' in vlm:
        return LLaVA15Config()
    else:
        raise NotImplementedError()


def get_llava_vcd(device: torch.device, vlm: Optional[str] = None, config: Optional[LLaVAConfig] = None):
    assert vlm is not None or config is not None
    assert not (vlm is not None and config is not None)

    if vlm is not None:
        config = get_llava_vcd_config_from_name(vlm)

    if isinstance(config, LLaVA15Config):
        return LLaVA15VCD(device, config)
    else:
        raise NotImplementedError()


#Generic Processor for the LLaVA version where chat_template is ACTUALLY correct
class LLaVAVCDProcessorFunction(ProcessorFunction):
    def __init__(self, processor):
        super().__init__(processor)

    def __call__(self, batch, *args, noise_steps=999, **kwargs):
        conversations = []
        for prompt in batch['prompt']:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            text_conversation = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            conversations.append(text_conversation)
        
        images = [img for img in batch['image']]
        images_cd = [self.add_diff_noise(image, noise_steps) for image in images]
        inputs = self.processor(images=images, text=conversations, padding="longest", return_tensors="pt")
        inputs['images_cd'] = images_cd
        return inputs

    def add_diff_noise(self, image, noise_steps=999):
        image = pil_to_tensor(image)/255
        return add_diffusion_noise(image, noise_steps)
    
#################
#LLaVA 1.5 / Next
#################
class LLaVA15VCD(Model):
    def __init__(self, device: torch.device, config: LLaVA15Config):

        evolve_vcd_sampling()

        model = LlavaForConditionalGeneration.from_pretrained(
            config.model_id,
            torch_dtype=torch.float16,
            attn_implementation = "flash_attention_2",
            token=ACCESS_TOKEN
        ).to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(config.model_id)

        for param in model.parameters():
            param.requires_grad = False

        processor_function = LLaVAVCDProcessorFunction(processor)
        super().__init__(model, processor, processor_function, config)


    def generate(self, inputs, *args, **kwargs):

        return super.generate(**inputs, **kwargs)