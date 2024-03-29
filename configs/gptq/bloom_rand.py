import transformers
import torch
from .models_utils import BaseLM, find_layers
from transformers import BloomForCausalLM, AutoTokenizer
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
from .quant import *
from .gptq import GPTQ

import numpy as np 
import pickle


from time import perf_counter

def calculate_hessian_error(err, hess):
    eigvals = torch.linalg.eigvalsh(hess)
    top_eigval = eigvals[-1]
    err_h = err * top_eigval
    return err_h

class BLOOMClass(BaseLM):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size

        self.model = BloomForCausalLM.from_pretrained(self.model_name, torch_dtype='auto')
        self.model.eval()
        self.seqlen = 2048

        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        self.vocab_size = self.tokenizer.vocab_size
        print('BLOOM vocab size: ', self.vocab_size)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 2048
    @property
    def max_gen_toks(self):
        print('max_gen_toks fn')
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0][:, :, :250680]

    @torch.no_grad()
    def _model_logits_on_dataset(self, dataset_inps):
        dataset_logits = []
        nsamples = len(dataset_inps)

        dev = self.device

        model = self.model

        print('Evaluation...')

        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.transformer.h

        model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
        layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = []
        outs = []

        for batch_idx, batch in enumerate(dataset_inps):
            inps.append(torch.zeros(
                (batch.shape[1], self.model.config.hidden_size), dtype=dtype,
            ))
            outs.append(torch.zeros(
                (batch.shape[1], self.model.config.hidden_size), dtype=dtype,
            ))

        cache = {'i': 0, 'attention_masks': [], 'alibis': []}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_masks'].append(kwargs['attention_mask'].detach().cpu())
                cache['alibis'].append(kwargs['alibi'].detach().cpu())
                raise ValueError

        layers[0] = Catcher(layers[0])
        for i in range(nsamples):
            batch = dataset_inps[i].to(dev)
            try:
                model(batch)
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
        torch.cuda.empty_cache()

        attention_masks = cache['attention_masks']
        alibis = cache['alibis']
        

        for i in range(len(layers)):
            layer = layers[i].to(dev)

            if self.args.nearest:
                subset = find_layers(layer)
                for n_idx, name in enumerate(subset):
                    quantizer = Quantizer()
                    rand_bit = self.args.wbits
                    print("rand_bit: ", rand_bit)
                    quantizer.configure(
                        rand_bit, perchannel=True, sym=False, mse=False
                    )
                    W = subset[name].weight.data
                    quantizer.find_params(W, weight=True)
                    subset[name].weight.data = quantize(
                        W, quantizer.scale, quantizer.zero, quantizer.maxq
                    ).to(next(iter(layer.parameters())).dtype)

            for j in range(nsamples):
                outs[j] = layer(inps[j].to(self.device),
                      attention_mask=attention_masks[j].to(self.device),
                      alibi=alibis[j].to(self.device))[0].detach().cpu()

            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            inps, outs = outs, inps

        model.transformer.ln_f = model.transformer.ln_f.to(dev)
        model.lm_head = model.lm_head.to(dev)

        for i in tqdm(range(nsamples), desc='Last Layer'):
            hidden_states = inps[i].unsqueeze(0).to(self.device)
            hidden_states = self.model.transformer.ln_f(hidden_states)
            batch_logits = F.log_softmax(self.model.lm_head(hidden_states)[0][:, :, :250680], dim=-1).cpu()
            dataset_logits.append(batch_logits)

        model.config.use_cache = use_cache
        return dataset_logits

    @torch.no_grad()
    def _model_logits_on_dataset2(self, dataset_inps):
        dataset_logits = []
        nbatches = len(dataset_inps)

        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        layers = self.model.transformer.h

        self.model.transformer.word_embeddings = self.model.transformer.word_embeddings.to(self.device)
        self.model.transformer.word_embeddings_layernorm = self.model.transformer.word_embeddings_layernorm.to(
            self.device)
        layers[0] = layers[0].to(self.device)

        dtype = next(iter(self.model.parameters())).dtype


        inps = []
        outs = []
        for batch_idx, batch in enumerate(dataset_inps):
            inps.append(torch.zeros(
                (batch.shape[1], self.model.config.hidden_size), dtype=dtype,
            ))
            outs.append(torch.zeros(
                (batch.shape[1], self.model.config.hidden_size), dtype=dtype,
            ))

        cache = {'i': 0, 'attention_masks': [], 'alibi': []}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp.cpu()
                cache['i'] += 1
                cache['attention_masks'].append(kwargs['attention_mask'].detach().cpu())
                cache['alibi'].append(kwargs['alibi'].detach().cpu())
                raise ValueError

        layers[0] = Catcher(layers[0])
        for i in range(nbatches):
            batch = dataset_inps[i].to(self.device)
            try:
                self.model(batch)
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        self.model.transformer.word_embeddings = self.model.transformer.word_embeddings.cpu()
        self.model.transformer.word_embeddings_layernorm = self.model.transformer.word_embeddings_layernorm.cpu()
        torch.cuda.empty_cache()  # TODO: maybe we don't need this?

        attention_masks = cache['attention_masks']
        alibis = cache['alibi']

        for i in range(len(layers)):
            print('layer: ', i)
            layer = layers[i].to(self.device)

            if self.args.wbits < 32 and self.args.nearest:
                subset = find_layers(layer)
                for name in subset:
                    if 'lm_head' in name:
                        continue
                    quantizer = Quantizer()
                    quantizer.configure(
                        self.args.wbits,
                        perchannel=True, sym=False, mse=False, norm=2.4
                    )
                    W = subset[name].weight.data
                    quantizer.find_params(W, weight=True)
                    subset[name].weight.data = quantize(
                        W, quantizer.scale, quantizer.zero, quantizer.maxq
                    ).to(next(iter(layer.parameters())).dtype)


            for j in range(nbatches):
                outs[j] = layer(inps[j].to(self.device),
                                attention_mask=attention_masks[j].to(self.device),
                                alibi=alibis[j].to(self.device))[0].detach().cpu()
            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            inps, outs = outs, inps

        self.model.transformer.ln_f = self.model.transformer.ln_f.to(self.device)
        self.model.lm_head = self.model.lm_head.to(self.device)

        for i in tqdm(range(nbatches), desc='Last Layer'):
            hidden_states = inps[i].unsqueeze(0).to(self.device)
            hidden_states = self.model.transformer.ln_f(hidden_states)
            batch_logits = F.log_softmax(self.model.lm_head(hidden_states)[0][:, :, :250680], dim=-1).cpu()
            dataset_logits.append(batch_logits)

        return dataset_logits

    def _model_logits_on_dataset_2(self, inps):
        # import pdb;pdb.set_trace()
        self.model = self.model.to(self.device)
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu() # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits


    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )

    @torch.no_grad()
    def bloom_sequential(self, dataloader):
        collected_information = {}
        print('Starting ...')
        profile_start_time = perf_counter()
        if self.args.profile:
            self.bloom_profile(dataloader)
            exit()

        model = self.model
        dev = self.device

        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.transformer.h

        model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
        layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (self.args.nsamples, self.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )
        cache = {'i': 0, 'attention_mask': None, 'alibi': None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                cache['alibi'] = kwargs['alibi']
                raise ValueError

        layers[0] = Catcher(layers[0])
        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        alibi = cache['alibi']

        print('Ready.')

        if self.args.ada_file is not None:
            file_name = self.args.ada_file
            with open(file_name, 'rb') as f:
                bit_assignment = pickle.load(f)
            # check numbers
            assert len(bit_assignment) == len(layers), "bit assignment length is not equal to layer length"
        else:
            bit_assignment = None 

        for i in range(len(layers)):
            layer = layers[i].to(dev)

            subset = find_layers(layer)
            gptq = {}
            if bit_assignment is not None:
                bit_for_layer = bit_assignment[i]
            for n_idx, name in enumerate(subset):
                gptq[name] = GPTQ(subset[name])
                rand_bit = self.args.wbits
                if self.args.rand_bit:
                    if rand_bit == 4:
                        rand_bit = np.random.choice([4, 8])
                    elif rand_bit == 3:
                        rand_bit = np.random.choice([3, 4])
                if bit_assignment is not None:
                    if 'qproj' in name or 'k_proj' in name or 'v_proj' in name or 'out' in name:
                        rand_bit = bit_for_layer[0]
                    else:
                        rand_bit = bit_for_layer[1]
                    print("use rand bit", rand_bit)
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    rand_bit, perchannel=True, sym=False, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(self.args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')

                (err, Hessian) = gptq[name].fasterquant(percdamp=self.args.percdamp, groupsize=self.args.groupsize)
                hess_err = calculate_hessian_error(err, Hessian)
                if not self.args.rand_bit and bit_assignment is None:
                    collected_information[(i, name)] = hess_err.detach().cpu().numpy()
                gptq[name].free()
            for j in range(self.args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]

            layers[i] = layer.cpu()
            del layer
            del gptq
            torch.cuda.empty_cache()

            inps, outs = outs, inps

        model.config.use_cache = use_cache
        profile_end_time = perf_counter()
        duration = profile_end_time - profile_start_time
        if not self.args.rand_bit and bit_assignment is None:
            file_path = self.args.prof_file
            model = self.args.model
            dataset = self.args.dataset
            model = model.replace('/', '_')
            file_path = f'{model}_{dataset}_hess_stat_{self.args.wbits}.pkl'
            profile_end_time = perf_counter()
            duration = profile_end_time - profile_start_time
            collected_information['duration'] = duration
            print(f'Profiled in {duration:.2f} seconds.')
            # store the collected information
            with open(file_path, 'wb') as f: pickle.dump(collected_information, f)
        exit()

    @torch.no_grad()
    def bloom_profile(self, dataloader):
        print('Starting ... Profile')

        profile_start_time = perf_counter()
        model = self.model
        dev = self.device

        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.transformer.h

        model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
        layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (self.args.nsamples, self.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )
        cache = {'i': 0, 'attention_mask': None, 'alibi': None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                cache['alibi'] = kwargs['alibi']
                raise ValueError

        layers[0] = Catcher(layers[0])
        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        alibi = cache['alibi']

        print('Ready.')

        if self.args.ada_file is not None:
            file_name = self.args.ada_file
            with open(file_name, 'rb') as f:
                bit_assignment = pickle.load(f)
            # check numbers
            assert len(bit_assignment) == len(layers), "bit assignment length is not equal to layer length"
        else:
            bit_assignment = None 

        collected_information = {} # xmax, xmin, wmax, wmin

        for i in range(len(layers)):
            layer = layers[i].to(dev)

            subset = find_layers(layer)
            gptq = {}
            if bit_assignment is not None:
                bit_for_layer = bit_assignment[i]

            def collect_in_w_scale(module, inp, out):
                # the interested information
                # for weight
                weight = module.weight.data
                wmax = weight.float().max(dim=-1, keepdim=True).values.detach().cpu().numpy()
                wmin = weight.float().min(dim=-1, keepdim=True).values.detach().cpu().numpy()
                # for input
                # per-channel
                # deterministic, only variance are concerned
                # inp_max = inp[0].float().max(dim=-1, keepdim=True).values.detach().cpu().numpy()
                # inp_min = inp[0].float().min(dim=-1, keepdim=True).values.detach().cpu().numpy()
                inp_var = inp[0].float().var(dim=-1, keepdim=True).cpu().numpy()

                if module.layer_idx not in collected_information:
                    collected_information[module.layer_idx] = {}
                if module.name not in collected_information:
                    collected_information[module.layer_idx][module.name] = {'x_var': inp_var, 'wmax': wmax, 'wmin': wmin}
                else:
                    # running average the inp_max and inp_min
                    collected_information[module.layer_idx][module.name]['x_var'] += inp_var

            handles = []
            for name in subset:
                subset[name].name = name
                subset[name].layer_idx = i
                handles.append(subset[name].register_forward_hook(collect_in_w_scale))


            for j in range(self.args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
            for h in handles:
                h.remove()
            
            for name in subset:
                collected_information[i][name]['x_var'] /= self.args.nsamples

            # for name in subset:
            #     print(i, name)
            #     print('Quantizing ...')
            #     gptq[name].fasterquant(percdamp=self.args.percdamp, groupsize=self.args.groupsize)
            # for j in range(self.args.nsamples):
            #     outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]

            layers[i] = layer.cpu()
            del gptq
            torch.cuda.empty_cache()

            inps, outs = outs, inps

        file_path = self.args.prof_file
        model = self.args.model
        dataset = self.args.dataset
        model = model.replace('/', '_')
        file_path = f'{model}_{dataset}_stat.pkl'
        profile_end_time = perf_counter()
        duration = profile_end_time - profile_start_time
        collected_information['duration'] = duration
        print(f'Profiled in {duration:.2f} seconds.')
        # store the collected information
        with open(file_path, 'wb') as f: pickle.dump(collected_information, f)


# for backwards compatibility
BLOOM = BLOOMClass