import os

import torch
from transformers import AutoModel, AutoTokenizer


MODEL_DIR = os.environ.get('MODEL_DIR', './models')

QUERY_INSTRUCTION = 'Represent this query to search a JSON object, the query is about some fields of the object.'
QUERY_PREFIX_JSON = f'Instruct: {QUERY_INSTRUCTION} \nQuery: '
DOC_INSTRUCTION = 'Represent the JSON object for retrieval'


def prepare_gte(model_path=MODEL_DIR + '/gte-modernbert-base'):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).eval().half()
    if torch.cuda.is_available():
        model = model.cuda()

    def _embed(batch, is_query=True):
        batch_dict = tokenizer(batch, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**batch_dict.to('cuda'))
        embeddings = outputs.last_hidden_state[:, 0].cpu()
        return embeddings

    return _embed


def prepare_drama(model_path=MODEL_DIR + '/drama-1b'):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().half()
    if torch.cuda.is_available():
        model = model.cuda()

    def _embed(batch, is_query=True):
        if is_query:
            embeddings = model.encode_queries(tokenizer, batch)
        else:
            embeddings = model.encode_documents(tokenizer, batch)
        embeddings = embeddings.cpu()
        return embeddings

    return _embed


def last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def prepare_qwen(model_path=MODEL_DIR + '/gte-Qwen2-7B-instuct'):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().half()
    if torch.cuda.is_available():
        model = model.cuda()

    def _embed(batch, is_query=True):
        if is_query:
            batch = [QUERY_PREFIX_JSON + t for t in batch]
        batch_dict = tokenizer(batch, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**batch_dict.to('cuda'))
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
        return embeddings

    return _embed


def prepare_e5mistral(model_path=MODEL_DIR + '/e5-mistral-7b-instruct'):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).eval().half()
    if torch.cuda.is_available():
        model = model.cuda()
    max_length = 4096

    def _embed(batch, is_query=True):
        if is_query:
            batch = [QUERY_PREFIX_JSON + t for t in batch]
        batch_dict = tokenizer(batch, max_length=4096, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**batch_dict.to('cuda'))
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
        return embeddings

    return _embed


def prepare_gritlm(model_path=MODEL_DIR + '/GritLM-7B'):
    from gritlm import GritLM

    model = GritLM(model_path, mode='embedding', torch_dtype='auto').eval()
    if torch.cuda.is_available():
        model = model.cuda()
    query_instruction = "<|user|>\nRepresent this query to search a JSON object, the query is about some fields of the object.\n<|embed|>\n"
    doc_instruction = "<|embed|>\n"

    def _embed(batch, is_query=True):
        instruction = query_instruction if is_query else doc_instruction
        embeddings = model.encode(batch, instruction=instruction, convert_to_tensor=True).cpu()
        return embeddings

    return _embed


def prepare_reasonir(model_path=MODEL_DIR + '/ReasonIR-8B'):
    model = AutoModel.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    def _embed(batch, is_query=True):
        instruction = ""
        embeddings = model.encode(batch, instruction=instruction, convert_to_tensor=True).cpu()
        return embeddings

    return _embed


def prepare_nvembedv2(model_path=MODEL_DIR + '/NV-Embed-v2'):
    # TODO: pip install transformers==4.42.4
    from transformers import AutoConfig

    # load model with tokenizer
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.text_config._name_or_path = model_path
    model = AutoModel.from_pretrained(model_path, config=config, torch_dtype="float16", trust_remote_code=True).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    max_length = 8192  # 32768
    query_prefix = QUERY_PREFIX_JSON
    doc_prefix = ""

    def _embed(batch, is_query=True):
        instruction = query_prefix if is_query else doc_prefix
        embeddings = model.encode(batch, instruction=instruction, max_length=max_length).cpu()
        return embeddings

    return _embed


def prepare_instructor(model_path=MODEL_DIR + '/instructor-large'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = InstructOR(model_path).to(device).eval()
    model.transformer.half()

    def _embed(batch, is_query=True):
        instruction = QUERY_INSTRUCTION if is_query else DOC_INSTRUCTION
        batch = [[instruction, t] for t in batch]
        embeddings = model.encode(batch, device=device).cpu()
        return embeddings

    return _embed


def prepare_bge(model_path=MODEL_DIR + '/bge-large-en-v1.5'):
    from sentence_transformers import SentenceTransformer

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_path, device=device).half().eval()

    def _embed(batch, is_query=True):
        if is_query:
            batch = [f'{QUERY_INSTRUCTION} {t}' for t in batch]
        embeddings = model.encode(batch, convert_to_tensor=True).cpu()
        return embeddings

    return _embed


def prepare_nomic2(model_path=MODEL_DIR + '/nomic-embed-text-v2-moe'):
    # pip install torch transformers einops git+https://github.com/nomic-ai/megablocks.git

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model = model.half().to(device).eval()

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _embed(batch, is_query=True):
        instruction = 'search_query: ' if is_query else 'search_document: '
        batch = [instruction + t for t in batch]
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        model_output = model(**encoded_input.to(device))
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings

    return _embed


def prepare_jina3(model_path=MODEL_DIR + '/jina-embeddings-v3'):
    # pip install torch transformers einops
    # pip install flash-attn --no-build-isolation

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = model.to(device).eval()

    def _embed(batch, is_query=True):
        task = 'retrieval.query' if is_query else 'retrieval.passage'
        embeddings = model.encode(batch, task=task, device=device, convert_to_tensor=True)
        return embeddings

    return _embed


def prepare_my(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'
    model = AutoModel.from_pretrained(model_path).eval().half()
    if torch.cuda.is_available():
        model = model.cuda()
    eod_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')

    # 在最后添加一个eos
    def _tokenize(to_tokenize, max_length=512):
        out = tokenizer(
            to_tokenize, padding=False, truncation='longest_first',
            max_length=max_length
        )
        for seq, att in zip(out["input_ids"], out["attention_mask"]):
            seq.append(eod_id)
            att.append(1)
        return tokenizer.pad(out, padding=True, return_tensors="pt")

    # 取最后一个hidden_state
    def last_token_pool(last_hidden_state, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        assert left_padding, "我在训练的时候使用了left_padding，最好保持一致"
        embeddings = last_hidden_state[:, -1]
        # # 我在训练的时候做了normalize
        # embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def _embed(batch, is_query=True):
        # 没添加instruction
        batch_dict = _tokenize(batch)
        with torch.no_grad():
            outputs = model(**batch_dict.to('cuda'))
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
        return embeddings

    return _embed


def prepare_myqwen3(model_path):
    from .qwen3embed import GTEEncoder

    model = GTEEncoder(model_path).eval()

    def _embed(batch, is_query=True):
        # 没添加instruction
        outputs = model(batch, is_query=is_query, return_sparse=False)
        return outputs['dense_embeddings']

    return _embed


######


class InstructOR(torch.nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        from transformers import T5EncoderModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.transformer = T5EncoderModel.from_pretrained(model_path)
        state = torch.load(model_path + '/2_Dense/pytorch_model.bin', map_location='cpu')
        out_features, in_features = state['linear.weight'].size()
        self.linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.linear.weight.data = state['linear.weight'].data
        self.max_seq_length = self.tokenizer.model_max_length
        self.do_lower_case = False

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask * kwargs["instruction_mask"]
        token_embeddings = out.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        embed = sum_embeddings / sum_mask
        embed = torch.nn.functional.normalize(self.linear(embed), p=2, dim=1)
        return embed

    def encode(self, sentences_batch, device):
        features = self.tokenize(sentences_batch)
        features = {k: v.to(device) for k, v in features.items()}
        embed = self.forward(**features)
        return embed

    def tokenize(self, texts):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
            to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

            # Lowercase
            if self.do_lower_case:
                to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

            input_features = self.tokenizer(
                *to_tokenize,
                padding="max_length",
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.max_seq_length,
            )

        elif isinstance(texts[0], list):
            assert isinstance(texts[0][1], str)
            assert (
                len(texts[0]) == 2
            ), "The input should have both instruction and input text"

            instructions = []
            instruction_prepended_input_texts = []
            for pair in texts:
                instruction = pair[0].strip()
                text = pair[1].strip()
                if self.do_lower_case:
                    instruction = instruction.lower()
                    text = text.lower()
                instructions.append(instruction)
                instruction_prepended_input_texts.append("".join([instruction, text]))

            input_features = self.tokenize(instruction_prepended_input_texts)
            instruction_features = self.tokenize(instructions)
            input_features = self.prepare_input_features(
                input_features, instruction_features
            )
        else:
            raise ValueError("not support other modes")

        output.update(input_features)
        return output

    @staticmethod
    def prepare_input_features(
        input_features, instruction_features, return_data_type: str = "pt"
    ):
        if return_data_type == "np":
            input_features["attention_mask"] = torch.from_numpy(
                input_features["attention_mask"]
            )
            instruction_features["attention_mask"] = torch.from_numpy(
                instruction_features["attention_mask"]
            )

        input_attention_mask_shape = input_features["attention_mask"].shape
        instruction_attention_mask = instruction_features["attention_mask"]

        # reducing the attention length by 1 in order to omit the attention corresponding to the end_token
        instruction_attention_mask = instruction_attention_mask[:, 1:]

        # creating instruction attention matrix equivalent to the size of the input attention matrix
        expanded_instruction_attention_mask = torch.zeros(
            input_attention_mask_shape, dtype=torch.int64
        )
        # assigning the the actual instruction attention matrix to the expanded_instruction_attention_mask
        # eg:
        # instruction_attention_mask: 3x3
        #  [[1,1,1],
        #   [1,1,0],
        #   [1,0,0]]
        # expanded_instruction_attention_mask: 3x4
        #  [[1,1,1,0],
        #   [1,1,0,0],
        #   [1,0,0,0]]
        expanded_instruction_attention_mask[
            : instruction_attention_mask.size(0), : instruction_attention_mask.size(1)
        ] = instruction_attention_mask

        # In the pooling layer we want to consider only the tokens corresponding to the input text
        # and not the instruction. This is achieved by inverting the
        # attention_mask corresponding to the instruction.
        expanded_instruction_attention_mask = 1 - expanded_instruction_attention_mask
        input_features["instruction_mask"] = expanded_instruction_attention_mask
        if return_data_type == "np":
            input_features["attention_mask"] = input_features["attention_mask"].numpy()
            instruction_features["attention_mask"] = instruction_features[
                "attention_mask"
            ].numpy()
        return input_features
