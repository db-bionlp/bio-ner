import copy, os
import json
import logging
from torch.utils.data import TensorDataset
from tqdm import tqdm
import math
logger = logging.getLogger(__name__)
import sys
sys.path.append('../')
from ddi_task.utils import *
from transformers import BertTokenizer, BasicTokenizer

class InputFeatures(object):
    def __init__(self, guid, input_ids, attention_mask, token_type_ids, label_id,
                ):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def read_csv(file, mode,args):
    print("Reading {} file ...".format(mode))

    if mode == 'train':
        file = os.path.join(file, 'train.tsv')
    if mode == 'dev':
        file = os.path.join(file, 'devel.tsv')
    if mode == 'test':
        file = os.path.join(file, 'test.tsv')

    examples = open(file, 'r', encoding='utf-8')

    samples = examples.read().strip().split('\n\n')

    sent_contents = []
    sent_lables = []
    labels = get_label(args)

    count = 0
    for sample in tqdm(samples, desc="Get contents and labels for {} features...".format(mode), ncols=100):
        word_content = []
        label_content = []
        for word_label in sample.split('\n'):
            count+=1
            word, label = word_label.split('\t')
            word_content .append(word)
            label_content.append(labels.index(label))
        sent_contents.append(word_content)
        sent_lables.append(label_content)
    print('number_of_{} set: {}'.format(mode, count))

    return sent_contents, sent_lables

def load_tokenize(args):
    # ADDITIONAL_SPECIAL_TOKENS = []
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    # tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})

    # tokenizer = BasicTokenizer(do_lower_case=True)
    return tokenizer

def makeFeatures(args,sent_list,sent_labels, mode):
    index = 0
    maxlength = 128
    print('Making {} Features'.format(mode))
    features = []
    tokenizer = load_tokenize(args)

    for sent, label, in zip(sent_list, sent_labels):
        index += 1
        tokens =sent

        if len(tokens) > maxlength-2:
            tokens = tokens[:maxlength]
        tokens = tokens
        if len(label) > maxlength-2:
            label = label[:maxlength]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        token_type = [0] * len(tokens)
        attention_mask = [1] * len(tokens)

        # Zero-pad up to the sequence length.
        padding_length = maxlength - len(tokens)

        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type = token_type + ([0] * padding_length)

        pad = len(get_label(args))-1
        label = label + ([pad] * padding_length)

        features.append(
            InputFeatures(guid=index-1,
                          input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type,
                          label_id=label,
                          ))
        if index < 4:
                logging.info("*** Example ***")
                logging.info("guid: %d", index-1)
                logging.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logging.info("attention_mask: %s", " ".join([str(x) for x in attention_mask]))
                logging.info("token_type_ids: %s", " ".join([str(x) for x in token_type]))
                logging.info("label: %s", " ".join([str(x) for x in label]))

    return features

def load_and_cache_examples(args, mode):

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}".format(
                mode,
                list(filter(None, args.model_name_or_path.split("/"))).pop(), ),
        )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)

    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)

        if mode == "train":
            sent_contents, sent_lables = read_csv(args.filename, mode,args)
            features = makeFeatures(args, sent_contents, sent_lables, mode)
        elif mode == "dev":
            sent_contents, sent_lables = read_csv(args.filename, mode,args)
            features = makeFeatures(args, sent_contents, sent_lables, mode)
        elif mode == "test":
            sent_contents, sent_lables = read_csv(args.filename, mode,args)
            features = makeFeatures(args, sent_contents, sent_lables, mode)
        else:
            raise Exception("For mode, Only train,dev,test is available")

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_guid = torch.tensor([f.guid for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_guid, all_input_ids, all_attention_mask,all_token_type_ids, all_label_ids,
                            )
    return dataset




