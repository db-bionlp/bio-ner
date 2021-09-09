import sys
import os
import json
import copy
from itertools import chain
sys.path.append('../')
sys.path.append('./')
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from ddi_task.utils import *
from torch.utils.data.distributed import DistributedSampler
from ddi_task.load_data import load_tokenize
logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        label_length = get_label(args)
        self.num_labels = len(label_length)

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        self.config_class, self.model_class,_ = MODEL_CLASSES[args.model_type]
        self.config = self.config_class.from_pretrained(args.model_name_or_path, num_labels=self.num_labels)
        self.model = self.model_class.from_pretrained(MODEL_PATH_MAP[args.model_type], config=self.config, args=self.args)

        if self.args.model_type == "scibert" :
            len_new_tokenizer = len(load_tokenize(args))
            self.model.resize_token_embeddings(len_new_tokenizer)

        # Setup CUDA, GPU & distributed training
        if args.local_rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.args.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            self.args.n_gpu = 1
        self.device = device

        print(
              "#  Number_of_GPU : {}  #\n".format(self.args.n_gpu)
              )

        if args.tpu:
            if args.tpu_ip_address:
                    os.environ["TPU_IP_ADDRESS"] = args.tpu_ip_address
            if args.tpu_name:
                os.environ["TPU_NAME"] = args.tpu_name
            if args.xrt_tpu_config:
                os.environ["XRT_TPU_CONFIG"] = args.xrt_tpu_config

            assert "TPU_IP_ADDRESS" in os.environ
            assert "TPU_NAME" in os.environ
            assert "XRT_TPU_CONFIG" in os.environ

            import torch_xla.core.xla_model as xm
            args.device = xm.xla_device()
            args.xla_model = xm

        if self.args.local_rank == 0:
            torch.distributed.barrier()

        self.model.to(self.device)

    def train(self):
        self.args.train_batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)
        train_sampler = RandomSampler(self.train_dataset) if self.args.local_rank == -1 else DistributedSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        if self.args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.local_rank],
                                                              output_device=self.args.local_rank,
                                                              find_unused_parameters=True)
        if self.args.parameter_averaging:
            storage_model = copy.deepcopy(self.model)
            storage_model.zero_init_params()
        else:
            storage_model = None

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.model.zero_grad()
        optimizer.zero_grad()
        global_step = 0
        tr_loss = 0.0

        # train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        set_seed(self.args)

        epochs = int(self.args.num_train_epochs)

        for i in range(epochs):
            print("[Epoch] : {}".format(i+1))

            epoch_iterator = tqdm(train_dataloader, desc="Iteration",ncols=100)
            for step, batch in enumerate(epoch_iterator):

                self.model.train()

                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'guid': batch[0],
                          'input_ids': batch[1],
                          'attention_mask': batch[2],
                          'token_type_ids': batch[3],
                          'labels': batch[4],
                          }
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                if self.args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0 and not self.args.tpu:
                    if self.args.fp16:
                       torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                       torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    if not self.args.parameter_averaging:
                        scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    # if self.args.local_rank in [-1, 0] and self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    #     if self.args.local_rank == -1:
                    #         self.evaluate('dev')

                    if self.args.local_rank in [-1, 0] and self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model(i+1)

                if self.args.tpu:
                    self.args.xla_model.optimizer_step(optimizer, barrier=True)
                    self.model.zero_grad()
                    global_step += 1

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

                if self.args.parameter_averaging:
                    storage_model.accumulate_params(self.model)

            self.evaluate('dev')

            self.predict('test')
        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev set is available")

        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset) if self.args.local_rank == -1 else DistributedSampler(self.eval_dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("\n***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating", ncols=100):

            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():

                inputs = {'guid': batch[0],
                          'input_ids': batch[1],
                          'attention_mask': batch[2],
                          'token_type_ids': batch[3],
                          'labels': batch[4],
                          }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                out_attention_mask = inputs['attention_mask'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                out_attention_mask = np.append(
                    out_attention_mask, inputs['attention_mask'].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=2)

        final_preds=[]
        final_label_ids = []

        for index, attention_mask in enumerate(out_attention_mask):
            length = 0
            for mask in attention_mask:
                if mask == 1:
                    length += 1
                else:
                    break

            final_label_ids.append(out_label_ids[index][:length])
            final_preds.append(preds[index][:length])

        preds = np.array(list(chain.from_iterable(final_preds)))
        out_label_ids = np.array(list(chain.from_iterable(final_label_ids)))

        print("length of dev : {}".format(preds.shape[0]))

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        if not os.path.exists(self.args.model_dir):
            os.mkdir(self.args.model_dir)

        eval_out_file = os.path.join(self.args.model_dir, 'eval_output.txt')

        result = compute_metrics(preds, out_label_ids)

        results.update(result)

        logger.info("***** Eval results *****")
        for key in results.keys():
            logger.info("  {} = {:.4f}".format(key, results[key]))

        with open(eval_out_file, 'a', encoding='utf-8') as f:
            for key in results.keys():
                f.write(" {} = {:.4f}\t".format(key, results[key]))
            f.write('\n\n')

        return results

    def predict(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        else:
            print("only test set is available")

        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        # Note that DistributedSampler samples randomly
        test_sampler = SequentialSampler(dataset) if self.args.local_rank == -1 else DistributedSampler(
            self.test_dataset)
        test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=self.args.eval_batch_size)

        # Test!
        logger.info("\n***** Running prediction on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        # test_loss = 0.0
        nb_test_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(test_dataloader, desc="Testing",ncols=100):
            # fp_indices = batch[11]
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'guid': batch[0],
                          'input_ids': batch[1],
                          'attention_mask': batch[2],
                          'token_type_ids': batch[3],
                          'labels': batch[4],
                          }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

            nb_test_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                out_attention_mask = inputs['attention_mask'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                out_attention_mask = np.append(
                    out_attention_mask, inputs['attention_mask'].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=2)

        final_preds = []
        final_label_ids = []

        for index, attention_mask in enumerate(out_attention_mask):
            length = 0
            for mask in attention_mask:
                if mask == 1:
                    length += 1
                else:
                    break

            final_label_ids.append(out_label_ids[index][:length])
            final_preds.append(preds[index][:length])

        preds = np.array(list(chain.from_iterable(final_preds)))
        out_label_ids = np.array(list(chain.from_iterable(final_label_ids)))

        print("length of test : {}".format(preds.shape[0]))

        # eval_loss = test_loss / nb_test_steps
        results = {}

        if not os.path.exists(self.args.model_dir):
            os.mkdir(self.args.model_dir)

        eval_out_file = os.path.join(self.args.model_dir, 'test_output.txt')

        result = compute_metrics(preds, out_label_ids)

        results.update(result)

        logger.info("***** Test results *****")
        for key in results.keys():
            logger.info("  {} = {:.4f}".format(key, results[key]))

        with open(eval_out_file, 'a', encoding='utf-8') as f:
            for key in results.keys():
                f.write(" {} = {:.4f}\t".format(key, results[key]))
            f.write('\n\n')

        return results

    def save_model(self, epoch):
        # Save model checkpoint
        output_dir = os.path.join(self.args.model_dir, 'epoch_{}'.format(epoch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")
        model_dir = self.args.model_dir
        self.config = self.config_class.from_pretrained(model_dir)
        self.model = self.model_class.from_pretrained(model_dir,
                                                      config=self.config,
                                                      args=self.args)
        self.model.to(self.device)
        logger.info("***** Model Loaded *****")
