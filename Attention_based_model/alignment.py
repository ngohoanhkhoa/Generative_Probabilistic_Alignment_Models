
import os, datetime
import torchtext
import tools as tools
import torch
import torch.nn as nn
from model_attention import make_model, USE_CUDA

class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, src, trg, pad_index=0):

        src, src_lengths = src

        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)

        trg, trg_lengths = trg
        # NOTE: commenting that out
        # self.trg = trg[:, :-1]  # QUESTION: why cut EOS? Because it's never fed in teacher forcing? (careful with that maybe!)
        self.trg = trg
        self.trg_lengths = trg_lengths
        self.trg_y = trg[:, 1:]  # I understand better why we cut SOS here for eval though
        self.trg_mask = (self.trg_y != pad_index)
        self.ntokens = (self.trg_y != pad_index).data.sum().item()

        if USE_CUDA:  # could probably write this with .to(device) instead
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                self.trg_mask = self.trg_mask.cuda()

class AttentionAlignment(object):
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        log_fname = os.path.join(args.model_dir, args.model_name, 
                                 'log' + '.' + \
                                 '{}'.format(datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')))
                                 
        tools.Logger.setup(log_file=log_fname, timestamp=False)
        self.log = tools.Logger.get()
        
        self.source_file = args.source_file
        self.target_file = args.target_file
        
        
        self.batch_size = args.batch_size
        
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.n_layers = args.n_layers
        self.dropout_proba = args.dropout_proba
        self.update_first = args.update_first
        self.word_bias = args.word_bias
        
        self.nb_epochs = args.nb_epochs
        self.learning_rate = args.learning_rate
        self.aux_loss = args.aux_loss
        self.aux_wait = args.aux_wait
        
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                # TODO: essayer autre chose? loi normale
                m.bias.data.fill_(0.0)
        if type(m) == nn.Embedding:
            # doing it like the LIG system
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
            # torch.nn.init.xavier_uniform_(m.weight)

    
    def run(self):
        train_set, SRC, TRG, PAD_INDEX = tools.prepare_data_torchtext(self.train_dataset)
        test_set, SRC, TRG, PAD_INDEX = tools.prepare_data_torchtext(self.test_dataset)
        
        train_iter = torchtext.data.BucketIterator(train_set, 
                                                   batch_size=self.batch_size,
                                                   train=True,
                                                   sort_within_batch=True,
                                                   sort_key=lambda x: (len(x.Source), len(x.Target)),
                                                   repeat=False,
                                                   device=self.device)
                                                   
        decode_iter = torchtext.data.Iterator(train_set, 
                                              batch_size=1,
                                              train=False,
                                              sort=False, 
                                              repeat=False,
                                              device=self.device)
                                              
        model = make_model(SRC.vocab, len(SRC.vocab), len(TRG.vocab),
                           emb_size=self.emb_size, 
                           hidden_size=self.hidden_size,
                           num_layers=self.n_layers, 
                           dropout=self.dropout_proba, 
                           generate_first=not self.update_first, 
                           word_bias=self.word_bias)
                           
        train_perplexities = self.train(model, 
                                        train_iter, 
                                        decode_iter, 
                                        SRC, TRG, PAD_INDEX,
                                        num_epochs=self.nb_epochs,
                                        lr=self.learning_rate,
                                        with_aux_loss=self.aux_loss,
                                        aux_wait=self.aux_wait)

    
        
    def train(self, model, 
              train_iter, 
              decode_iter, 
              SRC, TRG, pad_index,
              num_epochs=1,
              lr=0.001,
              with_aux_loss=False,
              aux_wait=0):
        
        if USE_CUDA:
            model.cuda()

        # optionally add label smoothing; see the Annotated Transformer
        criterion = nn.NLLLoss(reduction="sum", ignore_index=pad_index)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        model.apply(self.init_weights)
    
        train_perplexities = []
    
        batch_cumul = 0
        for epoch in range(1, num_epochs + 1):
            model.train()
            if with_aux_loss:
                # QUESTION: why do we instantiate these objects for each epoch (as opposed to the whole training, outside the for loop) ?
                Loss = AuxLossCompute(model.generator, criterion, optim)
            else:
                Loss = SimpleLossCompute(model.generator, criterion, optim)
                
            train_perplexity, batch_number = self.run_epoch((rebatch(pad_index, b) for b in train_iter),
                                                       model,
                                                       Loss,
                                                       print_every=print_every,
                                                       writer=writer, batch_counter=batch_cumul, curr_epoch=epoch, nb_epochs=num_epochs, aux_wait=aux_wait)
            batch_cumul += batch_number
            
            train_perplexities.append(train_perplexity)
            
            if batch_number % self.eval_step == 0:
                self.evaluate()
                
        
        
        def print_training_info(self):
            self.log.info('-----------------------------------------------------------------')
            self.log.info('-----------------------------------------------------------------')
    
    
    def run_epoch(self, data_iter, 
                  model, 
                  loss_compute, 
                  print_every=50, 
                  writer=None, 
                  batch_counter=0, 
                  curr_epoch=0,
                  nb_epochs=1, 
                  aux_wait=0):
                      
        """Standard Training and Logging Function"""
    
        start = time.time()
        total_tokens = 0
        total_loss = 0
        total_nll_loss = 0
        total_aux_loss = 0
        print_tokens = 0
    
        if isinstance(loss_compute, AuxLossCompute):
            with_aux = True
            lambda_aux = max(curr_epoch - aux_wait, 0) / nb_epochs
        else:
            with_aux = False
    
        for i, batch in enumerate(data_iter, 1):
    
            out, _, pre_output, attention_vectors = model.forward(batch.src, batch.trg,
                                               batch.src_mask, batch.trg_mask,
                                               batch.src_lengths, batch.trg_lengths)
            if with_aux:
                loss, nll_loss, aux_loss = loss_compute(pre_output, batch.trg_y, batch.nseqs,
                                                        attention_vectors, batch.src_lengths, batch.trg_lengths, batch.trg_mask, lambda_aux)
                total_nll_loss += nll_loss
                total_aux_loss += aux_loss
            else:
                loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
    
            total_loss += loss
            total_tokens += batch.ntokens
            print_tokens += batch.ntokens
    
            if model.training and i % print_every == 0:
                elapsed = time.time() - start
                logging.info("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                      (i, loss / batch.nseqs, print_tokens / elapsed))
                if with_aux:
                    logging.info("Epoch Step: %d NLL loss: %f" % (i, nll_loss / batch.nseqs))
                    logging.info("Epoch Step: %d AUX loss: %f" % (i, aux_loss / batch.nseqs))
    
                if writer is not None:
                    writer.add_scalar("batched/loss", loss / batch.nseqs, batch_counter + i)
                    if with_aux:
                        writer.add_scalar("batched/nll_loss", nll_loss / batch.nseqs, batch_counter + i)
                        writer.add_scalar("batched/aux_loss", aux_loss / batch.nseqs, batch_counter + i)
                start = time.time()
                print_tokens = 0
    
        # NOTE: with aux loss this isn't really a perplexity anymore
        return math.exp(total_loss / float(total_tokens)), i
    
class AuxLossCompute:
    """Compute sum of NLL and auxiliary loss."""
    
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
    
    def __call__(self, x, y, norm, matrix, src_length, trg_length, trg_mask, lambda_aux):
        x = self.generator(x)
        nll_loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        nll_loss = nll_loss / norm
        # old_aux_loss = calculate_aux_loss(matrix, src_length, trg_length)
        aux_loss = self.batch_calculate_aux_loss(matrix, src_length, trg_length, trg_mask)
        aux_loss = aux_loss / norm
    
        # NOTE: TEST, hard coded
        loss = nll_loss + lambda_aux * aux_loss
        # loss = (1.0 - lambda_aux) * nll_loss + lambda_aux * aux_loss
    
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        # QUESTION: note sure why SimpleLossCompute re-multiplies by norm...
        return loss.data.item() * norm, nll_loss.data.item() * norm , aux_loss.data.item() * norm
        
    def batch_calculate_aux_loss(self, matrix, src_length, trg_length, trg_mask):
        """ Optimize calculation of auxiliary loss to bias the number of units
        on the target side.
    
        matrix input of dimension [B, TN, SN] (batch size, target max length, source max length]
        """
    
        # mask invalid attention weights
        # NB: 22-01-2019, remove last line (EOS) on the target side
        # (I don't know how to avoid a for loop here w/ fancy indexing or if it's possible)
        mask_trg_mask = torch.zeros_like(trg_mask)
        for i in range(trg_length.size(0)):
            mask_trg_mask[i, trg_length[i] - 2] = 1
        trg_mask.masked_fill_(mask_trg_mask == 1, 0.0)
        matrix.data.masked_fill_(trg_mask.unsqueeze(2) == 0, 0.0)
    
        matrix_transpose = matrix.transpose(1, 2)  # [B, TN, TS] -> [B, TS, TN]
        product = torch.bmm(matrix, matrix_transpose)  # [B, TN, TN]
        # cf. "To take a batch diagonal, pass in dim1=-2, dim2=-1."
        # https://pytorch.org/docs/0.4.1/torch.html?highlight=diagonal#torch.diagonal
        dots = torch.diagonal(product, offset=-1, dim1=-2, dim2=-1)
        S = dots.sum(dim=1)
    
        # NOTE: Re-adding substraction of 1.0 (SOS token present in trg_length but not in the attention matrices)
        # standard AUX
        aux_losses = trg_length.float() - 1.0 - src_length.float() - S
    
        # variant: adding a mult. factor (avg. ratio nb_mb_wd / nb_fr_wd on first 100 sent)
        # avg_ratio = 0.789176
        # aux_losses = trg_length.float() - 1.0 - avg_ratio * src_length.float() - S
    
        # approximate absolute value in a differentiable manner
        abs_values = torch.sqrt(aux_losses * aux_losses + 0.001)
        total_aux = torch.sum(abs_values)
        return total_aux

class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm
        
