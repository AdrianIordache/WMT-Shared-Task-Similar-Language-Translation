from models      import *
from dataset     import *
from config_file import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', 
    dest = 'gpu', type = int, 
    default = 0, help = "GPU enable for running the procces"
)

args = parser.parse_args()
RANK = args.gpu

USER        = 'adrian'
QUIET       = False 
SAVE_TO_LOG = True

DEVICE = torch.device(f'cuda:{RANK}' if torch.cuda.is_available() else 'cpu')
GLOBAL_LOGGER = GlobalLogger(path_to_global_logger = f'logs/dataset-{DATASET_VERSION}/{USER}/gpu-{RANK}/global_logger.csv', save_to_log = SAVE_TO_LOG)

CFG = {
    'id'                              : GLOBAL_LOGGER.get_version_id(),
    'tokens_in_batch'                 : 4000,
    'label_smoothing'                 : 0.1,

    # Optimizer Hyper-parameters
    'learning_rate'                   : get_lr(step = 1, d_model = 512, warmup_steps = 8000),
    'warmup_steps'                    : 8000,
    'betas'                           : (0.9, 0.98),
    'eps'                             : 1e-9,

    # Vocabulary Hyper-parameters
    'vocab_size'                      : 37000,
    'max_seq_len'                     : 256,             # transformer (needs to be computed)

    # Architecture Hyper-parameters
    'architecture_type'               : 'transformer',
    'd_model'                         : 512,             # transformer & rnn  (can also be considered as embeddings size)
    'n_heads'                         : 8,               # transformer
    'd_queries'                       : 64,              # transformer
    'd_values'                        : 64,              # transformer
    'd_feed_forward'                  : 2048,            # transformer
    'n_layers'                        : 6,               # transformer
    'dropout'                         : 0.1,             # transformer

    'attention_dim'                   : 8,               # rnn
    'encoder_hidden_dim'              : 64,              # rnn
    'decoder_hidden_dim'              : 64,              # rnn
    'encoder_dropout'                 : 0.5,             # rnn
    'decoder_dropout'                 : 0.5,             # rnn

    # Training Script Parameters
    'n_steps'                         : 50000,
    'epochs'                          :'NA',

    'num_workers'                     : 4,
    'debug'                           : False, 
    'print_freq'                      : 100, 
    'observation'                     : None, # "Should be a string, more specific information for experiments"
    'save_to_log'                     : SAVE_TO_LOG
}

def train(loader, model, loss_fn, optimizer, epoch, step, config_file, logger):
    model.train() 

    losses_plot = []
    losses      = AverageMeter()
    start = end = time.time()

    for batch_idx, (src_sequences, tgt_sequences, src_sequence_lengths, tgt_sequence_lengths) in enumerate(loader):
        src_sequences        = src_sequences.to(DEVICE)  # (N, max_source_sequence_pad_length_this_batch)
        tgt_sequences        = tgt_sequences.to(DEVICE)  # (N, max_target_sequence_pad_length_this_batch)
        src_sequence_lengths = src_sequence_lengths.to(DEVICE)  # (N)
        tgt_sequence_lengths = tgt_sequence_lengths.to(DEVICE)  # (N)

        # (N, max_target_sequence_pad_length_this_batch, vocab_size)
        pred_sequences = model(src_sequences, tgt_sequences, src_sequence_lengths, tgt_sequence_lengths)  

        # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
        # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
        # Therefore, pads start after (length - 1) positions
        loss = loss_fn(
            inputs  = pred_sequences,
            targets = tgt_sequences[:, 1:],
            lengths = tgt_sequence_lengths.cpu() - 1
        )

        (loss / batches_per_step).backward()
        losses.update(loss.item(), (tgt_sequence_lengths - 1).sum().item())

        # Update model (i.e. perform a training step) only after gradients are accumulated from batches_per_step batches
        if (batch_idx + 1) % batches_per_step == 0:
            optimizer.step()
            optimizer.zero_grad()

            # This step is now complete
            step += 1
            change_lr(optimizer, new_lr = get_lr(step = step, d_model = config_file['d_model'], warmup_steps = config_file['warmup_steps']))

            end = time.time()
            if step % config_file['print_freq'] == 0 or step == (config_file['n_steps'] - 1):
                logger.print('[GPU {0}][TRAIN] Epoch: [{1}/{2}][{3}/{4}], Elapsed {remain:s}, Loss: {loss.value:.3f}({loss.average:.3f})'
                      .format(DEVICE, epoch + 1, epochs, step, config_file['n_steps'], 
                        remain   = time_since(start, float(batch_idx + 1) / loader.n_batches), 
                        loss     = losses)
                )

            losses_plot.append(losses.value)

    free_gpu_memory(DEVICE)
    return losses.average, np.mean(losses_plot)

def validate(loader, model, loss_fn):
    model.eval() 

    with torch.no_grad():
        losses = AverageMeter()
        losses_plot = []

        for batch_idx, (src_sequences, tgt_sequences, src_sequence_lengths, tgt_sequence_lengths) in enumerate(tqdm(loader, total = loader.n_batches)):
            src_sequences = src_sequences.to(DEVICE)  # (1, source_sequence_length)
            tgt_sequences = tgt_sequences.to(DEVICE)  # (1, target_sequence_length)
            src_sequence_lengths = src_sequence_lengths.to(DEVICE)  # (1)
            tgt_sequence_lengths = tgt_sequence_lengths.to(DEVICE)  # (1)

            # (1, target_sequence_length, vocab_size)
            pred_sequences = model(src_sequences, tgt_sequences, src_sequence_lengths, tgt_sequence_lengths) 

            # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
            # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
            # Therefore, pads start after (length - 1) positions
            loss = loss_fn(
                inputs  = pred_sequences,
                targets = tgt_sequences[:, 1:],
                lengths = tgt_sequence_lengths.cpu() - 1
            )

            losses.update(loss.item(), (tgt_sequence_lengths - 1).sum().item())
            losses_plot.append(losses.value)

    free_gpu_memory(DEVICE)
    return losses.average, np.mean(losses_plot)

if __name__ == "__main__":
    if SAVE_TO_LOG:
        PATH_TO_MODELS = os.path.join(PATH_TO_MODELS, f'{USER}', f'gpu-{RANK}', 'model-{}'.format(CFG['id']))
        if os.path.isdir(PATH_TO_MODELS) == False: os.makedirs(PATH_TO_MODELS)
        logger = Logger(os.path.join(PATH_TO_MODELS, 'model_{}.log'.format(CFG['id'])), distributed = QUIET)
    else:
        logger = Logger(distributed = QUIET)

    logger.print(f"Config File: {CFG}")

    trainloader = SequenceLoader(
        data_folder     = f"data/cleaned/dataset-{DATASET_VERSION}/", 
        vocab_size      = CFG['vocab_size'], 
        source_language = SRC_LANGUAGE, 
        target_language = TGT_LANGUAGE, 
        dataset_type    = "train", 
        tokens_in_batch = CFG['tokens_in_batch']
    )

    validloader = SequenceLoader(
        data_folder     = f"data/cleaned/dataset-{DATASET_VERSION}/", 
        vocab_size      = CFG['vocab_size'], 
        source_language = SRC_LANGUAGE, 
        target_language = TGT_LANGUAGE, 
        dataset_type    = "valid", 
        tokens_in_batch = CFG['tokens_in_batch']
    )

    model = get_model(CFG, DEVICE).to(DEVICE)

    optimizer  = torch.optim.Adam(
        params = [p for p in model.parameters() if p.requires_grad],
        lr     = CFG['learning_rate'],
        betas  = CFG['betas'],
        eps    = CFG['eps']
    )

    loss_fn = CrossEntropyLossSmoothed(eps = CFG['label_smoothing'], device = DEVICE).to(DEVICE)
    
    batches_per_step = 25000 // CFG['tokens_in_batch']
    epochs = (CFG['n_steps'] // (trainloader.n_batches // batches_per_step)) + 1

    best_model = None
    best_train_loss = np.inf 
    best_valid_loss = np.inf
    for epoch in range(epochs):
        step = epoch * trainloader.n_batches // batches_per_step

        train_avg_loss, train_loss_mean = train(
            loader      = trainloader,
            model       = model,
            loss_fn     = loss_fn,
            optimizer   = optimizer,
            epoch       = epoch,
            step        = step,
            config_file = CFG,
            logger      = logger  
        )

        valid_avg_loss, valid_loss_mean = validate(
            loader      = validloader,
            model       = model,
            loss_fn     = loss_fn,
        )

        logger.print(f"Epoch: [{epoch + 1}]/[{epochs}], Train Loss: {train_loss_mean:.3f}, Valid Loss: {valid_loss_mean:.3f}")

        if valid_loss_mean < best_valid_loss and CFG['save_to_log']:
            best_train_loss = train_loss_mean
            best_valid_loss = valid_loss_mean

            torch.save({
                'model': model.state_dict(),
            }, os.path.join(PATH_TO_MODELS, f"model_{CFG['id']}_name_{CFG['architecture_type']}_loss_{best_valid_loss:.2f}.pth"))

    OUTPUT['train_loss'] = best_train_loss
    OUTPUT['valid_loss'] = best_valid_loss
    GLOBAL_LOGGER.append(CFG, OUTPUT)
