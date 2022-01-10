from utils     import *
from dataset   import *
from models    import *
from translate import *

CFG = {
    'batch_size_t': 4,
    'batch_size_v': 1,
    
    # Optimizer Hyper-parameters
    'learning_rate': 0.0001,
    'betas': (0.9, 0.98),
    'eps': 1e-9,

    # Vocabulary Hyper-parameters
    'src_vocab_size': 32000,
    'tgt_vocab_size': 32000, 

    # Architecture Hyper-parameters
    'architecture_type': 'transformer',
    'embedding_size': 512,               # transformer & rnn

    'n_heads': 8,                        # transformer
    'ffn_hidden_dim': 2048,              # transformer
    'num_encoder_layers': 2,             # transformer
    'num_decoder_layers': 2,             # transformer

    'attention_dim': 8,                  # rnn
    'encoder_hidden_dim': 64,            # rnn
    'decoder_hidden_dim': 64,            # rnn
    'encoder_dropout': 0.5,              # rnn
    'decoder_dropout': 0.5,              # rnn

    # Training Script Parameters
    'epochs': 10,
    'num_workers': 4,
    'debug': False, 
    'print_freq': 100, 
    'observation': None, # "Should be a string, more specific information for experiments"
}


if __name__ == "__main__":
    print(f"Config File: {CFG}")
    with open(PATH_TO_CLEANED_TRAIN[SRC_LANGUAGE], 'r') as src_file: train_src_sentences = src_file.read().splitlines()
    with open(PATH_TO_CLEANED_TRAIN[TGT_LANGUAGE], 'r') as tgt_file: train_tgt_sentences = tgt_file.read().splitlines()

    with open(PATH_TO_CLEANED_VALID[SRC_LANGUAGE], 'r') as src_file: valid_src_sentences = src_file.read().splitlines()
    with open(PATH_TO_CLEANED_VALID[TGT_LANGUAGE], 'r') as tgt_file: valid_tgt_sentences = tgt_file.read().splitlines()
    
    trainset = TranslationDataset(
        src_sentences   = train_src_sentences,
        tgt_sentences   = train_tgt_sentences,
        sentpiece_model = PATH_TO_MODEL
    )

    validset = TranslationDataset(
        src_sentences   = valid_src_sentences,
        tgt_sentences   = valid_tgt_sentences,
        sentpiece_model = PATH_TO_MODEL
    )

    trainloader = DataLoader(
        dataset        = trainset, 
        batch_size     = CFG['batch_size_t'], 
        shuffle        = True, 
        collate_fn     = generate_batch,
        num_workers    = CFG['num_workers'], 
        worker_init_fn = seed_worker, 
        pin_memory     = False,
        drop_last      = False
    )

    validloader = DataLoader(
        dataset        = validset, 
        batch_size     = CFG['batch_size_v'], 
        shuffle        = True, 
        collate_fn     = generate_batch,
        num_workers    = CFG['num_workers'], 
        worker_init_fn = seed_worker, 
        pin_memory     = True,
        drop_last      = False
    )

    model = get_model(CFG)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model     = model.to(DEVICE)
    loss_fn   = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr = CFG['learning_rate'], betas = CFG['betas'], eps = CFG['eps'])

    best_loss = np.inf
    for epoch in range(CFG['epochs']):
        train_avg_loss, train_loss_mean = train_epoch(model, trainloader, optimizer, loss_fn, epoch, CFG)
        valid_avg_loss, valid_loss_mean = valid_epoch(model, validloader, loss_fn, CFG)
        print(f"Epoch: [{epoch + 1}]/[{CFG['epochs']}], Train Loss: {train_loss_mean:.3f}, Valid Loss: {valid_loss_mean:.3f}")
