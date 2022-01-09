from utils   import *
from dataset import *
from models  import *

CFG = {
    'batch_size_t': 1,
    'batch_size_v': 1,
    'learning_rate': 0.0001,
    'betas': (0.9, 0.98),
    'eps': 1e-9,

    'n_heads': 2,
    'embedding_size': 128,
    'ffn_hidden_dim': 128,

    'num_encoder_layers': 1,
    'num_decoder_layers': 1,

    'epochs': 3,
    'num_workers': 0,
    'debug': False, 
    'print_freq': 50, 
}

if __name__ == "__main__":
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

    SRC_VOCAB_SIZE = 32000
    TGT_VOCAB_SIZE = 32000

    model = Seq2SeqTransformer(
        CFG['num_encoder_layers'], 
        CFG['num_decoder_layers'], 
        CFG['embedding_size'], 
        CFG['n_heads'], 
        SRC_VOCAB_SIZE, 
        TGT_VOCAB_SIZE, 
        CFG['ffn_hidden_dim']
    )

    print(CFG)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model     = model.to(DEVICE)
    loss_fn   = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr = CFG['learning_rate'], betas = CFG['betas'], eps = CFG['eps'])

    for epoch in range(CFG['epochs']):
        train_avg_loss, train_loss_mean = train_epoch(model, trainloader, optimizer, loss_fn, epoch, CFG)
        valid_avg_loss, valid_loss_mean = valid_epoch(model, validloader, loss_fn, CFG)
        print(f"Epoch: [{epoch + 1}]/[{CFG['epochs']}], Train Loss: {train_avg_loss:.3f}, Valid Loss: {valid_avg_loss:.3f}")

        break
