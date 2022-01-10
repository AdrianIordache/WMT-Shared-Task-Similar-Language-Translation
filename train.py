from utils   import *
from dataset import *
from models  import *
import sys

CFG = {
    'batch_size_t': 16,
    'batch_size_v': 1,
    'learning_rate': 0.0001,
    'betas': (0.9, 0.98),
    'eps': 1e-9,

    'type': 'rnn',

    'n_heads': 8,
    'embedding_size': 512,
    'ffn_hidden_dim': 2048,

    'num_encoder_layers': 2,
    'num_decoder_layers': 2,

    'epochs': 1,
    'num_workers': 4,
    'debug': False, 
    'print_freq': 100, 
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

    SRC_VOCAB_SIZE = 4000
    TGT_VOCAB_SIZE = 4000

    if CFG['type'].lower() == 'transformer':
        model = Seq2SeqTransformer(
            CFG['num_encoder_layers'], 
            CFG['num_decoder_layers'], 
            CFG['embedding_size'], 
            CFG['n_heads'], 
            SRC_VOCAB_SIZE, 
            TGT_VOCAB_SIZE, 
            CFG['ffn_hidden_dim']
        )
    elif CFG['type'].lower() == 'rnn':
        INPUT_DIM = SRC_VOCAB_SIZE
        OUTPUT_DIM = TGT_VOCAB_SIZE
        # ENC_EMB_DIM = 256
        # DEC_EMB_DIM = 256
        # ENC_HID_DIM = 512
        # DEC_HID_DIM = 512
        # ATTN_DIM = 64
        # ENC_DROPOUT = 0.5
        # DEC_DROPOUT = 0.5

        ENC_EMB_DIM = CFG['embedding_size']
        DEC_EMB_DIM = CFG['embedding_size']
        ENC_HID_DIM = 64
        DEC_HID_DIM = 64
        ATTN_DIM = 8
        ENC_DROPOUT = 0.5
        DEC_DROPOUT = 0.5

        enc = EncoderRNN(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        attn = AttentionRNN(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
        dec = DecoderRNN(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
        model = Seq2SeqRNN(enc, dec, DEVICE)
    else:
        sys.exit('Invalid model name. Options are: Transformer / RNN')

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