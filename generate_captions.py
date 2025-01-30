import os, time
import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
from PIL import Image
import torchvision
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class CNN_Encoder(nn.Module):
    def __init__(self, encoded_image_size=14, attention_method="ByPixel"):
        super(CNN_Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.attention_method = attention_method

        resnet = torchvision.models.resnet50(pretrained=True)  # pretrained ImageNet ResNet

        # Remove linear and pool layers (since we're not doing classification)
        # Specifically, Remove: AdaptiveAvgPool2d(output_size=(1, 1)), Linear(in_features=2048, out_features=1000, bias=True)]
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        if self.attention_method == "ByChannel":
            self.cnn1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.relu = nn.ReLU(inplace=True)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images [batch_size, encoded_image_size=14, encoded_image_size=14, 2048]
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        if self.attention_method == "ByChannel":  # [batch_size, 2048, 8, 8] -> # [batch_size, 512, 8, 8]
            out = self.relu(self.bn1(self.cnn1(out)))
        out = self.adaptive_pool(out)  # [batch_size, 2048/512, 8, 8] -> [batch_size, 2048/512, 14, 14]
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

channel_number = 512

class ScaledDotProductAttention(nn.Module):
    def __init__(self, QKVdim):
        super(ScaledDotProductAttention, self).__init__()
        self.QKVdim = QKVdim

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size, n_heads, -1(len_q), QKVdim]
        :param K, V: [batch_size, n_heads, -1(len_k=len_v), QKVdim]
        :param attn_mask: [batch_size, n_heads, len_q, len_k]
        """
        # scores: [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.QKVdim)
        # Fills elements of self tensor with value where mask is True.
        scores.to(device).masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V).to(device)  # [batch_size, n_heads, len_q, QKVdim]
        return context, attn


class Multi_Head_Attention(nn.Module):
    def __init__(self, Q_dim, K_dim, QKVdim, n_heads=8, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.W_Q = nn.Linear(Q_dim, QKVdim * n_heads).to(device)
        self.W_K = nn.Linear(K_dim, QKVdim * n_heads).to(device)
        self.W_V = nn.Linear(K_dim, QKVdim * n_heads).to(device)
        self.n_heads = n_heads
        self.QKVdim = QKVdim
        self.embed_dim = Q_dim
        self.dropout = nn.Dropout(p=dropout)
        self.W_O = nn.Linear(self.n_heads * self.QKVdim, self.embed_dim).to(device)

    def forward(self, Q, K, V, attn_mask):
        """
        In self-encoder attention:
                Q = K = V: [batch_size, num_pixels=196, encoder_dim=2048]
                attn_mask: [batch_size, len_q=196, len_k=196]
        In self-decoder attention:
                Q = K = V: [batch_size, max_len=52, embed_dim=512]
                attn_mask: [batch_size, len_q=52, len_k=52]
        encoder-decoder attention:
                Q: [batch_size, 52, 512] from decoder
                K, V: [batch_size, 196, 2048] from encoder
                attn_mask: [batch_size, len_q=52, len_k=196]
        return _, attn: [batch_size, n_heads, len_q, len_k]
        """
        residual, batch_size = Q, Q.size(0)
        # q_s: [batch_size, n_heads=8, len_q, QKVdim] k_s/v_s: [batch_size, n_heads=8, len_k, QKVdim]
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        # attn_mask: [batch_size, self.n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # attn: [batch_size, n_heads, len_q, len_k]
        # context: [batch_size, n_heads, len_q, QKVdim]
        context, attn = ScaledDotProductAttention(self.QKVdim)(q_s, k_s, v_s, attn_mask)
        # context: [batch_size, n_heads, len_q, QKVdim] -> [batch_size, len_q, n_heads * QKVdim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.QKVdim).to(device)
        # output: [batch_size, len_q, embed_dim]
        output = self.W_O(context)
        output = self.dropout(output)
        return nn.LayerNorm(self.embed_dim).to(device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout):
        super(PoswiseFeedForwardNet, self).__init__()
        """
        Two fc layers can also be described by two cnn with kernel_size=1.
        """
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=d_ff, kernel_size=1).to(device)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=embed_dim, kernel_size=1).to(device)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim

    def forward(self, inputs):
        """
        encoder: inputs: [batch_size, len_q=196, embed_dim=2048]
        decoder: inputs: [batch_size, max_len=52, embed_dim=512]
        """
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        return nn.LayerNorm(self.embed_dim).to(device)(output + residual)


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, dropout, attention_method, n_heads):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=embed_dim, QKVdim=64, n_heads=n_heads, dropout=dropout)
        if attention_method == "ByPixel":
            self.dec_enc_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=2048, QKVdim=64, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=embed_dim, d_ff=2048, dropout=dropout)
        elif attention_method == "ByChannel":
            self.dec_enc_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=196, QKVdim=64, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=embed_dim, d_ff=2048, dropout=dropout)  # need to change

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_inputs: [batch_size, max_len=52, embed_dim=512]
        :param enc_outputs: [batch_size, num_pixels=196, 2048]
        :param dec_self_attn_mask: [batch_size, 52, 52]
        :param dec_enc_attn_mask: [batch_size, 52, 196]
        """
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, n_layers, vocab_size, embed_dim, dropout, attention_method, n_heads):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.tgt_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding.from_pretrained(self.get_position_embedding_table(embed_dim), freeze=True)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, dropout, attention_method, n_heads) for _ in range(n_layers)])
        self.projection = nn.Linear(embed_dim, vocab_size, bias=False).to(device)
        self.attention_method = attention_method

    def get_position_embedding_table(self, embed_dim):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / embed_dim)
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_idx) for hid_idx in range(embed_dim)]

        embedding_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(52)])
        embedding_table[:, 0::2] = np.sin(embedding_table[:, 0::2])  # dim 2i
        embedding_table[:, 1::2] = np.cos(embedding_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(embedding_table).to(device)

    def get_attn_pad_mask(self, seq_q, seq_k):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # In wordmap, <pad>:0
        # pad_attn_mask: [batch_size, 1, len_k], one is masking
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    def get_attn_subsequent_mask(self, seq):
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequent_mask = np.triu(np.ones(attn_shape), k=1)
        subsequent_mask = torch.from_numpy(subsequent_mask).byte().to(device)
        return subsequent_mask

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        :param encoder_out: [batch_size, num_pixels=196, 2048]
        :param encoded_captions: [batch_size, 52]
        :param caption_lengths: [batch_size, 1]
        """
        batch_size = encoder_out.size(0)
        # Sort input data by decreasing lengths.
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        sort_ind = sort_ind.to(torch.device('cuda'))
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # dec_outputs: [batch_size, max_len=52, embed_dim=512]
        # dec_self_attn_pad_mask: [batch_size, len_q=52, len_k=52], 1 if id=0(<pad>)
        # dec_self_attn_subsequent_mask: [batch_size, 52, 52], Upper triangle of an array with 1.
        # dec_self_attn_mask for self-decoder attention, the position whose val > 0 will be masked.
        # dec_enc_attn_mask for encoder-decoder attention.
        # e.g. 9488, 23, 53, 74, 0, 0  |  dec_self_attn_mask:
        # 0 1 1 1 2 2
        # 0 0 1 1 2 2
        # 0 0 0 1 2 2
        # 0 0 0 0 2 2
        # 0 0 0 0 1 2
        # 0 0 0 0 1 1
        dec_outputs = self.tgt_emb(encoded_captions) + self.pos_emb(torch.LongTensor([list(range(52))]*batch_size).to(device))
        dec_outputs = self.dropout(dec_outputs)
        dec_self_attn_pad_mask = self.get_attn_pad_mask(encoded_captions, encoded_captions)
        dec_self_attn_subsequent_mask = self.get_attn_subsequent_mask(encoded_captions)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        if self.attention_method == "ByPixel":
            dec_enc_attn_mask = (torch.tensor(np.zeros((batch_size, 52, 196))).to(device) == torch.tensor(np.ones((batch_size, 52, 196))).to(device))
        elif self.attention_method == "ByChannel":
            dec_enc_attn_mask = (torch.tensor(np.zeros((batch_size, 52, channel_number))).to(device) == torch.tensor(np.ones((batch_size, 52, channel_number))).to(device))

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # attn: [batch_size, n_heads, len_q, len_k]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, encoder_out, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        predictions = self.projection(dec_outputs)
        return predictions, encoded_captions, decode_lengths, sort_ind, dec_self_attns, dec_enc_attns


class EncoderLayer(nn.Module):
    def __init__(self, dropout, attention_method, n_heads):
        super(EncoderLayer, self).__init__()
        """
        In "Attention is all you need" paper, dk = dv = 64, h = 8, N=6
        """
        if attention_method == "ByPixel":
            self.enc_self_attn = Multi_Head_Attention(Q_dim=2048, K_dim=2048, QKVdim=64, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=2048, d_ff=4096, dropout=dropout)
        elif attention_method == "ByChannel":
            self.enc_self_attn = Multi_Head_Attention(Q_dim=196, K_dim=196, QKVdim=64, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=196, d_ff=512, dropout=dropout)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :param enc_inputs: [batch_size, num_pixels=196, 2048]
        :param enc_outputs: [batch_size, len_q=196, d_model=2048]
        :return: attn: [batch_size, n_heads=8, 196, 196]
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, n_layers, dropout, attention_method, n_heads):
        super(Encoder, self).__init__()
        if attention_method == "ByPixel":
            self.pos_emb = nn.Embedding.from_pretrained(self.get_position_embedding_table(), freeze=True)
        # self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([EncoderLayer(dropout, attention_method, n_heads) for _ in range(n_layers)])
        self.attention_method = attention_method

    def get_position_embedding_table(self):
        def cal_angle(position, hid_idx):
            x = position % 14
            y = position // 14
            x_enc = x / np.power(10000, hid_idx / 1024)
            y_enc = y / np.power(10000, hid_idx / 1024)
            return np.sin(x_enc), np.sin(y_enc)
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_idx)[0] for hid_idx in range(1024)] + [cal_angle(position, hid_idx)[1] for hid_idx in range(1024)]

        embedding_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(196)])
        return torch.FloatTensor(embedding_table).to(device)

    def forward(self, encoder_out):
        """
        :param encoder_out: [batch_size, num_pixels=196, dmodel=2048]
        """
        batch_size = encoder_out.size(0)
        positions = encoder_out.size(1)
        if self.attention_method == "ByPixel":
            encoder_out = encoder_out + self.pos_emb(torch.LongTensor([list(range(positions))]*batch_size).to(device))
        # encoder_out = self.dropout(encoder_out)
        # enc_self_attn_mask: [batch_size, 196, 196]
        enc_self_attn_mask = (torch.tensor(np.zeros((batch_size, positions, positions))).to(device)
                              == torch.tensor(np.ones((batch_size, positions, positions))).to(device))
        enc_self_attns = []
        for layer in self.layers:
            encoder_out, enc_self_attn = layer(encoder_out, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return encoder_out, enc_self_attns


class Transformer(nn.Module):
    """
    See paper 5.4: "Attention Is All You Need" - https://arxiv.org/abs/1706.03762
    "Apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
    In addition, apply dropout to the sums of the embeddings and the positional encodings in both the encoder
    and decoder stacks." (Now, we dont't apply dropout to the encoder embeddings)
    """
    def __init__(self, vocab_size, embed_dim, encoder_layers, decoder_layers, dropout=0.1, attention_method="ByPixel", n_heads=8):
        super(Transformer, self).__init__()
        self.encoder = Encoder(encoder_layers, dropout, attention_method, n_heads)
        self.decoder = Decoder(decoder_layers, vocab_size, embed_dim, dropout, attention_method, n_heads)
        self.embedding = self.decoder.tgt_emb
        self.attention_method = attention_method

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self, enc_inputs, encoded_captions, caption_lengths):
        """
        preprocess: enc_inputs: [batch_size, 14, 14, 2048]/[batch_size, 196, 2048] -> [batch_size, 196, 2048]
        encoded_captions: [batch_size, 52]
        caption_lengths: [batch_size, 1], not used
        The encoder or decoder is composed of a stack of n_layers=6 identical layers.
        One layer in encoder: Multi-head Attention(self-encoder attention) with Norm & Residual
                            + Feed Forward with Norm & Residual
        One layer in decoder: Masked Multi-head Attention(self-decoder attention) with Norm & Residual
                            + Multi-head Attention(encoder-decoder attention) with Norm & Residual
                            + Feed Forward with Norm & Residual
        """
        batch_size = enc_inputs.size(0)
        encoder_dim = enc_inputs.size(-1)
        if self.attention_method == "ByPixel":
            enc_inputs = enc_inputs.view(batch_size, -1, encoder_dim)
        elif self.attention_method == "ByChannel":
            enc_inputs = enc_inputs.view(batch_size, -1, encoder_dim).permute(0, 2, 1)  # (batch_size, 2048, 196)

        encoder_out, enc_self_attns = self.encoder(enc_inputs)
        # encoder_out: [batch_size, 196, 2048]
        predictions, encoded_captions, decode_lengths, sort_ind, dec_self_attns, dec_enc_attns = self.decoder(encoder_out, encoded_captions, caption_lengths)
        alphas = {"enc_self_attns": enc_self_attns, "dec_self_attns": dec_self_attns, "dec_enc_attns": dec_enc_attns}
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

beam_size = 3
def caption_image_beam_search(encoder, decoder, img, word_map):
    """
    Reads an image and captions it with beam search.

    Args
    encoder: encoder model
    decoder: decoder model
    image_path: path to image
    word_map: word map
    beam_size: number of sequences to consider at each decode-step
    
    Args:
    caption, weights for visualization
    """

    k = beam_size
    Caption_End = False
    vocab_size = len(word_map)

    # Read image and process
    # img = imageio.imread(image_path)
    # Convert any image with more than 3(rgb) channels to RGB
    img = img.convert('RGB')
    img = np.array(img)

    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = np.array(Image.fromarray(img).resize((256, 256)))
    # img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(-1)
    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # [1, num_pixels=196, encoder_dim]
    num_pixels = encoder_out.size(1)
    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']] * 52] * k).to(device)  # (k, 52)

    # Tensor to store top k sequences; now they're just <start>
    seqs = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)
    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        cap_len = torch.LongTensor([52]).repeat(k, 1)  # [s, 1]
        scores, _, _, alpha_dict, _ = decoder(encoder_out, k_prev_words, cap_len)
        scores = scores[:, step - 1, :].squeeze(1)  # [s, 1, vocab_size] -> [s, vocab_size]
        # choose the last layer, transformer decoder is comosed of a stack of 6 identical layers.
        alpha = alpha_dict["dec_enc_attns"][-1]  # [s, n_heads=8, len_q=52, len_k=196]
        # TODO: AVG Attention to Visualize
        # for i in range(len(alpha_dict["dec_enc_attns"])):
        #     n_heads = alpha_dict["dec_enc_attns"][i].size(1)
        #     for j in range(n_heads):
        #         pass
        # the second dim corresponds to the Multi-head attention = 8, now 0
        # the third dim corresponds to cur caption position
        alpha = alpha[:, 0, step-1, :].view(k, 1, enc_image_size, enc_image_size)  # [s, 1, enc_image_size, enc_image_size]

        scores = F.log_softmax(scores, dim=1)
        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)
        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds]], dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        # Set aside complete sequences
        if len(complete_inds) > 0:
            Caption_End = True
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        
        k_prev_words = k_prev_words[incomplete_inds]
        k_prev_words[:, :step + 1] = seqs  # [s, 52]
        # k_prev_words[:, step] = next_word_inds[incomplete_inds]  # [s, 52]

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    assert Caption_End
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq


def generate_caption(img):
    checkpoint = "./model/checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar" # path to model
    word_map = "./dataset/generated_data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json" # path to word map JSON

    # Load model
    checkpoint = torch.load(checkpoint, map_location=str(device), weights_only=False)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    # print(encoder)
    # print(decoder)

    # Load word map (word2ix)
    with open(word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # idx to word

    
    with torch.no_grad():
        seq = caption_image_beam_search(encoder, decoder, img, word_map)
    
    words = [rev_word_map[ind] for ind in seq]
    words = words[1:-1]

    return " ".join(words)


if __name__ == '__main__':
    img = "./image/blabla.png" # path to image, file or folder
    checkpoint = "./model/checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar" # path to model
    word_map = "./dataset/generated_data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json" # path to word map JSON'
    # decoder_mode ="transformer"  # transformer
    save_img_dir = "./output/" # path to save annotated img
    beam_size = 3 # beam size for beam search
    # dont_smooth ='store_false' # do not smooth alpha overlay

    # Load model
    checkpoint = torch.load(checkpoint, map_location=str(device), weights_only=False)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    # print(encoder)
    # print(decoder)

    # Load word map (word2ix)
    with open(word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # idx to word

    
    with torch.no_grad():
        seq = caption_image_beam_search(encoder, decoder, img, word_map)
    
    words = [rev_word_map[ind] for ind in seq]
    words = words[1:-1]

    print(" ".join(words))