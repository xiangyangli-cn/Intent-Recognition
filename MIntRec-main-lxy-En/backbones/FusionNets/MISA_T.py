import torch
from torch import nn
from torch.autograd import Function
from ..SubNets.FeatureNets import BERTEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = ['MISA_T']

class ReverseLayerF(Function):
    """
    Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
    """
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None

class MISA_T(nn.Module):
    
    def __init__(self, args):
        print("Backbone: T-A---000000")
        #exit()

        super(MISA_T, self).__init__()

        self.text_subnet = BERTEncoder.from_pretrained('./bert_config/bert-base-uncased', cache_dir = args.cache_path)
        #self.visual_size = args.video_feat_dim
        self.acoustic_size = args.audio_feat_dim
        self.text_size = args.text_feat_dim

        self.dropout_rate = args.dropout_rate
        self.output_dim = args.num_labels
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.input_sizes = input_sizes = [self.text_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.acoustic_size)]
        self.args = args
        
        rnn = nn.LSTM if args.rnncell == "lstm" else nn.GRU

        #self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        #self.vrnn2 = rnn(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.arnn2 = rnn(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True)

        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0], out_features=args.hidden_size))
        self.project_t.add_module('project_t_activation', self.activation)
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(args.hidden_size))
        print(hidden_sizes)
        print(args.hidden_size)
        #exit()
        
        #self.project_v = nn.Sequential()
        #self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1] * 4, out_features=args.hidden_size))
        #self.project_v.add_module('project_v_activation', self.activation)
        #self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(args.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[1] * 4, out_features=args.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(args.hidden_size))

        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        #self.private_v = nn.Sequential()
        #self.private_v.add_module('private_v_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        #self.private_v.add_module('private_v_activation_1', nn.Sigmoid())

        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        #self.recon_v = nn.Sequential()
        #self.recon_v.add_module('recon_v_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))

        if not args.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(args.dropout_rate))
            self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=args.hidden_size, out_features=len(hidden_sizes)))

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=args.hidden_size, out_features = 4))

        self.fusion = nn.Sequential() #fusion---
        #self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=args.hidden_size * 1, out_features=args.hidden_size * 1))
        #self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(self.dropout_rate))
        #self.fusion.add_module('fusion_layer_1_activation', self.activation)
        #self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=args.hidden_size * 1, out_features= self.output_dim))

        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=768, out_features=args.hidden_size ))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(self.dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=args.hidden_size , out_features= self.output_dim))

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        #self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))

        encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
   
  

    def _shared_private(self, utterance_t, utterance_a):
        
        # Projecting to same sized space
        self.utt_t_orig  = self.project_t(utterance_t)
        #self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig  = self.project_a(utterance_a)
        
       
    
    def _extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):

        packed_sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        
        if self.args.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        padded_h1 = padded_h1.permute(1, 0, 2)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths, batch_first=True, enforce_sorted=False)

        if self.args.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def forward(self, text_feats, video_feats, audio_feats):

        batch_size = text_feats.size(0)
        bert_sent, bert_sent_mask, bert_sent_type = text_feats[:,0], text_feats[:,1], text_feats[:,2]
        bert_output = self.text_subnet(text_feats) # [batch_size, seq_len, 768]

        # Use the mean value of bert of the front real sentence length as the final representation of text.
        masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
        mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  
        text_feats = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
        #print(text_feats.shape) #16*768
        #exit()

        lengths = mask_len.squeeze().int().detach().cpu().view(-1)

        # extract features from visual modality
        #final_h1v, final_h2v = self._extract_features(video_feats, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        #video_feats = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from acoustic modality
        final_h1a, final_h2a = self._extract_features(audio_feats, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        audio_feats = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        #print(audio_feats.shape)
        #exit()

        self._shared_private(text_feats, audio_feats)
        
       
        #h = self.utt_a_orig+text_feats  #audio+text
        #h = self.utt_a_orig  #audio
        #print(h.shape)
        #exit()
        #h=self.utt_t_orig
        h=text_feats
       


        #print(h.shape) 16*256
        #exit()
        logits = self.fusion(h)

        return logits