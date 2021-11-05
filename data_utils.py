# coding=utf-8


from random import randint, shuffle, choice
from random import random as rand
import math
import numpy as np
import torch
import torch.utils.data



def get_random_word(vocab_words):
    i = randint(0, len(vocab_words)-1)
    return vocab_words[i]


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if x[0] is None:
            batch_tensors.append(None)
        elif isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            try:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))
            except:
                batch_tensors.append(None)
    return batch_tensors


def _get_word_split_index(tokens, st, end):
    split_idx = []
    i = st
    while i < end:
        if (not tokens[i].startswith('##')) or (i == st):
            split_idx.append(i)
        i += 1
    split_idx.append(end)
    return split_idx


def _expand_whole_word(tokens, st, end):
    new_st, new_end = st, end
    while (new_st >= 0) and tokens[new_st].startswith('##'):
        new_st -= 1
    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
        new_end += 1
    return new_st, new_end


# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.


def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    if len(tokens_a) + len(tokens_b) > max_len-3:
        while len(tokens_a) + len(tokens_b) > max_len-3:
            if len(tokens_a) > len(tokens_b):
                tokens_a = tokens_a[:-1]
            else:
                tokens_b = tokens_b[:-1]
    return tokens_a, tokens_b


def truncate_tokens_signle(tokens_a, max_len):
    if len(tokens_a) > max_len-2:
        tokens_a = tokens_a[:max_len-2]
    return tokens_a


from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class Seq2SeqDataset(torch.utils.data.Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, file, batch_size, tokenizer, max_len, 
            short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[]):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.max_len = max_len  # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order

        # read the file into memory
        self.ex_list = []
        # with open(file, "r", encoding='utf-8') as f:
        #     for i, line in enumerate(f):
        #         sample = eval(line.strip())
        #         src_tk = tokenizer.tokenize(sample["src_text"])
        #         tgt_tk = tokenizer.tokenize(sample["tgt_text"])
        #         assert len(src_tk) > 0
        #         assert len(tgt_tk) > 0
        #         self.ex_list.append((src_tk, tgt_tk))

        file_data = open(file, "r", encoding='utf-8')
        threads = min(8, cpu_count())
        with Pool(threads) as p:
            annotate_ = partial(
                self.read_data,
                tokenizer=self.tokenizer)
            self.ex_list = list(
                tqdm(
                    p.imap(annotate_, file_data.readlines(), chunksize=32),
                    total=len(file_data.readlines()),
                    desc="convert squad examples to features",
                )
            )
        # fin = open("look_new.json", "w",encoding="utf-8")
        # for jj, m in enumerate(self.ex_list):
        #     fin.write(str(jj)+"\t"+str(m)+"\n")
        print('Load {0} documents'.format(len(self.ex_list)))
        # exit()
    def read_data(self, line, tokenizer):
        sample = eval(line.strip())
        # src_tk = tokenizer.tokenize(sample["src_text"])
        # tgt_tk = tokenizer.tokenize(sample["tgt_text"])
        src_tk = sample["src_text"]
        tgt_tk = sample["tgt_text"]
        return (src_tk, tgt_tk)

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        # proc = choice(self.bi_uni_pipeline)
        new_instance = ()
        for proc in self.bi_uni_pipeline:
            new_instance += proc(instance)
        return new_instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list)-1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)
        

class Pipeline():
    """ Pre-process Pipeline Class : callable """

    def __init__(self):
        super().__init__()
        self.skipgram_prb = None
        self.skipgram_size = None
        self.pre_whole_word = None
        self.mask_whole_word = None
        self.vocab_words = None
        self.call_count = 0
        self.offline_mode = False
        self.skipgram_size_geo_list = None
        self.span_same_mask = False

    def init_skipgram_size_geo_list(self, p):
        if p > 0:
            g_list = []
            t = p
            for _ in range(self.skipgram_size):
                g_list.append(t)
                t *= (1-p)
            s = sum(g_list)
            self.skipgram_size_geo_list = [x/s for x in g_list]

    def __call__(self, instance):
        raise NotImplementedError

    # pre_whole_word: tokenize to words before masking
    # post whole word (--mask_whole_word): expand to words after masking
    def get_masked_pos(self, tokens, n_pred, add_skipgram=False, mask_segment=None, protect_range=None):
        if self.pre_whole_word:
            pre_word_split = _get_word_split_index(tokens, 0, len(tokens))
        else:
            pre_word_split = list(range(0, len(tokens)+1))

        span_list = list(zip(pre_word_split[:-1], pre_word_split[1:]))

        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()
        if mask_segment:
            for i, sp in enumerate(span_list):
                sp_st, sp_end = sp
                if (sp_end-sp_st == 1) and tokens[sp_st].endswith('SEP]'):
                    segment_index = i
                    break
        for i, sp in enumerate(span_list):
            sp_st, sp_end = sp
            if (sp_end-sp_st == 1) and (tokens[sp_st].endswith('CLS]') or tokens[sp_st].endswith('SEP]')):
                special_pos.add(i)
            else:
                if mask_segment:
                    if ((i < segment_index) and ('a' in mask_segment)) or ((i > segment_index) and ('b' in mask_segment)):
                        cand_pos.append(i)
                else:
                    cand_pos.append(i)
        shuffle(cand_pos)

        masked_pos = set()
        for i_span in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            cand_st, cand_end = span_list[i_span]
            if len(masked_pos)+cand_end-cand_st > n_pred:
                continue
            if any(p in masked_pos for p in range(cand_st, cand_end)):
                continue

            n_span = 1
            rand_skipgram_size = 0
            # ngram
            if self.skipgram_size_geo_list:
                # sampling ngram size from geometric distribution
                rand_skipgram_size = np.random.choice(
                    len(self.skipgram_size_geo_list), 1, p=self.skipgram_size_geo_list)[0] + 1
            else:
                if add_skipgram and (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                    rand_skipgram_size = min(
                        randint(2, self.skipgram_size), len(span_list)-i_span)
            for n in range(2, rand_skipgram_size+1):
                tail_st, tail_end = span_list[i_span+n-1]
                if (tail_end-tail_st == 1) and (tail_st in special_pos):
                    break
                if len(masked_pos)+tail_end-cand_st > n_pred:
                    break
                n_span = n
            st_span, end_span = i_span, i_span + n_span

            if self.mask_whole_word:
                # pre_whole_word==False: position index of span_list is the same as tokens
                st_span, end_span = _expand_whole_word(
                    tokens, st_span, end_span)

            skip_pos = None

            for sp in range(st_span, end_span):
                for mp in range(span_list[sp][0], span_list[sp][1]):
                    if not(skip_pos and (mp in skip_pos)) and (mp not in special_pos) and not(protect_range and (protect_range[0] <= mp < protect_range[1])):
                        masked_pos.add(mp)

        if len(masked_pos) < n_pred:
            shuffle(cand_pos)
            for pos in cand_pos:
                if len(masked_pos) >= n_pred:
                    break
                if pos not in masked_pos:
                    masked_pos.add(pos)
        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            # shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]
        return masked_pos

    def replace_masked_tokens(self, tokens, masked_pos):
        if self.span_same_mask:
            masked_pos = sorted(list(masked_pos))
        prev_pos, prev_rand = None, None
        for pos in masked_pos:
            if self.span_same_mask and (pos-1 == prev_pos):
                t_rand = prev_rand
            else:
                t_rand = rand()
            if t_rand < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif t_rand < 0.9:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
            prev_pos, prev_rand = pos, t_rand


class Preprocess4Seq2seq(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, max_pred, mask_prob, vocab_words, indexer, 
            max_len=512, skipgram_prb=0, skipgram_size=0, 
            mask_whole_word=False, mask_source_words=True, tokenizer=None):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        # tensor([[1, 0, 0,  ..., 0, 0, 0],
        # [1, 1, 0,  ..., 0, 0, 0],
        # [1, 1, 1,  ..., 0, 0, 0],
        # ...,
        # [1, 1, 1,  ..., 1, 0, 0],
        # [1, 1, 1,  ..., 1, 1, 0],
        # [1, 1, 1,  ..., 1, 1, 1]])
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.mask_source_words = mask_source_words
        self.tokenizer = tokenizer

    def __call__(self, instance):
        next_sentence_label = None
        tokens_a, tokens_b = instance[:2]
        # "从19:40开始至零时粗略统计有14轮广告播出，现场观众对过多的广告提出异议，甚至在第10轮广告后，每次播出广告全场都会嘘声一片，连主持人也不得不几次安抚观众，据浙江卫视人士介绍，广告创收已超过1亿元，而这个数字待最终结算完还将上涨。（新京报） \u200b\u200b\u200b"
        # "中国好声音4小时播出14轮广告 广告创收超亿元"
        tokens_a = self.tokenizer.tokenize(tokens_a)
        tokens_b = self.tokenizer.tokenize(tokens_b)
        # -3  for special tokens [CLS], [SEP], [SEP]
        tokens_a, tokens_b = truncate_tokens_pair(tokens_a, tokens_b, self.max_len)
        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        # ['[CLS]', '从', '19', ':', '40', '开', '始', '至', '零', '时', '粗', '略', '统', '计', '有', '14', '轮', '广', '告', '播', '出', '，', '现', '场', '观', '众', '对', '过', '多', '的', '广', '告', '提', '出', '异', '议', '，', '甚', '至', '在', '第', '10', '轮', '广', '告', '后', '，', '每', '次', '播', '出', '广', '告', '全', '场', '都', '会', '嘘', '声', '一', '片', '，', '连', '主', '持', '人', '也', '不', '得', '不', '几', '次', '安', '抚', '观', '众', '，', '据', '浙', '江', '卫', '视', '人', '士', '介', '绍', '，', '广', '告', '创', '收', '已', '超', '过', '1', '亿', '元', '，', '而', '这', '个', '数', '字', '待', '最', '终', '结', '算', '完', '还', '将', '上', '涨', '。', '（', '新', '京', '报', '）', '[SEP]', '中', '国', '好', '声', '音', '4', '小', '时', '播', '出', '14', '轮', '广', '告', '广', '告', '创', '收', '超', '亿', '元', '[SEP]']
        segment_ids = [4]*(len(tokens_a)+2) + [5]*(len(tokens_b)+1)
        # [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        
        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        effective_length = len(tokens_b)
        if self.mask_source_words:
            effective_length += len(tokens_a)
        n_pred = min(self.max_pred, max(1, int(round(effective_length*self.mask_prob))))
        
        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
                cand_pos.append(i)
            elif self.mask_source_words and (i < len(tokens_a)+2) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
                cand_pos.append(i)
            else:
                special_pos.add(i)
        shuffle(cand_pos)
        # [141, 125, 122, 135, 129, 132, 136, 139, 126, 133, 138, 134, 130, 121, 131, 124, 137, 127, 128, 120, 123, 140]

        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end
                while (new_st >= 0) and tokens[new_st].startswith('##'):
                    new_st -= 1
                while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                    new_end += 1
                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                # directly mask
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break
        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]
        # [125, 141, 122, 135]

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # ['4', '[SEP]', '好', '告']
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        masked_ids = self.indexer(masked_tokens)
        # [125, 102, 1962, 1440]
        # Token Indexing
        input_ids = self.indexer(tokens)
        # [101, 794, 8131, 131, 8164, 2458, 1993, 5635, 7439, 3198, 5110, 4526, 5320, 6369, 3300, 8122, 6762, 2408, 1440, 3064, 1139, 8024, 4385, 1767, 6225, 830, 2190, 6814, 1914, 4638, 2408, 1440, 2990, 1139, 2460, 6379, 8024, 4493, 5635, 1762, 5018, 8108, 6762, 2408, 1440, 1400, 8024, 3680, 3613, 3064, 1139, 2408, 1440, 1059, 1767, 6963, 833, 1656, 1898, 671, 4275, 8024, 6825, 712, 2898, 782, 738, 679, 2533, 679, 1126, 3613, 2128, 2836, 6225, 830, 8024, 2945, 3851, 3736, 1310, 6228, 782, 1894, 792, 5305, 8024, 2408, 1440, 1158, 3119, 2347, 6631, 6814, 122, 783, 1039, 8024, 5445, 6821, 702, 3144, 2099, 2521, 3297, 5303, 5310, 5050, 2130, 6820, 2199, 677, 3885, 511, 8020, 3173, 776, 2845, 8021, 102, 704, 1744, 103, 1898, 7509, 103, 2207, 3198, 3064, 1139, 8122, 6762, 2408, 1440, 2408, 103, 1158, 3119, 6631, 783, 1039, 103]

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        input_mask[:, :len(tokens_a) + 2].fill_(1)      # tokens_a 双向注意力编码
        second_st, second_end = len(tokens_a) + 2, len(tokens_a) + len(tokens_b) + 3    # 120, 142
        input_mask[second_st: second_end, second_st: second_end].copy_(
            self._tril_matrix[: second_end - second_st, : second_end - second_st])
        # tensor([[1, 1, 1,  ..., 0, 0, 0],
        # [1, 1, 1,  ..., 0, 0, 0],
        # [1, 1, 1,  ..., 0, 0, 0],
        # ...,
        # [1, 1, 1,  ..., 0, 0, 0],
        # [1, 1, 1,  ..., 0, 0, 0],
        # [1, 1, 1,  ..., 0, 0, 0]])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, next_sentence_label)


class Preprocess4Seq2seqDecode(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, max_tgt_length=128):
        super().__init__()
        self.max_len = max_len
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        # tensor([[1, 0, 0,  ..., 0, 0, 0],
        # [1, 1, 0,  ..., 0, 0, 0],
        # [1, 1, 1,  ..., 0, 0, 0],
        # ...,
        # [1, 1, 1,  ..., 1, 0, 0],
        # [1, 1, 1,  ..., 1, 1, 0],
        # [1, 1, 1,  ..., 1, 1, 1]])
        self.max_tgt_length = max_tgt_length

    def __call__(self, instance):
        tokens_a, max_len_in_batch = instance  # 382

        # Add Special Tokens
        padded_tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        assert len(padded_tokens_a) <= max_len_in_batch + 2

        if max_len_in_batch + 2 > len(padded_tokens_a):
            padded_tokens_a += ['[PAD]'] * \
                (max_len_in_batch + 2 - len(padded_tokens_a))
        assert len(padded_tokens_a) == max_len_in_batch + 2
        
        max_len = min(self.max_tgt_length + max_len_in_batch + 2, self.max_len)   # 512
        tokens = padded_tokens_a
        # ['[CLS]', '{', "'", 'sr', '##c', '_', 'text', "'", ':', "'", '这', '个', '周', '末', '，', '无', '数', '人', '的', '手', '机', '被', '一', '篇', '《', '每', '对', '母', '子', '都', '是', '生', '死', '之', '交', '》', '微', '信', '文', '章', '刷', '了', '屏', '。', '[UNK]', '十', '年', '前', '的', '今', '天', '，', '我', '拼', '着', '命', '生', '下', '了', '儿', '子', '；', '十', '年', '前', '的', '今', '天', '，', '儿', '子', '拼', '着', '命', '来', '到', '我', '身', '边', '。', '每', '对', '母', '子', '都', '是', '这', '样', '拼', '着', '命', '才', '能', '相', '见', '，', '可', '是', '我', '却', '没', '有', '保', '护', '好', '他', '。', '[UNK]', '这', '样', '的', '文', '字', '让', '每', '一', '个', '为', '人', '父', '母', '者', '读', '来', '动', '容', '。', '虽', '然', '事', '情', '的', '具', '体', '细', '节', '还', '在', '核', '实', '，', '但', '校', '园', '欺', '凌', '的', '话', '题', '，', '再', '一', '次', '引', '发', '了', '社', '会', '的', '集', '中', '关', '注', '。', '《', '关', '于', '防', '治', '中', '小', '学', '生', '欺', '凌', '和', '暴', '力', '的', '指', '导', '意', '见', '》', '发', '布', '还', '不', '到', '一', '个', '月', '，', '真', '正', '发', '挥', '作', '用', '还', '需', '要', '各', '方', '面', '认', '真', '消', '化', '落', '实', '。', '校', '园', '欺', '凌', '，', '不', '能', '只', '是', '一', '个', '[UNK]', '开', '过', '分', '了', '的', '玩', '笑', '[UNK]', '。', '目', '前', '可', '以', '说', '，', '事', '件', '的', '有', '效', '预', '防', '，', '事', '件', '发', '生', '时', '的', '及', '时', '、', '妥', '善', '处', '理', '，', '事', '件', '发', '生', '后', '的', '惩', '戒', '和', '科', '学', '教', '育', '，', '都', '还', '十', '分', '缺', '乏', '。', '学', '校', '及', '整', '个', '社', '会', '对', '于', '校', '园', '欺', '凌', '的', '危', '害', '性', '和', '应', '对', '方', '法', '，', '亟', '待', '在', '深', '层', '认', '知', '上', '提', '高', '。', '避', '免', '校', '园', '欺', '凌', '，', '老', '师', '、', '家', '长', '与', '学', '生', '之', '间', '，', '从', '内', '心', '深', '处', '尊', '重', '彼', '此', '、', '珍', '视', '彼', '此', '，', '校', '园', '才', '会', '真', '正', '成', '为', '被', '美', '好', '和', '希', '望', '浸', '润', '的', '地', '方', '。', '[UNK]', '人', '民', '日', '报', '评', '论', '[UNK]', '微', '信', '公', '号', '人', '民', '[SEP]']
        segment_ids = [4]*(len(padded_tokens_a)) + [5]*(max_len - len(padded_tokens_a))
        # [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

        position_ids = []
        for i in range(len(tokens_a) + 2):
            position_ids.append(i)
        for i in range(len(tokens_a) + 2, max_len_in_batch + 2):
            position_ids.append(0)          # 超过tokens_a长度，但不足批次长度填充0
        for i in range(max_len_in_batch + 2, max_len):
            position_ids.append(i + len(tokens_a) + 2 - (max_len_in_batch + 2))
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511]
        
        # Token Indexing
        input_ids = self.indexer(tokens)
        # [101, 169, 112, 12109, 8177, 142, 10539, 112, 131, 112, 6821, 702, 1453, 3314, 8024, 3187, 3144, 782, 4638, 2797, 3322, 6158, 671, 5063, 517, 3680, 2190, 3678, 2094, 6963, 3221, 4495, 3647, 722, 769, 518, 2544, 928, 3152, 4995, 1170, 749, 2242, 511, 100, 1282, 2399, 1184, 4638, 791, 1921, 8024, 2769, 2894, 4708, 1462, 4495, 678, 749, 1036, 2094, 8039, 1282, 2399, 1184, 4638, 791, 1921, 8024, 1036, 2094, 2894, 4708, 1462, 3341, 1168, 2769, 6716, 6804, 511, 3680, 2190, 3678, 2094, 6963, 3221, 6821, 3416, 2894, 4708, 1462, 2798, 5543, 4685, 6224, 8024, 1377, 3221, 2769, 1316, 3766, 3300, 924, 2844, 1962, 800, 511, 100, 6821, 3416, 4638, 3152, 2099, 6375, 3680, 671, 702, 711, 782, 4266, 3678, 5442, 6438, 3341, 1220, 2159, 511, 6006, 4197, 752, 2658, 4638, 1072, 860, 5301, 5688, 6820, 1762, 3417, 2141, 8024, 852, 3413, 1736, 3619, 1119, 4638, 6413, 7579, 8024, 1086, 671, 3613, 2471, 1355, 749, 4852, 833, 4638, 7415, 704, 1068, 3800, 511, 517, 1068, 754, 7344, 3780, 704, 2207, 2110, 4495, 3619, 1119, 1469, 3274, 1213, 4638, 2900, 2193, 2692, 6224, 518, 1355, 2357, 6820, 679, 1168, 671, 702, 3299, 8024, 4696, 3633, 1355, 2916, 868, 4500, 6820, 7444, 6206, 1392, 3175, 7481, 6371, 4696, 3867, 1265, 5862, 2141, 511, 3413, 1736, 3619, 1119, 8024, 679, 5543, 1372, 3221, 671, 702, 100, 2458, 6814, 1146, 749, 4638, 4381, 5010, 100, 511, 4680, 1184, 1377, 809, 6432, 8024, 752, 816, 4638, 3300, 3126, 7564, 7344, 8024, 752, 816, 1355, 4495, 3198, 4638, 1350, 3198, 510, 1980, 1587, 1905, 4415, 8024, 752, 816, 1355, 4495, 1400, 4638, 2674, 2770, 1469, 4906, 2110, 3136, 5509, 8024, 6963, 6820, 1282, 1146, 5375, 726, 511, 2110, 3413, 1350, 3146, 702, 4852, 833, 2190, 754, 3413, 1736, 3619, 1119, 4638, 1314, 2154, 2595, 1469, 2418, 2190, 3175, 3791, 8024, 766, 2521, 1762, 3918, 2231, 6371, 4761, 677, 2990, 7770, 511, 6912, 1048, 3413, 1736, 3619, 1119, 8024, 5439, 2360, 510, 2157, 7270, 680, 2110, 4495, 722, 7313, 8024, 794, 1079, 2552, 3918, 1905, 2203, 7028, 2516, 3634, 510, 4397, 6228, 2516, 3634, 8024, 3413, 1736, 2798, 833, 4696, 3633, 2768, 711, 6158, 5401, 1962, 1469, 2361, 3307, 3863, 3883, 4638, 1765, 3175, 511, 100, 782, 3696, 3189, 2845, 6397, 6389, 100, 2544, 928, 1062, 1384, 782, 3696, 102]

        # Zero Padding
        input_mask = torch.zeros(
            max_len, max_len, dtype=torch.long)
        input_mask[:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(padded_tokens_a), max_len

        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])
            
        return (input_ids, segment_ids, position_ids, input_mask)


class Preprocess4BiLM(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, max_pred, mask_prob, vocab_words, indexer, 
            max_len=512, skipgram_prb=0, skipgram_size=0, 
            mask_whole_word=False, mask_source_words=True, tokenizer=None):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        # tensor([[1, 0, 0,  ..., 0, 0, 0],
        # [1, 1, 0,  ..., 0, 0, 0],
        # [1, 1, 1,  ..., 0, 0, 0],
        # ...,
        # [1, 1, 1,  ..., 1, 0, 0],
        # [1, 1, 1,  ..., 1, 1, 0],
        # [1, 1, 1,  ..., 1, 1, 1]])
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.mask_source_words = mask_source_words
        self.tokenizer = tokenizer

    def __call__(self, instance):
        tokens_a, tokens_b = instance[:2]
        if rand() <= 0.5:
            next_sentence_label = 1.0
        else:
            tokens_a, tokens_b = tokens_b, tokens_a
            next_sentence_label = 0.0

        tokens_a = self.tokenizer.tokenize(tokens_a)
        tokens_b = self.tokenizer.tokenize(tokens_b)
        # -3  for special tokens [CLS], [SEP], [SEP]
        tokens_a, tokens_b = truncate_tokens_pair(tokens_a, tokens_b, self.max_len)
        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        effective_length = len(tokens_b)
        if self.mask_source_words:
            effective_length += len(tokens_a)
        n_pred = min(self.max_pred, max(
            1, int(round(effective_length*self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
                cand_pos.append(i)
            elif self.mask_source_words and (i < len(tokens_a)+2) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
                cand_pos.append(i)
            else:
                special_pos.add(i)
        shuffle(cand_pos)

        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end
                while (new_st >= 0) and tokens[new_st].startswith('##'):
                    new_st -= 1
                while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                    new_end += 1
                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                # directly mask
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        masked_ids = self.indexer(masked_tokens)
        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        input_mask = torch.ones(self.max_len, self.max_len, dtype=torch.long)
        # input_mask[:, :len(tokens_a)+2].fill_(1)
        # second_st, second_end = len(
        #     tokens_a)+2, len(tokens_a)+len(tokens_b)+3
        # input_mask[second_st:second_end, second_st:second_end].copy_(
        #     self._tril_matrix[:second_end-second_st, :second_end-second_st])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)
                
        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, next_sentence_label)


class Preprocess4RightLM(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, max_pred, mask_prob, vocab_words, indexer, 
            max_len=512, skipgram_prb=0, skipgram_size=0, 
            mask_whole_word=False, mask_source_words=True, tokenizer=None):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.mask_source_words = mask_source_words
        self.tokenizer = tokenizer

    def __call__(self, instance):
        next_sentence_label = None
        tokens_a, _ = instance[:2]
        tokens_a = self.tokenizer.tokenize(tokens_a)
        tokens_a = truncate_tokens_signle(tokens_a, self.max_len)
        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [2]*(len(tokens_a)+2)

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        effective_length = 0
        if self.mask_source_words:
            effective_length += len(tokens_a)
        n_pred = min(self.max_pred, max(
            1, int(round(effective_length*self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            # if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
            #     cand_pos.append(i)
            if (tk != '[CLS]') and (tk != '[SEP]'):
                cand_pos.append(i)
            else:
                special_pos.add(i)

        shuffle(cand_pos)

        masked_pos = set()

        try:
            max_cand_pos = max(cand_pos)
        except:
            max_cand_pos = 0

        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end
                while (new_st >= 0) and tokens[new_st].startswith('##'):
                    new_st -= 1
                while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                    new_end += 1
                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                # directly mask
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        masked_ids = self.indexer(masked_tokens)
        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        input_mask = torch.ones(self.max_len, self.max_len, dtype=torch.long)
        # input_mask[:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = 0, len(tokens_a)+2
        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, next_sentence_label)


class Preprocess4LeftLM(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0, mask_whole_word=False, mask_source_words=True, tokenizer=None):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self._tril_matrix = torch.triu(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.mask_source_words = mask_source_words
        self.tokenizer = tokenizer

    def __call__(self, instance):
        next_sentence_label = None
        tokens_a, _ = instance[:2]

        tokens_a = self.tokenizer.tokenize(tokens_a)
        tokens_a = truncate_tokens_signle(tokens_a, self.max_len)
        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']

        segment_ids = [3]*(len(tokens_a)+2)

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        effective_length = 0
        if self.mask_source_words:
            effective_length += len(tokens_a)
        n_pred = min(self.max_pred, max(
            1, int(round(effective_length*self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            # if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
            #     cand_pos.append(i)
            if (tk != '[CLS]') and (tk != '[SEP]'):
                cand_pos.append(i)
            else:
                special_pos.add(i)

        shuffle(cand_pos)

        masked_pos = set()

        try:
            max_cand_pos = max(cand_pos)
        except:
            max_cand_pos = 0

        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end
                while (new_st >= 0) and tokens[new_st].startswith('##'):
                    new_st -= 1
                while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                    new_end += 1
                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                # directly mask
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        masked_ids = self.indexer(masked_tokens)
        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        input_mask = torch.ones(self.max_len, self.max_len, dtype=torch.long)
        # input_mask[:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = 0, len(tokens_a)+2
        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, next_sentence_label)
