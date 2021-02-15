import collections

import tensorflow as tf
from prepro_utils import preprocess_text, partial, encode_ids, encode_pieces
import sentencepiece as spm

SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4

special_symbols = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "<cls>": 3,
    "<sep>": 4,
    "<pad>": 5,
    "<mask>": 6,
    "<eod>": 7,
    "<eop>": 8,
}

UNK_ID = special_symbols["<unk>"]
CLS_ID = special_symbols["<cls>"]
SEP_ID = special_symbols["<sep>"]
MASK_ID = special_symbols["<mask>"]
EOD_ID = special_symbols["<eod>"]


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 label_x_id,
                 label_gather,
                 label_mask_x,
                 label_mask_gather,
                 label_index,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.label_x_id = label_x_id
        self.label_gather = label_gather
        self.label_mask_x = label_mask_x
        self.label_mask_gather = label_mask_gather
        self.label_index = label_index
        self.is_real_example = is_real_example


def process_seq(words, labels, sp, x, lower=True):
    assert len(words) == len(labels)
    prepro_func = partial(preprocess_text, lower=lower)

    tokens = []
    label = []
    label_x = []
    is_start_token = []
    for i in range(len(words)):
        t = encode_ids(sp, prepro_func(words[i]))
        tokens.extend(t)
        label.extend([int(labels[i])] * len(t))
        label_x.append(int(labels[i]))
        is_start_token.append(1)
        for _ in range(len(t) - 1):
            label_x.append(x)
            is_start_token.append(0)

    return tokens, label, label_x, is_start_token


def get_data(input_file, max_seq_length, sp, encoder, lower=True):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        seqs = []
        for line in lines:
            line = line.strip()
            if line:
                line = line.split("\t")
                if seqs:
                    seqs[-1].append((line[0], encoder[line[1]] if line[1] in encoder.keys() else 0))
                else:
                    seqs.append([(line[0], encoder[line[1]] if line[1] in encoder.keys() else 0)])
            else:
                seqs.append([])

        tokens = []
        label = []
        label_x = []
        is_start_token = []
        for s in seqs:
            if not s:
                continue
            ws, ls = zip(*s)
            t, l, lx, ist = process_seq(ws, ls, sp, encoder["X"], lower)
            if len(t) > max_seq_length:
                yield tokens, label, label_x, is_start_token
                tokens = []
                label = []
                label_x = []
                is_start_token = []
                t = [t[i:i + max_seq_length] for i in range(0, len(t), max_seq_length)]
                l = [l[i:i + max_seq_length] for i in range(0, len(l), max_seq_length)]
                lx = [lx[i:i + max_seq_length] for i in range(0, len(lx), max_seq_length)]
                ist = [ist[i:i + max_seq_length] for i in range(0, len(ist), max_seq_length)]
                z = zip(t, l, lx, ist)
                for i in z:
                    yield i
                continue

            if len(t) + len(tokens) > max_seq_length:
                yield tokens, label, label_x, is_start_token
                tokens = t
                label = l
                label_x = lx
                is_start_token = ist
            else:
                tokens.extend(t)
                label.extend(l)
                label_x.extend(lx)
                is_start_token.extend(ist)

        if tokens:
            yield tokens, label, label_x, is_start_token


def single_example(tokens, labels, labels_x, is_start_token, max_length):
    tokens_length = len(tokens)
    input_ids = []
    input_mask = []
    segment_ids = []
    label_id = []
    label_x_id = []
    label_gather = []
    label_mask_x = []
    label_mask_gather = []
    label_index = []

    i = 0
    for s in is_start_token:
        if s:
            label_index.append(i)
            label_gather.append(labels[i])
            label_mask_gather.append(1)
        i += 1
    for _ in range(max_length - len(label_gather)):
        label_gather.append(0)

    label_index.extend([i, i + 1, i + 2])
    label_mask_gather.extend([1, 1, 1])
    for _ in range(max_length - len(label_mask_gather)):
        label_mask_gather.append(0)
        label_index.append(0)

    input_ids.extend(tokens)
    input_mask.extend([0] * tokens_length)
    segment_ids.extend([SEG_ID_A] * tokens_length)
    label_id.extend(labels)
    label_x_id.extend(labels_x)
    label_mask_x.extend(is_start_token)

    input_ids.extend([SEP_ID, SEP_ID, CLS_ID])
    input_mask.extend([0, 0, 0])
    segment_ids.extend([SEG_ID_A, SEG_ID_B, SEG_ID_CLS])
    label_id.extend([0, 0, 0])
    label_x_id.extend([0, 0, 0])
    label_mask_x.extend([1, 1, 1])

    for _ in range(max_length - tokens_length - 3):
        input_ids.append(0)
        input_mask.append(1)
        segment_ids.append(SEG_ID_PAD)
        label_id.append(0)
        label_x_id.append(0)
        label_mask_x.append(0)

    assert len(input_ids) == max_length
    assert len(input_mask) == max_length
    assert len(segment_ids) == max_length
    assert len(label_id) == max_length
    assert len(label_x_id) == max_length
    assert len(label_mask_gather) == max_length

    return InputFeatures(input_ids, input_mask, segment_ids, label_id, label_x_id, label_gather, label_mask_x,
                         label_mask_gather, label_index)


def file_based_convert_examples_to_features(examples, output_file):
    tf.logging.info("Start writing tfrecord %s.", output_file)
    writer = tf.python_io.TFRecordWriter(output_file)
    tf.logging.info("totle %d examples", len(examples))
    for ex_index, example in enumerate(examples):
        if ex_index % 100 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(example.input_ids)
        features["input_mask"] = create_float_feature(example.input_mask)
        features["segment_ids"] = create_int_feature(example.segment_ids)
        features["label_ids"] = create_int_feature(example.label_id)
        features["label_x_id"] = create_int_feature(example.label_x_id)
        features["label_gather"] = create_int_feature(example.label_gather)
        features["label_mask_x"] = create_float_feature(example.label_mask_x)
        features["label_mask_gather"] = create_float_feature(example.label_mask_gather)
        features["label_index"] = create_int_feature(example.label_index)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    tf.logging.info("write finish!")
    writer.close()


def convert_tsv_to_tfrecord(input_file, output_file, max_seq_length, sp_model, encoder, lower):
    tf.logging.set_verbosity(tf.logging.INFO)
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model)
    examples = []
    for data in get_data(input_file, max_seq_length - 3, sp, encoder, lower):
        examples.append(single_example(*data, max_length=max_seq_length))

    if "train" in input_file:
        tokens = 0
        words = 0
        first_word = 0
        prepro_func = partial(preprocess_text, lower=lower)
        with open(input_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    line = line.split("\t")
                    if line[-1] != "O":
                        words += 1
                        pieces = encode_pieces(sp, prepro_func(line[0]))
                        tokens += len(pieces)
                        first_word += len(pieces[0])
        print("{} {} {} {}".format(input_file, tokens, words, first_word))

    file_based_convert_examples_to_features(examples, output_file)
    return len(examples)


if __name__ == '__main__':
    m = {"O": 0, "B-Chemical": 1, "I-Chemical": 2, "E-Chemical": 3, "S-Chemical": 4, "B-Disease": 5,
                    "I-Disease": 6, "E-Disease": 7, "S-Disease": 8, "X": 9}
    convert_tsv_to_tfrecord("data/BC5CDR-IOBES/train.tsv", "cache/train.tfrecord", 512,
                            "E:/tfhub-module/xlnet_cased_L-12_H-768_A-12/spiece.model", m, True)

    # sp = spm.SentencePieceProcessor()
    # sp.load("E:/tfhub-module/xlnet_cased_L-12_H-768_A-12/spiece.model")
    # prepro_func = partial(preprocess_text, lower=True)
    # s = [17, 23, 6159, 3141, 814, 17, 13, 17, 12674, 701, 9323, 11581, 23157, 25, 2133, 153, 672, 17, 26, 17, 23, 1487, 17]
    # print(sp.decode_ids(s))
    # for i in s:
    #     print(sp.decode_ids([i]))
