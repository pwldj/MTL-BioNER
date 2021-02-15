import create_data
import os
import tensorflow as tf


class DataProcessor:
    def __init__(self, sp, lower, input_dir, cache_dir, max_seq_length):
        self.train_step = 4500
        self.sp = sp
        self.lower = lower
        self.input_dir = input_dir
        self.cache_dir = cache_dir
        self.max_seq_length = max_seq_length
        self.task_name = ""
        self.map = dict()

    def create_tfrecord(self):
        if not tf.io.gfile.exists(self.get_path("train")):
            create_data.convert_tsv_to_tfrecord(os.path.join(self.input_dir, self.task_name, "train.tsv"),
                                                self.get_path("train"),
                                                self.max_seq_length, self.sp, self.map, self.lower)
        if not tf.io.gfile.exists(self.get_path("eval")):
            create_data.convert_tsv_to_tfrecord(os.path.join(self.input_dir, self.task_name, "devel.tsv"),
                                                self.get_path("eval"),
                                                self.max_seq_length, self.sp, self.map, self.lower)
        if not tf.io.gfile.exists(self.get_path("test")):
            create_data.convert_tsv_to_tfrecord(os.path.join(self.input_dir, self.task_name, "test.tsv"),
                                                self.get_path("test"),
                                                self.max_seq_length, self.sp, self.map, self.lower)

    def get_path(self, mode):
        return os.path.join(self.cache_dir, "{}_{}_{}.tfrecord".format(self.task_name, mode, self.max_seq_length))

    def get_train_data(self):
        return self.get_path("train")

    def get_dev_data(self):
        return self.get_path("eval")

    def get_test_data(self):
        return self.get_path("test")

    def get_train_stap(self):
        c = 0
        for record in tf.python_io.tf_record_iterator(self.get_path("train")):
            c += 1
        return ((c * 7) // 300 + 1) * 300


class Conll2003Processor(DataProcessor):
    def __init__(self, sp, lower, input_dir, cache_dir, max_seq_length):
        super(Conll2003Processor, self).__init__(sp, lower, input_dir, cache_dir, max_seq_length)
        self.task_name = "CoNLL-2003"
        self.map = {'O': 0, 'B-ORG': 1, 'I-ORG': 2, 'B-MISC': 3, 'I-MISC': 4, 'B-PER': 5, 'I-PER': 6, 'B-LOC': 7,
                    'I-LOC': 8, "X": 9}
        self.decode_map = {value: key for key, value in self.map.items()}
        self.classes = len(self.map) - 1
        self.create_tfrecord()


class BC5CDRProcessor(DataProcessor):
    def __init__(self, sp, lower, input_dir, cache_dir, max_seq_length):
        super(BC5CDRProcessor, self).__init__(sp, lower, input_dir, cache_dir, max_seq_length)
        self.task_name = "BC5CDR-IOBES"
        self.map = {"O": 0, "B-Chemical": 1, "I-Chemical": 2, "E-Chemical": 3, "S-Chemical": 4, "B-Disease": 5,
                    "I-Disease": 6, "E-Disease": 7, "S-Disease": 8, "X": 9}
        self.classes = len(self.map) - 1
        self.decode_map = {value: key for key, value in self.map.items()}
        self.create_tfrecord()


class AnatEMProcessor(DataProcessor):
    def __init__(self, sp, lower, input_dir, cache_dir, max_seq_length):
        super(AnatEMProcessor, self).__init__(sp, lower, input_dir, cache_dir, max_seq_length)
        self.task_name = "AnatEM-IOBES"
        self.map = {'O': 0, 'B-Anatomy': 1, 'I-Anatomy': 2, 'E-Anatomy': 3, 'S-Anatomy': 4, "X": 5}
        self.classes = len(self.map) - 1
        self.decode_map = {value: key for key, value in self.map.items()}
        self.create_tfrecord()


class BC2GMProcessor(DataProcessor):
    def __init__(self, sp, lower, input_dir, cache_dir, max_seq_length):
        super(BC2GMProcessor, self).__init__(sp, lower, input_dir, cache_dir, max_seq_length)
        self.task_name = "BC2GM-IOBES"
        self.map = {'O': 0, 'B-GENE': 1, 'I-GENE': 2, 'E-GENE': 3, 'S-GENE': 4, "X": 5}
        self.classes = len(self.map) - 1
        self.decode_map = {value: key for key, value in self.map.items()}
        self.create_tfrecord()


class BC4CHEMDProcessor(DataProcessor):
    def __init__(self, sp, lower, input_dir, cache_dir, max_seq_length):
        super(BC4CHEMDProcessor, self).__init__(sp, lower, input_dir, cache_dir, max_seq_length)
        self.task_name = "BC4CHEMD-IOBES"
        self.map = {'O': 0, 'B-Chemical': 1, 'I-Chemical': 2, 'E-Chemical': 3, 'S-Chemical': 4, "X": 5}
        self.classes = len(self.map) - 1
        self.decode_map = {value: key for key, value in self.map.items()}
        self.create_tfrecord()


class BioNLP09Processor(DataProcessor):
    def __init__(self, sp, lower, input_dir, cache_dir, max_seq_length):
        super(BioNLP09Processor, self).__init__(sp, lower, input_dir, cache_dir, max_seq_length)
        self.task_name = "BioNLP09-IOBES"
        self.map = {'O': 0, 'B-Protein': 1, 'I-Protein': 2, 'E-Protein': 3, 'S-Protein': 4, "X": 5}
        self.classes = len(self.map) - 1
        self.decode_map = {value: key for key, value in self.map.items()}
        self.create_tfrecord()


class BioNLP11IDProcessor(DataProcessor):
    def __init__(self, sp, lower, input_dir, cache_dir, max_seq_length):
        super(BioNLP11IDProcessor, self).__init__(sp, lower, input_dir, cache_dir, max_seq_length)
        self.task_name = "BioNLP11ID-IOBES"
        self.map = {'O': 0, 'B-Chemical': 1, 'I-Chemical': 2, 'E-Chemical': 3, 'S-Chemical': 4,
                    'B-Protein': 5, 'I-Protein': 6, 'E-Protein': 7, 'S-Protein': 8,
                    'B-Organism': 9, 'I-Organism': 10, 'E-Organism': 11, 'S-Organism': 12,
                    'B-Regulon_operon': 13, 'I-Regulon_operon': 14, 'E-Regulon_operon': 15, 'S-Regulon_operon': 16,
                    "X": 17}

        self.classes = len(self.map) - 1
        self.decode_map = {value: key for key, value in self.map.items()}
        self.create_tfrecord()


class BioNLP11EPIProcessor(DataProcessor):
    def __init__(self, sp, lower, input_dir, cache_dir, max_seq_length):
        super(BioNLP11EPIProcessor, self).__init__(sp, lower, input_dir, cache_dir, max_seq_length)
        self.task_name = "BioNLP11EPI-IOBES"
        self.map = {'O': 0, 'B-Protein': 1, 'I-Protein': 2, 'E-Protein': 3, 'S-Protein': 4, "X": 5}
        self.classes = len(self.map) - 1
        self.decode_map = {value: key for key, value in self.map.items()}
        self.create_tfrecord()


class BioNLP13CGProcessor(DataProcessor):
    def __init__(self, sp, lower, input_dir, cache_dir, max_seq_length):
        super(BioNLP13CGProcessor, self).__init__(sp, lower, input_dir, cache_dir, max_seq_length)
        self.task_name = "BioNLP13CG-IOBES"
        self.map = {'O': 0, 'B-Organ': 1, 'I-Organ': 2, 'E-Organ': 3, 'S-Organ': 4,
                    'B-Organism_substance': 5, 'I-Organism_substance': 6, 'E-Organism_substance': 7,
                    'S-Organism_substance': 8,
                    'B-Organism': 9, 'I-Organism': 10, 'E-Organism': 11, 'S-Organism': 12,
                    'B-Organism_subdivision': 13, 'I-Organism_subdivision': 14, 'E-Organism_subdivision': 15,
                    'S-Organism_subdivision': 16,
                    'B-Simple_chemical': 17, 'I-Simple_chemical': 18, 'E-Simple_chemical': 19, 'S-Simple_chemical': 20,
                    'B-Developing_anatomical_structure': 21, 'I-Developing_anatomical_structure': 22,
                    'E-Developing_anatomical_structure': 23, 'S-Developing_anatomical_structure': 24,
                    'B-Cellular_component': 25, 'I-Cellular_component': 26, 'E-Cellular_component': 27,
                    'S-Cellular_component': 28,
                    'B-Anatomical_system': 29, 'I-Anatomical_system': 30, 'E-Anatomical_system': 31,
                    'S-Anatomical_system': 32,
                    'B-Gene_or_gene_product': 33, 'I-Gene_or_gene_product': 34, 'E-Gene_or_gene_product': 35,
                    'S-Gene_or_gene_product': 36,
                    'B-Pathological_formation': 37, 'I-Pathological_formation': 38, 'E-Pathological_formation': 39,
                    'S-Pathological_formation': 40,
                    'B-Amino_acid': 41, 'I-Amino_acid': 42, 'E-Amino_acid': 43, 'S-Amino_acid': 44,
                    'B-Cancer': 45, 'I-Cancer': 46, 'E-Cancer': 47, 'S-Cancer': 48,
                    'B-Multi_tissue_structure': 49, 'I-Multi_tissue_structure': 50, 'E-Multi_tissue_structure': 51,
                    'S-Multi_tissue_structure': 52,
                    'B-Cell': 53, 'I-Cell': 54, 'E-Cell': 55, 'S-Cell': 56,
                    'B-Tissue': 57, 'I-Tissue': 58, 'E-Tissue': 59, 'S-Tissue': 60,
                    'B-Immaterial_anatomical_entity': 61, 'I-Immaterial_anatomical_entity': 62,
                    'E-Immaterial_anatomical_entity': 63, 'S-Immaterial_anatomical_entity': 64,
                    'X': 65}

        self.classes = len(self.map) - 1
        self.decode_map = {value: key for key, value in self.map.items()}
        self.create_tfrecord()


class BioNLP13GEProcessor(DataProcessor):
    def __init__(self, sp, lower, input_dir, cache_dir, max_seq_length):
        super(BioNLP13GEProcessor, self).__init__(sp, lower, input_dir, cache_dir, max_seq_length)
        self.task_name = "BioNLP13GE-IOBES"
        self.map = {'O': 0, 'B-Protein': 1, 'I-Protein': 2, 'E-Protein': 3, 'S-Protein': 4, "X": 5}
        self.classes = len(self.map) - 1
        self.decode_map = {value: key for key, value in self.map.items()}
        self.create_tfrecord()


class BioNLP13PCProcessor(DataProcessor):
    def __init__(self, sp, lower, input_dir, cache_dir, max_seq_length):
        super(BioNLP13PCProcessor, self).__init__(sp, lower, input_dir, cache_dir, max_seq_length)
        self.task_name = "BioNLP13PC-IOBES"
        self.map = {'O': 0,
                    'B-Gene_or_gene_product': 1, 'I-Gene_or_gene_product': 2, 'E-Gene_or_gene_product': 3,
                    'S-Gene_or_gene_product': 4,
                    'B-Complex': 5, 'I-Complex': 6, 'E-Complex': 7, 'S-Complex': 8,
                    'B-Cellular_component': 9, 'I-Cellular_component': 10, 'E-Cellular_component': 11,
                    'S-Cellular_component': 12,
                    'B-Simple_chemical': 13, 'I-Simple_chemical': 14, 'E-Simple_chemical': 15, 'S-Simple_chemical': 16,
                    "X": 17}

        self.classes = len(self.map) - 1
        self.decode_map = {value: key for key, value in self.map.items()}
        self.create_tfrecord()


class CRAFTProcessor(DataProcessor):
    def __init__(self, sp, lower, input_dir, cache_dir, max_seq_length):
        super(CRAFTProcessor, self).__init__(sp, lower, input_dir, cache_dir, max_seq_length)
        self.task_name = "CRAFT-IOBES"
        self.map = {'O': 0, 'B-CHEBI': 1, 'I-CHEBI': 2, 'E-CHEBI': 3, 'S-CHEBI': 4,
                    'B-GO': 5, 'I-GO': 6, 'E-GO': 7, 'S-GO': 8,
                    'B-SO': 9, 'I-SO': 10, 'E-SO': 11, 'S-SO': 12,
                    'B-Taxon': 13, 'I-Taxon': 14, 'E-Taxon': 15, 'S-Taxon': 16,
                    'B-GGP': 17, 'I-GGP': 18, 'E-GGP': 19, 'S-GGP': 20,
                    'B-CL': 21, 'I-CL': 22, 'E-CL': 23, 'S-CL': 24, 'X': 25}

        self.classes = len(self.map) - 1
        self.decode_map = {value: key for key, value in self.map.items()}
        self.create_tfrecord()


class ExPTMProcessor(DataProcessor):
    def __init__(self, sp, lower, input_dir, cache_dir, max_seq_length):
        super(ExPTMProcessor, self).__init__(sp, lower, input_dir, cache_dir, max_seq_length)
        self.task_name = "Ex-PTM-IOBES"
        self.map = {'O': 0, 'B-Protein': 1, 'I-Protein': 2, 'E-Protein': 3, 'S-Protein': 4, 'X': 5}

        self.classes = len(self.map) - 1
        self.decode_map = {value: key for key, value in self.map.items()}
        self.create_tfrecord()


class JNLPBAProcessor(DataProcessor):
    def __init__(self, sp, lower, input_dir, cache_dir, max_seq_length):
        super(JNLPBAProcessor, self).__init__(sp, lower, input_dir, cache_dir, max_seq_length)
        self.task_name = "JNLPBA-IOBES"
        self.map = {'O': 0, 'B-cell_line': 1, 'I-cell_line': 2, 'E-cell_line': 3, 'S-cell_line': 4,
                    'B-protein': 5, 'I-protein': 6, 'E-protein': 7, 'S-protein': 8,
                    'B-RNA': 9, 'I-RNA': 10, 'E-RNA': 11, 'S-RNA': 12,
                    'B-DNA': 13, 'I-DNA': 14, 'E-DNA': 15, 'S-DNA': 16,
                    'B-cell_type': 17, 'I-cell_type': 18, 'E-cell_type': 19, 'S-cell_type': 20, 'X': 21}

        self.classes = len(self.map) - 1
        self.decode_map = {value: key for key, value in self.map.items()}
        self.create_tfrecord()


class linnaeusProcessor(DataProcessor):
    def __init__(self, sp, lower, input_dir, cache_dir, max_seq_length):
        super(linnaeusProcessor, self).__init__(sp, lower, input_dir, cache_dir, max_seq_length)
        self.task_name = "linnaeus-IOBES"
        self.map = {'O': 0, 'B-Species': 1, 'I-Species': 2, 'E-Species': 3, 'S-Species': 4, 'X': 5}

        self.classes = len(self.map) - 1
        self.decode_map = {value: key for key, value in self.map.items()}
        self.create_tfrecord()


class NCBIProcessor(DataProcessor):
    def __init__(self, sp, lower, input_dir, cache_dir, max_seq_length):
        super(NCBIProcessor, self).__init__(sp, lower, input_dir, cache_dir, max_seq_length)
        self.task_name = "NCBI-disease-IOBES"
        self.map = {'O': 0, 'B-Disease': 1, 'I-Disease': 2, 'E-Disease': 3, 'S-Disease': 4, 'X': 5}

        self.classes = len(self.map) - 1
        self.decode_map = {value: key for key, value in self.map.items()}
        self.create_tfrecord()

# Conll2003Processor("tfhub-module/xlnet_cased_L-24_H-1024_A-16/spiece.model", True, "data", "cache", 512)
# BC5CDRProcessor("tfhub-module/xlnet_cased_L-24_H-1024_A-16/spiece.model", True, "data", "cache", 512)
# AnatEMProcessor("tfhub-module/xlnet_cased_L-24_H-1024_A-16/spiece.model", True, "data", "cache", 512)
# BC2GMProcessor("tfhub-module/xlnet_cased_L-24_H-1024_A-16/spiece.model", True, "data", "cache", 512)
# BC4CHEMDProcessor("tfhub-module/xlnet_cased_L-24_H-1024_A-16/spiece.model", True, "data", "cache", 512)
# BioNLP09Processor("tfhub-module/xlnet_cased_L-24_H-1024_A-16/spiece.model", True, "data", "cache", 512)
# BioNLP11IDProcessor("tfhub-module/xlnet_cased_L-24_H-1024_A-16/spiece.model", True, "data", "cache", 512)
# BioNLP11EPIProcessor("tfhub-module/xlnet_cased_L-24_H-1024_A-16/spiece.model", True, "data", "cache", 512)
# BioNLP13CGProcessor("tfhub-module/xlnet_cased_L-24_H-1024_A-16/spiece.model", True, "data", "cache", 512)
# BioNLP13GEProcessor("tfhub-module/xlnet_cased_L-24_H-1024_A-16/spiece.model", True, "data", "cache", 512)
# BioNLP13PCProcessor("tfhub-module/xlnet_cased_L-24_H-1024_A-16/spiece.model", True, "data", "cache", 512)
# CRAFTProcessor("tfhub-module/xlnet_cased_L-24_H-1024_A-16/spiece.model", True, "data", "cache", 512)
# ExPTMProcessor("tfhub-module/xlnet_cased_L-24_H-1024_A-16/spiece.model", True, "data", "cache", 512)
# JNLPBAProcessor("tfhub-module/xlnet_cased_L-24_H-1024_A-16/spiece.model", True, "data", "cache", 512)
# linnaeusProcessor("tfhub-module/xlnet_cased_L-24_H-1024_A-16/spiece.model", True, "data", "cache", 512)
# NCBIProcessor("tfhub-module/xlnet_cased_L-24_H-1024_A-16/spiece.model", True, "data", "cache", 512)
# f.close()
