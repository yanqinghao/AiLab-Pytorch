import logging
import torch
import io
import sys
import six
import csv
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from tqdm import tqdm
from collections import Counter


def build_vocab_from_iterator(iterator):
    """
    Build a Vocab from an iterator.

    Arguments:
        iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.
    """

    counter = Counter()
    with tqdm(unit_scale=0, unit="lines") as t:
        for _, tokens in iterator:
            counter.update(tokens)
            t.update(1)
    word_vocab = Vocab(counter)
    return word_vocab


def unicode_csv_reader(unicode_csv_data, **kwargs):
    r"""Since the standard csv library does not handle unicode in Python 2, we need a wrapper.
    Borrowed and slightly modified from the Python docs:
    https://docs.python.org/2/library/csv.html#csv-examples

    Arguments:
        unicode_csv_data: unicode csv data (see example below)

    Examples:
        >>> from torchtext.utils import unicode_csv_reader
        >>> import io
        >>> with io.open(data_path, encoding="utf8") as f:
        >>>     reader = unicode_csv_reader(f)

    """

    # Fix field larger than field limit error
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)

    csv.field_size_limit(sys.maxsize)
    if kwargs:
        columns = []
        if "label" in kwargs.keys() and kwargs["label"]:
            columns += [kwargs["label"]]
        if "features" in kwargs.keys() and kwargs["features"]:
            columns += kwargs["features"]
        if six.PY2:
            # csv.py doesn't do Unicode; encode temporarily as UTF-8:
            csv_reader = csv.DictReader(utf_8_encoder(unicode_csv_data))
            for row in csv_reader:
                # decode UTF-8 back to Unicode, cell by cell:
                yield [row[col].decode("utf-8") for col in columns]
        else:
            for line in csv.DictReader(unicode_csv_data):
                yield [line[col] for col in columns]
    else:
        if six.PY2:
            # csv.py doesn't do Unicode; encode temporarily as UTF-8:
            csv_reader = csv.reader(utf_8_encoder(unicode_csv_data))
            for row in csv_reader:
                # decode UTF-8 back to Unicode, cell by cell:
                yield [cell.decode("utf-8") for cell in row]
        else:
            for line in csv.reader(unicode_csv_data):
                yield line


def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode("utf-8")


FILEPATH = "common/data/sentiment_analysis/"
URLS = {
    "AG_NEWS": "ag_news_csv.tar.gz",
    "SogouNews": "amazon_review_full_csv.tar.gz",
    "DBpedia": "amazon_review_polarity_csv.tar.gz",
    "YelpReviewPolarity": "dbpedia_csv.tar.gz",
    "YelpReviewFull": "sogou_news_csv.tar.gz",
    "YahooAnswers": "yahoo_answers_csv.tar.gz",
    "AmazonReviewPolarity": "yelp_review_full_csv.tar.gz",
    "AmazonReviewFull": "yelp_review_polarity_csv.tar.gz",
}


def _csv_iterator(data_path, ngrams, yield_cls=False, **kwargs):
    tokenizer = get_tokenizer("basic_english")
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f, **kwargs)
        for row in reader:
            if not kwargs or ("label" in kwargs.keys() and kwargs["label"]):
                tokens = " ".join(row[1:])
            else:
                tokens = " ".join(row)
            tokens = tokenizer(tokens)
            if yield_cls:
                yield int(row[0]) - 1, ngrams_iterator(tokens, ngrams)
            else:
                yield None, ngrams_iterator(tokens, ngrams)


def _create_data_from_iterator(vocab, iterator, include_unk):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit="lines") as t:
        for cls, tokens in iterator:
            if include_unk:
                tokens = torch.tensor([vocab[token] for token in tokens])
            else:
                token_ids = list(
                    filter(
                        lambda x: x is not Vocab.UNK, [vocab[token] for token in tokens]
                    )
                )
                tokens = torch.tensor(token_ids)
            if len(tokens) == 0:
                logging.info("Row contains no tokens.")
            data.append((cls, tokens))
            labels.append(cls)
            t.update(1)
    return data, set(labels)


class TextClassificationDataset(torch.utils.data.Dataset):
    """Defines an abstract text classification datasets.
       Currently, we only support the following datasets:

             - AG_NEWS
             - SogouNews
             - DBpedia
             - YelpReviewPolarity
             - YelpReviewFull
             - YahooAnswers
             - AmazonReviewPolarity
             - AmazonReviewFull

    """

    def __init__(self, vocab, data, labels):
        """Initiate text-classification dataset.

        Arguments:
            vocab: Vocabulary object used for dataset.
            data: a list of label/tokens tuple. tokens are a tensor after
                numericalizing the string tokens. label is an integer.
                [(label1, tokens1), (label2, tokens2), (label2, tokens3)]
            label: a set of the labels.
                {label1, label2}

        Examples:
            See the examples in examples/text_classification/

        """

        super(TextClassificationDataset, self).__init__()
        self._data = data
        self._labels = labels
        self._vocab = vocab

    def __getitem__(self, i):
        return self._data[i][0], self._data[i][1], i

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_labels(self):
        return self._labels

    def get_vocab(self):
        return self._vocab


def _setup_datasets(
    dataset_name, extracted_files, root=".data", ngrams=2, vocab=None, include_unk=False
):
    # if os.path.exists(root):
    #     os.makedirs(root)
    # shutil.copy(
    #     os.path.join(file_path, URLS[dataset_name]),
    #     "{}/{}".format(root, URLS[dataset_name]),
    # )
    # dataset_tar = "{}/{}".format(root, URLS[dataset_name])
    # dataset_tar = downloadTextDataset(dataset_name, storageType, root=root)
    # dataset_tar = download_from_url(URLS[dataset_name], root=root)
    # extracted_files = extract_archive(dataset_tar)

    for fname in extracted_files:
        if fname.endswith("train.csv"):
            train_csv_path = fname
        if fname.endswith("test.csv"):
            test_csv_path = fname

    if vocab is None:
        logging.info("Building Vocab based on {}".format(train_csv_path))
        vocab = build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info("Vocab has {} entries".format(len(vocab)))
    logging.info("Creating training data")
    train_data, train_labels = _create_data_from_iterator(
        vocab, _csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk
    )
    logging.info("Creating testing data")
    test_data, test_labels = _create_data_from_iterator(
        vocab, _csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk
    )
    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (
        TextClassificationDataset(vocab, train_data, train_labels),
        TextClassificationDataset(vocab, test_data, test_labels),
    )


def AG_NEWS(*args, **kwargs):
    """ Defines AG_NEWS datasets.
        The labels includes:
            - 1 : World
            - 2 : Sports
            - 3 : Business
            - 4 : Sci/Tech

    Create supervised learning dataset: AG_NEWS

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.AG_NEWS(ngrams=3)

    """

    return _setup_datasets(*(("AG_NEWS",) + args), **kwargs)


def SogouNews(*args, **kwargs):
    """ Defines SogouNews datasets.
        The labels includes:
            - 1 : Sports
            - 2 : Finance
            - 3 : Entertainment
            - 4 : Automobile
            - 5 : Technology

    Create supervised learning dataset: SogouNews

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.SogouNews(ngrams=3)

    """

    return _setup_datasets(*(("SogouNews",) + args), **kwargs)


def DBpedia(*args, **kwargs):
    """ Defines DBpedia datasets.
        The labels includes:
            - 1 : Company
            - 2 : EducationalInstitution
            - 3 : Artist
            - 4 : Athlete
            - 5 : OfficeHolder
            - 6 : MeanOfTransportation
            - 7 : Building
            - 8 : NaturalPlace
            - 9 : Village
            - 10 : Animal
            - 11 : Plant
            - 12 : Album
            - 13 : Film
            - 14 : WrittenWork

    Create supervised learning dataset: DBpedia

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.DBpedia(ngrams=3)

    """

    return _setup_datasets(*(("DBpedia",) + args), **kwargs)


def YelpReviewPolarity(*args, **kwargs):
    """ Defines YelpReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity.
            - 2 : Positive polarity.

    Create supervised learning dataset: YelpReviewPolarity

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YelpReviewPolarity(ngrams=3)

    """

    return _setup_datasets(*(("YelpReviewPolarity",) + args), **kwargs)


def YelpReviewFull(*args, **kwargs):
    """ Defines YelpReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended).

    Create supervised learning dataset: YelpReviewFull

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YelpReviewFull(ngrams=3)

    """

    return _setup_datasets(*(("YelpReviewFull",) + args), **kwargs)


def YahooAnswers(*args, **kwargs):
    """ Defines YahooAnswers datasets.
        The labels includes:
            - 1 : Society & Culture
            - 2 : Science & Mathematics
            - 3 : Health
            - 4 : Education & Reference
            - 5 : Computers & Internet
            - 6 : Sports
            - 7 : Business & Finance
            - 8 : Entertainment & Music
            - 9 : Family & Relationships
            - 10 : Politics & Government

    Create supervised learning dataset: YahooAnswers

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.YahooAnswers(ngrams=3)

    """

    return _setup_datasets(*(("YahooAnswers",) + args), **kwargs)


def AmazonReviewPolarity(*args, **kwargs):
    """ Defines AmazonReviewPolarity datasets.
        The labels includes:
            - 1 : Negative polarity
            - 2 : Positive polarity

    Create supervised learning dataset: AmazonReviewPolarity

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the datasets are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
       >>> train_dataset, test_dataset = torchtext.datasets.AmazonReviewPolarity(ngrams=3)

    """

    return _setup_datasets(*(("AmazonReviewPolarity",) + args), **kwargs)


def AmazonReviewFull(*args, **kwargs):
    """ Defines AmazonReviewFull datasets.
        The labels includes:
            1 - 5 : rating classes (5 is highly recommended)

    Create supervised learning dataset: AmazonReviewFull

    Separately returns the training and test dataset

    Arguments:
        root: Directory where the dataset are saved. Default: ".data"
        ngrams: a contiguous sequence of n items from s string text.
            Default: 1
        vocab: Vocabulary used for dataset. If None, it will generate a new
            vocabulary based on the train data set.
        include_unk: include unknown token in the data (Default: False)

    Examples:
        >>> train_dataset, test_dataset = torchtext.datasets.AmazonReviewFull(ngrams=3)

    """

    return _setup_datasets(*(("AmazonReviewFull",) + args), **kwargs)


class TextClassificationPredictDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, features, label=None):
        """
        Initiate text-classification dataset.
        """
        super(TextClassificationPredictDataset, self).__init__()
        self.csv_path = csv_path
        self._features = features
        self._label = label
        self._data = None
        self._labels = None
        self._vocab = None
        self._ngrams = None

    def __getitem__(self, i):
        return self._data[i][0], self._data[i][1], i

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def set_vocab(self, vocab):
        self._vocab = vocab

    def set_ngrams(self, ngrams):
        self._ngrams = ngrams

    def set_data(self):
        include_unk = False
        yield_cls_label = False
        if self._label:
            yield_cls_label = True
        params = {"features": self._features, "label": self._label}
        self._data, self._labels = _create_data_from_iterator(
            self._vocab,
            _csv_iterator(
                self.csv_path, self._ngrams, yield_cls=yield_cls_label, **params
            ),
            include_unk,
        )

    def get_labels(self):
        return self._labels

    def get_vocab(self):
        return self._vocab


DATASETS = {
    "AG_NEWS": AG_NEWS,
    "SogouNews": SogouNews,
    "DBpedia": DBpedia,
    "YelpReviewPolarity": YelpReviewPolarity,
    "YelpReviewFull": YelpReviewFull,
    "YahooAnswers": YahooAnswers,
    "AmazonReviewPolarity": AmazonReviewPolarity,
    "AmazonReviewFull": AmazonReviewFull,
    "PRED_Data": TextClassificationPredictDataset,
}


LABELS = {
    "AG_NEWS": {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"},
    "SogouNews": {
        1: "Sports",
        2: "Finance",
        3: "Entertainment",
        4: "Automobile",
        5: "Technology",
    },
    "DBpedia": {
        1: "Company",
        2: "EducationalInstitution",
        3: "Artist",
        4: "Athlete",
        5: "OfficeHolder",
        6: "MeanOfTransportation",
        7: "Building",
        8: "NaturalPlace",
        9: "Village",
        10: "Animal",
        11: "Plant",
        12: "Album",
        13: "Film",
        14: "WrittenWork",
    },
    "YelpReviewPolarity": {1: "Negative polarity", 2: "Positive polarity"},
    "YelpReviewFull": {
        1: "score 1",
        2: "score 2",
        3: "score 3",
        4: "score 4",
        5: "score 5",
    },
    "YahooAnswers": {
        1: "Society & Culture",
        2: "Science & Mathematics",
        3: "Health",
        4: "Education & Reference",
        5: "Computers & Internet",
        6: "Sports",
        7: "Business & Finance",
        8: "Entertainment & Music",
        9: "Family & Relationships",
        10: "Politics & Government",
    },
    "AmazonReviewPolarity": {1: "Negative polarity", 2: "Positive polarity"},
    "AmazonReviewFull": {
        1: "score 1",
        2: "score 2",
        3: "score 3",
        4: "score 4",
        5: "score 5",
    },
}

