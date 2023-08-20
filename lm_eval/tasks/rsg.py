# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""
import numpy as np
import sklearn
import transformers.data.metrics.squad_metrics as squad_metrics
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, acc_all, f1_per_dataset, metric_max_over_ground_truths, matthews_corrcoef, yesno_rus
from lm_eval.utils import general_detokenize


# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""


# TODO: Replace `NewTask` with the name of your Task.
class DaNetQA(Task):
    VERSION = 0
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "RussianNLP/russian_super_glue"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = "danetqa"

    def has_training_docs(self):
        # TODO: Fill in the return with `True` if the Task has training data; else `False`.
        return True

    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return True

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return False

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # TODO: Return the training document generator from `self.dataset`.
                # If you need to process the data, `map` over the documents with
                # the custom processing function, `self._process_doc`. E.g.
                # `map(self._process_doc, self.dataset["validation"])`
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            # TODO: Return the validation document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["validation"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        return f"{doc['passage']}\nВопрос: {doc['question']}?\nОтвет:"
    
    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["passage"]

    def doc_to_target(self, doc):
        # TODO: Fill in the `target` ("gold answer") variable.
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.
        target = ""
        return " " + yesno_rus(doc["label"])

    def construct_requests(self, doc, ctx):

        ll_yes, _ = rf.loglikelihood(ctx, " да")
        ll_no, _ = rf.loglikelihood(ctx, " нет")

        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = doc["label"]

        acc = 1.0 if (ll_yes > ll_no) == gold else 0.0

        return {"acc": acc}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}

class RuCoS(Task):
    VERSION = 0
    DATASET_PATH = "RussianNLP/russian_super_glue"
    DATASET_NAME = "rucos"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        # In RuCoS, each doc manifests multiple "examples" in the context of few shot example packing.
        # Each doc consists of multiple answer candidates, each of which is scored yes/no.
        if self.has_training_docs():
            self._training_docs = []
            for doc in self.dataset["train"]:
                self._training_docs.append(self._process_doc(doc))
        return self._training_docs

    def validation_docs(self):
        # See: training_docs
        for doc in self.dataset["validation"]:
            yield self._process_doc(doc)

    def test_docs(self):
        for doc in self.dataset["test"]:
            yield self._process_doc(doc)

    @classmethod
    def _process_doc(cls, doc):
        return {
                "passage": doc["passage"],
                "query": doc["query"],
                "entities": sorted(list(set(doc["entities"]))), #doc["entities"], #
                "answers": sorted(list(set(doc["answers"]))), #["some answer"], #
        }

    def doc_to_text(self, doc):
        initial_text, *highlights = doc["passage"].strip().split("\n@highlight\n")
        text = initial_text + "\n\n"
        for highlight in highlights:
            text += f"  - {highlight}.\n"
        return text
    
    @classmethod
    def format_answer(cls, query, entity):
        return f"  - {query}".replace("@placeholder", entity)

    def doc_to_target(self, doc):
        # We only output the first correct entity in a doc
        return self.format_answer(query=doc["query"], entity=doc["answers"][0])
    
    def construct_requests(self, doc, ctx):
        requests = [
            rf.loglikelihood(ctx, self.format_answer(query=doc["query"], entity=entity))
            for entity in doc["entities"]
        ]
        return requests

    def process_results(self, doc, results):
        # ReCoRD's evaluation is actually deceptively simple:
        # - Pick the maximum likelihood prediction entity
        # - Evaluate the accuracy and token F1 PER EXAMPLE
        # - Average over all examples
        max_idx = np.argmax(np.array([result[0] for result in results]))

        prediction = doc["entities"][max_idx]
        gold_label_set = doc["answers"]
        f1 = metric_max_over_ground_truths(
            squad_metrics.compute_f1, prediction, gold_label_set
        )
        em = metric_max_over_ground_truths(
            squad_metrics.compute_exact, prediction, gold_label_set
        )

        return {
            "f1": f1,
            "em": em,
        }

    def higher_is_better(self):
        return {
            "f1": True,
            "em": True,
        }

    def aggregation(self):
        return {
            "f1": mean,
            "em": mean,
        }
    
class RusWinogradSchemaChallenge(Task):
    VERSION = 0
    # A Winograd schema is a pair of sentences that differ in only one or two words and that contain an ambiguity that is resolved in opposite ways in the two sentences and requires the use of world knowledge and reasoning for its resolution.
    # RWSD ~ WSC from SuperGLUE
    DATASET_PATH = "RussianNLP/russian_super_glue"
    DATASET_NAME = "rwsd"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                # GPT-3 Paper's format only uses positive examples for fewshot "training"
                self._training_docs = [
                    doc for doc in self.dataset["train"] if doc["label"]
                ]
            return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]
    
    def doc_to_text(self, doc):
        raw_passage = doc["text"]
        pre = " ".join(raw_passage.split()[: doc["span2_index"]])
        post = raw_passage[len(pre) + len(doc["span2_text"]) + 1 :]
        passage = general_detokenize(pre + " *{}*".format(doc["span2_text"]) + post)
        noun = doc["span1_text"]
        pronoun = doc["span2_text"]
        text = (
            f"Текст: {passage}\n"
            + f'Вопрос: в тексте, приведенном выше, правда ли, что "*{pronoun}*" относится к "*{noun}*"?\n'
            + "Ответ:"
        )
        return text

    def doc_to_target(self, doc):
        return " " + yesno_rus(doc["label"])

    def construct_requests(self, doc, ctx):

        ll_yes, _ = rf.loglikelihood(ctx, " да")
        ll_no, _ = rf.loglikelihood(ctx, " нет")

        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = doc["label"]

        acc = 1.0 if (ll_yes > ll_no) == gold else 0.0

        return {"acc": acc}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}
    

class RUSSE(Task):
    ## Russian word-in-context task
    VERSION = 0
    DATASET_PATH = "RussianNLP/russian_super_glue"
    DATASET_NAME = "russe"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        return (
            "Предложение 1: {}\nПредложение 2: {}\nВопрос: Слово '{}' имеет одно и то же значение"
            " в двух предложениях выше?\nОтвет:".format(
                doc["sentence1"],
                doc["sentence2"],
                doc["sentence1"][doc["start1"] : doc["end1"]],
            )
        )

    def doc_to_target(self, doc):
        return " {}".format({0: "нет", 1: "да", -1: "неизвестно"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_yes, _ = rf.loglikelihood(ctx, " да")
        ll_no, _ = rf.loglikelihood(ctx, " нет")

        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = doc["label"]

        acc = 1.0 if (ll_yes > ll_no) == gold else 0.0

        return {"acc": acc}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}
    
class TERRa(Task):
    VERSION = 0
    DATASET_PATH = "RussianNLP/russian_super_glue"
    DATASET_NAME = "terra"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        return "{}\nВопрос: {} - это правда или ложь?\nОтвет:".format(
            doc["premise"],
            doc["hypothesis"],
        )

    def doc_to_target(self, doc):
        # 0 = entailment
        # 1 = not_entailment
        return " {}".format({0: "правда", 1: "ложь", -1: "unknown"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " правда")
        ll_false, _ = rf.loglikelihood(ctx, " ложь")
        return ll_true, ll_false

    def process_results(self, doc, results):
        ll_true, ll_false = results
        pred = ll_false > ll_true
        gold = doc["label"]
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}
    
class MuSeRC(Task):
    VERSION = 0
    DATASET_PATH = "RussianNLP/russian_super_glue"
    DATASET_NAME = "muserc"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        return f"{doc['paragraph']}\nВопрос:{doc['question']}\nОтвет:"

    def doc_to_target(self, doc):
        return " " + self.format_answer(answer=doc["answer"], label=doc["label"])

    @staticmethod
    def format_answer(answer, label):
        label_str = "да" if label else "нет"
        return f"{answer}\nЭто правильный ответ на вопрос? {label_str}"

    def construct_requests(self, doc, ctx):
        true_choice = self.format_answer(answer=doc["answer"], label=True)
        false_choice = self.format_answer(answer=doc["answer"], label=False)

        ll_true_choice, _ = rf.loglikelihood(ctx, f" {true_choice}")
        ll_false_choice, _ = rf.loglikelihood(ctx, f" {false_choice}")

        return ll_true_choice, ll_false_choice

    def process_results(self, doc, results):
        ll_true_choice, ll_false_choice = results
        pred = ll_true_choice > ll_false_choice
        return {"acc": (pred, doc),
                "f1": (pred, doc)}

    def higher_is_better(self):
        return {"acc": True,
                "f1": True}

    def aggregation(self):
        return {"acc": acc_all,
                "f1": f1_per_dataset}
    
class PARus(Task):
    VERSION = 0
    DATASET_PATH = "RussianNLP/russian_super_glue"
    DATASET_NAME = "parus"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        # Drop the period
        connector = {
            "cause": "потому что",
            "effect": "и тогда",
        }[doc["question"]]
        return doc["premise"].strip()[:-1] + f", {connector}"

    def doc_to_target(self, doc):
        correct_choice = doc["choice1"] if doc["label"] == 0 else doc["choice2"]
        # Connect the sentences
        return " " + self.convert_choice(correct_choice)

    def construct_requests(self, doc, ctx):
        choice1 = " " + self.convert_choice(doc["choice1"])
        choice2 = " " + self.convert_choice(doc["choice2"])

        ll_choice1, _ = rf.loglikelihood(ctx, choice1)
        ll_choice2, _ = rf.loglikelihood(ctx, choice2)

        return ll_choice1, ll_choice2

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        acc = 1.0 if pred == gold else 0.0

        return {"acc": acc}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}

    @staticmethod
    def convert_choice(choice):
        return choice[0].lower() + choice[1:]
    
class RussianCommitmentBank(Task):
    VERSION = 0
    DATASET_PATH = "RussianNLP/russian_super_glue"
    DATASET_NAME = "rcb"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        return "{}\nИз этого следует, что \"{}\"? \nОтвет:".format(
            doc["premise"],
            doc["hypothesis"],
        )

    def doc_to_target(self, doc):
        # True = entailment
        # False = contradiction
        # Neither = neutral
        return " {}".format({0: "да", 1: "нет", 2: "возможно", -1: "неизвестно"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " да")
        ll_false, _ = rf.loglikelihood(ctx, " нет")
        ll_neither, _ = rf.loglikelihood(ctx, " возможно")

        return ll_true, ll_false, ll_neither

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        acc = 1.0 if pred == gold else 0.0

        return {"acc": acc, "f1": (pred, gold)}

    def higher_is_better(self):
        return {"acc": True, "f1": True}

    @classmethod
    def cb_multi_fi(cls, items):
        preds, golds = zip(*items)
        preds = np.array(preds)
        golds = np.array(golds)
        f11 = sklearn.metrics.f1_score(y_true=golds == 0, y_pred=preds == 0)
        f12 = sklearn.metrics.f1_score(y_true=golds == 1, y_pred=preds == 1)
        f13 = sklearn.metrics.f1_score(y_true=golds == 2, y_pred=preds == 2)
        avg_f1 = mean([f11, f12, f13])
        return avg_f1

    def aggregation(self):
        return {
            "acc": mean,
            "f1": self.cb_multi_fi,
        }

class LiDiRus(Task):
    VERSION = 0
    DATASET_PATH = "RussianNLP/russian_super_glue"
    DATASET_NAME = "lidirus"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return "{}\nИз этого следует, что \"{}\"?\nОтвет:".format(
            doc["sentence1"],
            doc["sentence2"],
        )

    def doc_to_target(self, doc):
        # 0 = entailment
        # 1 = not_entailment
        return " {}".format({0: "да", 1: "нет"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " да")
        ll_false, _ = rf.loglikelihood(ctx, " нет")
        return ll_true, ll_false

    def process_results(self, doc, results):
        ll_true, ll_false = results
        pred = ll_true > ll_false
        pred_res = 0 if pred else 1
        gold = doc["label"]
        return {"mcc": (gold, pred_res)}

    def higher_is_better(self):
        return {"mcc": True}

    def aggregation(self):
        return {"mcc": matthews_corrcoef}